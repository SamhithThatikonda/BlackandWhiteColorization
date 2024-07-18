# to run the files
# python bw2color_image.py --image images/robin_williams.jpg --prototxt model/colorization_deploy_v2.prototxt --model model/colorization_release_v2.caffemodel --points model/pts_in_hull.npy

# import packages
import numpy as np
import argparse
import cv2
import matplotlib.image as img

# parse the arguments
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--image", type=str, required=True,help="path to input black and white image")
arg.add_argument("-p", "--prototxt", type=str, required=True,help="path to Caffe prototxt file")
arg.add_argument("-m", "--model", type=str, required=True,help="path to Caffe pre-trained model")
arg.add_argument("-c", "--points", type=str, required=True,help="path to cluster center points")
args = vars(arg.parse_args())

# load our serialized black and white colorizer model and cluster
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
pts = np.load(args["points"])

# add the cluster centers as 1x1 convolutions to the model
clas = net.getLayerId("class8_ab")
conv = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(clas).blobs = [pts.astype("float32")]
net.getLayer(conv).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# load the input image, scale the pixel intensities to the
# either 0 or 1, and then convert the image from to LAB space from BGR
image = cv2.imread(args["image"])
# scaled = image.astype("float32") / 255.0
scaled=img.imread(args["image"]).astype(np.float32)/255
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# perform mean centering
#image is resized to 224X224 and the channels are spilt to extract the 'L' channel
#mean centering is performed next
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# pass the L channel through the network which will *predict* the 'a'
# and 'b' channel values
'print("[INFO] colorizing image...")'
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# resize the predicted 'ab' volume to the same dimensions as our input image
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# grab the 'L' channel from the *original* input image 
# and concatenate the original 'L' channel with the
# predicted 'ab' channels
L = cv2.split(lab)[0]
color = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# convert the output image from the Lab color space to RGB, then
# clip any values that fall outside the range [0, 1]
color = cv2.cvtColor(color, cv2.COLOR_LAB2BGR)
color = np.clip(color, 0, 1)

# the current colorized image is represented as a floating point
# data type in the range [0, 1] -- let's convert to an unsigned
# 8-bit integer representation in the range [0, 255]
color = (255 * color).astype("uint8")

# show the original and output colorized images
cv2.imshow("Original", image)
cv2.imshow("Colorized", color)
cv2.waitKey(0)
