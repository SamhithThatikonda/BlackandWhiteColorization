--This is a project for colorization of gray scale images.
In our bw2color_image.py we are using the training model of https://richzhang.github.io/colorization/resources/colorful_eccv2016.pdf . We used the above training model to test the images as the training model which was trained on millions of images.

--We also constructed bw2color_video.py which captures either a video or live video feed from webcam in grayscale and tries to colorize it.

-- We also constructed our own training model in train.ipynb, which uses ResNet-18 and Convolution, following a regression approach. We trained our model on the MIT dataset which consists of 40,000 images http://places.csail.mit.edu/ .

-- Another extension was implementing a bilateral filter during post processing of the image which smoothens the output image and gives the resultant image clarity in the case of noisy images.

-- Lastly, we also made a website implementing our model to share with our friends and family!

1. bw2color_image.pyz

###How to run

import the following packages:

pip install opencv-python
pip install matplotlib
pip install numpy

The arguments which are required are as follows:
--image: This is the path of your grayscale image
--prototxt: It is a configuration file used to tell caffe how we want the network trained (example: model/colorization_deploy_v2.prototxt)
--model: caffe train produces a binary .caffemodel file which easily integrates trained models into data pipelines (example : model/colorization_release_v2.caffemodel)
--points: path to cluster center points (example: model/pts_in_hull.npy)

###command
```
> python bw2color_image.py --image images/e.jpg --prototxt model/colorization_deploy_v2.prototxt --model model/colorization_release_v2.caffemodel --points model/pts_in_hull.npy
```

2. bw2color_video.py

###How to run

import the following packages:

pip install opencv-python
pip install matplotlib
pip install numpy
pip install imutils

The arguments which are required are as follows:

--input: This is the path to your grayscale video (Note if you do not provide the input video then the webcam will start in grayscale and you will see colorization of that video feed.)
--prototxt: It is a configuration file used to tell caffe how we want the network trained (example: model/colorization_deploy_v2.prototxt)
--model: caffe train produces a binary .caffemodel file which easily integrates trained models into data pipelines (example : model/colorization_release_v2.caffemodel)
--points: path to cluster center points (example: model/pts_in_hull.npy)
--width: input width dimension of frame (default: 500)

##command for webcam
```
>python bw2color_video.py --prototxt model/colorization_deploy_v2.prototxt --model model/colorization_release_v2.caffemodel --points model/pts_in_hull.npy
```
##command for imput video
```
>python bw2color_video.py --prototxt model/colorization_deploy_v2.prototxt --model model/colorization_release_v2.caffemodel --points model/pts_in_hull.npy --input video/video_name.mp4
```

3. training_model.ipynb
###Getting Started

import the following packages:

pip install matplotlib
pip install numpy
pip install scikit-image
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
(Note torch vision should be 0.4 and up)

##Download the MIT dataset

wget http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz

##Unzip the dataset

tar -xzf testSetPlaces205_resize.tar.gz

**(Note: Place your dataset in proper folders)**

Dataset can be trained on a GPU or CPU just be careful that you are using only one thing
You can set your epochs for training the data accordingly

4. Website
To run the application follow the steps below.

## Getting Started

We	will	be	using	Flask	as
our	web	framework.Flask	is	a	Python-based	framework. If	you	do	not	have	Python	3	on	your	local	machine,	we	recommend	that	you	look	through	the	Python	downloads	page	(https://www.python.org/downloads/) and	install	Python	3 in	whatever	way	is	appropriate	for	your	machine. In	the	end,	you	should	be	able	to	enter

```
> python3 --version
```
and	see	a	version	number	of	3.3	or	higher.

### Installing
Next,	you	will	need	to	install	the	Flask	package	within	the	Python	setup. This	is	easily	done
by	entering:

```
> pip3 install Flask
```
## Open the application

To	test	your	setup,	please	download	the	ZIP	package	available	as	part	of	this	project.
The	directory, "website", includes	files	and
libraries	that	we	will	be	using.

**Before running the files, we open the "run.py" and set the path for 'UPLOAD_FOLDER' to the 'images' folder within the website folder.**

In	a terminal	window,	please	navigate	to	the "website" directory	and
enter	the	following	command:

```
> python3 run.py
* Running on http://127.0.0.1:8080/ (Press CTRL+C to quit)
* Restarting with stat
* Debugger is active!
...
```

This	command	runs	the	included	Python	file,	which	in	turns	starts	a	Flask	web	server	on	a local	address	and	port	number	(http://127.0.0.1:8080).	Now,	open	a	web	browser	(Chrome,	Safari,	Firefox,	or	any	browser),	and	point	it	to	this	address;	when	you	do	so,
Flask	will	“serve”	the	web	page	provided	and	show	you	a	page just like the one we included in our paper.

5. bilateral_filter.py
###command
```
> # python bilateral_filter.py --image images/e.jpg --prototxt model/colorization_deploy_v2.prototxt --model model/colorization_release_v2.caffemodel --points model/pts_in_hull.npy
```
