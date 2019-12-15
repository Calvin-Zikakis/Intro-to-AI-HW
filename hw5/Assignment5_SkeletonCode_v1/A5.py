from __future__ import print_function
import keras
import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import os
from datetime import datetime as dt
from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from PIL import Image

def getVGGFeatures(fileList, layerName):
	#Initial Model Setup
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)
	
	#Confirm number of files passed is what was expected
	rArray = []
	print ("Number of Files Passed:")
	print(len(fileList))

	result_array = []

	for iPath in fileList:
		#Time Printing for Debug, you can comment this out if you wish
		#now = datetime.now()
		#current_time = now.strftime("%H:%M:%S")
		#print("Current Time =", current_time)
		try:
			#Read Image
			img = image.load_img(iPath)
			#Update user as to which image is being processed
			print("Getting Features " +iPath)
			#Get image ready for VGG16
			img = img.resize((224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			#Generate Features
			internalFeatures = model.predict(x)
			result_array.append(internalFeatures)
			rArray.append((iPath, internalFeatures))	

		except:
			print ("Failed "+ iPath)
	return rArray, result_array

def cropImage(image, x1, y1, x2, y2):
	area = (x1, y1, x2, y2)
	cropped_img = image.crop(area)
	return cropped_img



def standardizeImage(image, x, y):
	img = image.resize((x, y))
	return img




def preProcessImages(images):
	dir = os.path.dirname(os.path.realpath(__file__))
	for i in images:
		path = dir + '/uncropped/'
		img = Image.open(path + str(i))
		
		x = i.split(".")
		name = x[0]
		number = x[1]
		boundary = x[2]
		filetype = x[3]

		y = boundary.split(',')

		x1 = y[0]
		y1 = y[1]
		x2 = y[2]
		y2 = y[3]

		temp = img.copy()

		cropped = cropImage(temp, int(x1), int(y1), int(x2), int(y2))

		standardize = standardizeImage(cropped, 60, 60)
		standardize.save(dir + '/cropped/' + name + "." + number + "." + "cropped" +'.'+ filetype)



def visualizeWeight():
	#You can change these parameters if you need to
	utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
	in_shape = 60 * 60
	n_classes = 6

	# split data
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages, labels, test_size=0.20)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

	# one hot 
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	y_valid = keras.utils.to_categorical(y_valid, n_classes)


	model = Sequential()
	model.add(Dense(275, input_shape=(in_shape,)))
	model.add(Activation('relu'))

	model.add(Dense(72))
	model.add(Activation('relu'))

	model.add(Dropout(.1))
	
	model.add(Dense(6))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


	history = model.fit(X_train, y_train,batch_size=18, epochs=50,verbose=2,validation_data=(X_valid, y_valid))


	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


	return history


def trainFaceClassifier_VGG(extractedFeatures, labels):

	in_shape = 100352
	n_classes = 6

	# split data
	X_train, X_test, y_train, y_test = train_test_split(extractedFeatures, labels, test_size=0.20)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

	# one hot 
	y_train = keras.utils.to_categorical(y_train, n_classes)
	y_test = keras.utils.to_categorical(y_test, n_classes)
	y_valid = keras.utils.to_categorical(y_valid, n_classes)


	model = Sequential()
	model.add(Dense(100, input_shape=(in_shape,)))
	model.add(Activation('relu'))                            

	model.add(Dense(24))
	model.add(Activation('softmax'))

	model.add(Dense(6))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


	history = model.fit(X_train, y_train,batch_size=128, epochs=20,verbose=2,validation_data=(X_valid, y_valid))


	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


	return history


if __name__ == '__main__':

	names = ['bracco', 'butler', 'gilpin', 'harmon', 'radcliffe', 'vartan']
	

	#-------------- the code below preprocesses the images and stores them in a folder inside the uncropped folder. This take a while so it is commented out--------------
	
	path = 'uncropped'
	filenames = []
	folder = os.fsencode(path)

	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith(('.jpeg', '.jpg', '.png')):
			filenames.append(filename)
	filenames.sort()
	
	preProcessImages(filenames)
	

	#-------------- this section of the code pulls the names from the cropped folder--------------
	path = 'cropped'
	filenames = []
	folder = os.fsencode(path)

	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith(('.jpeg', '.jpg', '.png')):
			filenames.append(filename)

	filenames.sort()


	#--------------code below pulls image data, gray scales it, applys a label and flattens it--------------


	flattened_images = []
	labels = np.array([])
	#flattned image array and label array

	dir = os.path.dirname(os.path.realpath(__file__))
	path = dir + '/cropped/'


	for i in filenames:
		x = i.split(".")
		name = x[0]

		index = names.index(str(name))

		img = Image.open(path + str(i))
		img = img.convert('L')
		pix = np.array(img) / 255.0

		pix = pix.flatten()

		pix = list(pix)


		flattened_images.append(pix)
		labels = np.append(labels, index)


	
	flattened_images = np.array(flattened_images)
	#flattens images and creates labels

	#--------------MNIST Classifier--------------


	model = trainFaceClassifier(flattened_images, labels)


	#--------------VGG16 Classifier--------------
	filename_path = []
	for element in filenames:
		filename_path.append('cropped/'+ element)
	


	r_array, result = getVGGFeatures(filename_path,"block4_pool")
	result = np.asarray(result)


	result_list = []


	for i in range(len(filename_path)):
		result_list.append(result[i].flatten())


	result = np.array(result_list)



	model = trainFaceClassifier_VGG(result, labels)


