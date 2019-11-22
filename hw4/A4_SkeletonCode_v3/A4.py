import numpy as np
import utils
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas as pd
import random
import matplotlib.pyplot as plt



#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	df = pd.read_csv("cazi6864-TrainingData.csv")
	return df
def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
	instance_copy = instance
	
	for i in range(len(instance_copy)):
		rand = random.random()
		#random variable 

		if (rand < percent_distortion):
		# random distortion rate

			if (instance_copy[i] == 0):
				instance_copy[i] = 1

			elif(instance_copy[i] == 1):
				instance_copy[i] = 0

			#flip bits

	return instance_copy


class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])

	def addSinglePattern(self, p):
		#Update the hopfield matrix using the passed pattern
		for i in range(len(p)):
			for j in range(len(p)):
				#loop over the matrix's 		
				if(i != j):
					self.h[i][j] += (2 * p[i] - 1)*(2 * p[j] - 1)
					#simple submation
		
	def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix

		for i in patterns:
			self.addSinglePattern(i)


	def retrieve(self, inputPattern):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.

		changes = 1
		temp = 0

		while changes != 0:
			#only do interations we need to do
			for i in range(0, len(inputPattern)):
				changes = 0
				temp = 0
				
				for j in range(0, len(inputPattern)):
					if(j != i):
						temp = (self.h[i][j] * inputPattern[i]) + temp
						#update weight

				if (temp >= 0 and inputPattern[i] == 0):
					inputPattern[i] = 1
					changes += 1

				elif(inputPattern[i] == 1):
					inputPattern[i] = 0
					changes += 1

		return inputPattern
				
						




	def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'


		input_array = np.array(inputPattern)
		five_array = np.array(five)
		two_array = np.array(two)

		five_distace = sum((five_array-input_array)**2)
		two_distance = sum((two_array-input_array)**2)
		#calculate distance

		if (five_distace > two_distance):
			return "two"
		elif(five_distace < two_distance):
			return "five"
		else:
			return "unknown"
		#return result





if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)

	#utils.visualize(five)
	#utils.visualize(two)

	
	#hopfieldNet.fit(patterns)

	hopfieldNet.addSinglePattern(two)
	hopfieldNet.addSinglePattern(five)

	
	X_train = np.array([two, five])
	y_train = np.array(["two", "five"])
	#training data

	df = pd.read_csv("cazi6864-TrainingData.csv")
	#using pandas for this

	Label = "label"
	Features = ["r00","r01","r02","r03","r04","r10","r11","r12","r13","r14","r20","r21","r22","r23","r24","r30","r31","r32","r33","r34","r40","r41","r42","r43","r44"]
	#features and labels	

	X_test, y_test = np.array(df[Features]), np.array(df[Label])
	 

	for count, elem in enumerate(X_test):
		if count <= 3:
			print("Label: two")
		else:
			print("Label: five")

		print("Classified as: ",hopfieldNet.classify(elem), "\n")



	# ----- Time for SKlearn to try ------
	# ------------- Part 3 ---------------

	clf = MLPClassifier()
	clf.fit(X_train, y_train)
	#fit training data

	y_predict = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, y_predict)
	#using accuracy score metric for testing
	print("MLP Classifier")
	print("accuracy score = ", score)


	# ------ Code to run distortion ------
	# ------------- Part 4 ---------------
	hopfield = HopfieldNetwork(25)

	hopfield.addSinglePattern(two)
	hopfield.addSinglePattern(five)


	rates = np.arange(0, .51, .01)

	accuracy_hopfield = []
	accuracy_sklearn = []


	for rate in rates:
		# we don't actually have to train more than the original time in the previous questions. This increases preformance.
		# instead, my distort input function returns a new copy of the previous pattern so it doesnt overwrite the original.
	

		results_hopfield = []
		trueResults = []
		distorted_X_test = []

		for elem in X_test:
			#create our new array of distorted inputs
			distorted_X_test.append(distort_input(elem, rate))

		for elem in distorted_X_test:
			#need to classify each element.

			result = hopfield.classify(elem)
			#predict with classifier I built
			results_hopfield.append(result)


		y_predict = clf.predict(distorted_X_test)
		#predict using Sklearn


		accuracy_hopfield_temp = 0
		accuracy_sklearn_temp = 0

		for count, element in enumerate(y_test):
			#calulate accuracy
			if (results_hopfield[count] == element):
				accuracy_hopfield_temp += 1
			if (y_predict[count] == element):
				accuracy_sklearn_temp +=  1
		
		
		accuracy_hopfield_temp = accuracy_hopfield_temp/len(y_test)
		accuracy_sklearn_temp = accuracy_sklearn_temp/len(y_test)

		accuracy_hopfield.append(accuracy_hopfield_temp)
		accuracy_sklearn.append(accuracy_sklearn_temp)


	'''
	plt.plot(rates, accuracy_hopfield, color="red")
	plt.plot(rates, accuracy_sklearn, color="blue")
	plt.legend(('Self Built Hopfield', 'SKlearn MLP'),loc='upper right')
	plt.xlabel("Distortion Rates")
	plt.ylabel("Accuracy (Percentage)")

	
	
	#plt.show()
	'''
	# ------ Varing layers in MLP ------
	# ------------- Part 5---------------




	Mlp_One_Layer = MLPClassifier(solver="lbfgs")
	Mlp_Two_Layer = MLPClassifier(hidden_layer_sizes=(100, 100),solver="lbfgs")
	Mlp_Three_Layer = MLPClassifier(hidden_layer_sizes=(100, 100, 100),solver="lbfgs")

	df = pd.read_csv("cazi6864-TrainingData.csv")
	df2 = pd.read_csv("NewInput.csv")
	#using pandas for this


	Label = "label"
	Features = ["r00","r01","r02","r03","r04","r10","r11","r12","r13","r14","r20","r21","r22","r23","r24","r30","r31","r32","r33","r34","r40","r41","r42","r43","r44"]
	#features and labels	

	X_test1, y_test1 = np.array(df[Features]), np.array(df[Label])
	X_test2, y_test2 = np.array(df2[Features]), np.array(df2[Label])



	X_train = np.concatenate((X_test1, X_test2))
	y_train = np.concatenate((y_test1, y_test2))
	#use all the data we have for training. 
	 

	Mlp_One_Layer.fit(X_train, y_train)
	Mlp_Two_Layer.fit(X_train, y_train)
	Mlp_Three_Layer.fit(X_train, y_train)
	#fit all our classifiers


	rates = np.arange(0, .51, .01)
	#distoriton rates


	accuracy_One_Layer = []
	accuracy_Two_Layer = []
	accuracy_Three_Layer = []



	for rate in rates:
		# we don't actually have to train more than the original time in the previous questions. This increases preformance.
		# instead, my distort input function returns a new copy of the previous pattern so it doesnt overwrite the original.
	
		distorted_X_test = []

		for elem in X_train:
			#create our new array of distorted inputs
			distorted_X_test.append(distort_input(elem, rate))


		y_predict_one_layer = Mlp_One_Layer.predict(distorted_X_test)
		y_predict_two_layer = Mlp_Two_Layer.predict(distorted_X_test)
		y_predict_three_layer = Mlp_Three_Layer.predict(distorted_X_test)
		#predict using Sklearn


		accuracy_One_Layer_temp = 0
		accuracy_Two_Layer_temp = 0
		accuracy_Three_Layer_temp = 0

		for count, element in enumerate(y_train):
			#calulate accuracy
			if (y_predict_one_layer[count] == element):
				accuracy_One_Layer_temp +=  1

			if (y_predict_two_layer[count] == element):
				accuracy_Two_Layer_temp +=  1

			if (y_predict_three_layer[count] == element):
				accuracy_Three_Layer_temp +=  1
		
		accuracy_One_Layer_temp = accuracy_One_Layer_temp/len(y_train)
		accuracy_Two_Layer_temp = accuracy_Two_Layer_temp/len(y_train)
		accuracy_Three_Layer_temp = accuracy_Three_Layer_temp/len(y_train)

		accuracy_One_Layer.append(accuracy_One_Layer_temp)
		accuracy_Two_Layer.append(accuracy_Two_Layer_temp)
		accuracy_Three_Layer.append(accuracy_Three_Layer_temp)

	'''
	plt.plot(rates, accuracy_One_Layer, color="blue")
	plt.plot(rates, accuracy_Two_Layer, color="green")
	plt.plot(rates, accuracy_Three_Layer, color="red")

	plt.legend(('MLP One Layer', 'MLP Two Layers','MLP Three Layers'),loc='upper right')
	plt.xlabel("Distortion Rates")
	plt.ylabel("Accuracy (Percentage)")

	#plt.show()
	'''

	print("end")





