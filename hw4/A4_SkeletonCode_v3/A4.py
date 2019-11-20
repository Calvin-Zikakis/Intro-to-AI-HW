import numpy as np
import utils


#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	print("TODO")

def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
    print("TODO")
    utils.raiseNotDefined()


class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])

	def addSinglePattern(self, p):
		#Update the hopfield matrix using the passed pattern
		for i in range(len(p):
			for j in range(len(p)):
				#loop over the matrix's 		
				if(i != j):
					self.h[i][j] += (2 * p[i] - 1)*(2 * p[j] - 1)
					#simple submation
		
	def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix

		for i in pattern:
			addSinglePattern(self, i)


	def retrieve(self, inputPattern):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.

		changes = 1
		temp = 0
		while changes != 0:
			for count_i, i in enumerate(self.h[0][:]):
				changes = 0
				for count_j in enumerate(self.h[:][0]):
					if (count_i != count_j):
						temp = np.sum,(i * j)
						



	def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'




if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)

	utils.visualize(five)
	utils.visualize(two)


	#hopfieldNet.fit(patterns)
