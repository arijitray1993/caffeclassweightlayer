# This is a hack for weighting different classes in a softmax classification differently in caffe. 
# Author : Arijit Arren Ray

# USAGE:
# In your prototxt file containing the network, add this layer in the following way wherever necessary. 
# It takes in a num_examples x channels x .... input, where num_examples is the number of examples in a batch, channels are the number of classes, .... can be just a number for a single class output or a image w x h for image segmentation where each pixel is a class.  
'''
layer {
  name: "PythonWeightlayer"
  type: "Python"
  bottom: "score" # this contains num_examples X channels X ... probabilities where each channel is the probability of each class. 
  top: "PythonWeightLayer"
  python_param {
    module: "weightclass"
    layer: "weightlayer"
    param_str: "'mult': [-1,0.63,-1.5]"  # this has the weights for each channel, the length of this MUST match the number of classes, in this case, three. Set multiplier as -1 if a class is to be not weighted. Otherwise, higher the weight, more the weight to the class. A weight lower than -1 makes it less important than others.  
  }
}

'''
# make sure this file is placed in the PYTHONPATH for caffe. Mostly, it should be $blah$/caffe/python/ , where $blah$ is wherever you set up the caffe folder during installation. 

import caffe
import numpy as np
import yaml

class weightlayer(caffe.Layer):
	
	def setup(self, bottom, top):
		if len(bottom) < 1:
			raise Exception("Need at least one layer to scale...")
		
		self.mult=yaml.load(self.param_str)['mult']
		if len(self.mult)!=bottom[0].data.shape[1]:
			raise Exception("Need a multiplier for each output class. Please input multiplier as -1 if not to be weighted.")
		for channel,m in enumerate(self.mult):
			print "Multipler for Channel "+str(channel)+" : "+str(-m) 
	#	print self.mult

	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].shape)

	def forward(self, bottom, top):
	#	top[0].reshape(*bottom[0].shape)
	#	multiplier=np.ones(bottom[0].shape[2])
	#	multiplier[self.layer]*=self.mult
	#	print(bottom[0].data.shape)
		
		multiplier=np.ones(bottom[0].data.shape)
		
		for channel,m in enumerate(self.mult):
			multiplier[:,channel]*=(-m)
		
		top[0].data[...] = bottom[0].data*(multiplier)
	
		self.multiplierg=multiplier
	#	print top[0].data.shape
		
	def backward(self, top, propagate_down, bottom):
		bottom[0].diff[...] = self.multiplierg * top[0].diff

