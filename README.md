# caffeclassweightlayer
Weigh classes of a classification network in caffe differently. 

This is a hack for weighting different classes in a softmax classification differently in caffe. 

USAGE:

In your prototxt file containing the network, add this layer in the following way wherever necessary. 

It takes in a ```num_examples x channels x ....``` input, where ```num_examples``` is the number of examples in a batch, ```channels``` are the number of classes, ```....``` can be just a number for a single class output or an image w x h for image segmentation where each pixel is a class. \\ 

```
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
```

Make sure the file ```weightclass.py``` is placed in the ```$PYTHONPATH``` for caffe. Mostly, it should be ```$blah$/caffe/python/``` , where ```$blah$``` is wherever you set up the caffe folder during installation.
