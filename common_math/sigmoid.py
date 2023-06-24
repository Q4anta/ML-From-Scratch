import numpy as np 

def sigmoid(x):
 return 1/(1 + np.exp(-x))

def sigmoidGrad(x):
    v = sigmoid(x)
    return v*(1-v)