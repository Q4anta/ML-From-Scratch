import numpy as np

def log_loss(target, output):
    m = len(target)
    loss = -1./m*(np.dot(target, np.log(output)) + np.dot((1-target),np.log(1-output)))
    return loss

def log_loss_grad(target, output):
    m = len(target)
    grad = -1/m * (np.divide(target, output) - np.divide(1-target, 1-output))
    return grad
                  
                