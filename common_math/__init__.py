from .gini import gini
from .entropy import entropy
from .std_devn import std_devn
from .sigmoid import sigmoid, sigmoidGrad
from .relu import relu, reluGrad
from .log_loss import log_loss, log_loss_grad

activationMap = {
    'relu' : relu, 
    'sigmoid' : sigmoid, 
    'identity' : lambda x: x,
    }
activationGradMap = {
    'relu' : reluGrad, 
    'sigmoid' : sigmoidGrad, 
    'identity' : lambda x: 1.0,
    }