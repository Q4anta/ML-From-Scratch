def relu(x):
    return max([x,0])

def reluGrad(x):
    return 1.0 if x > 0 else 0
