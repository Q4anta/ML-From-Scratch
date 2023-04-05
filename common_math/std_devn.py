import numpy as np
def std_devn(data):
    return np.sqrt(((data - data.mean())**2).sum()/len(data))