import numpy as np
def entropy(data):
    df = data.value_counts(normalize=True)
    return - (df*df.apply(np.log2)).sum()