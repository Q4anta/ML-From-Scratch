def gini(data):
    df = data.value_counts(normalize=True)
    return 1 - (df**2).sum()