import numpy as np

def BIC(features):
    lda = np.log(features)
    lldas = np.round(np.log10(lda)*2)/2
    return lldas