import scipy.io as sio
import numpy as np
from random import randint


def get_data(mat_path):
    
    mat = sio.loadmat(mat_path)

    x_train = mat['trainFeatures']
    y_train = mat['trainLabels']
    x_test = mat['testFeatures']
    y_test = mat['testLabels']
    
    return np.float32(x_train), y_train, np.float32(x_test), y_test

def get_loc_sample_indices(y_labels):
    indices_with_loc = []
    
    for i, y_sample in enumerate(y_labels):
        if np.any(y_sample.flatten()):
            indices_with_loc.append(i)
            
    return np.array(indices_with_loc)

#from sklearn.preprocessing import Normalizer

#def normalize_data(x, x_test):
#    n_timesteps = len(x[0])
#    n_features = len(x[0][0])
#    
#    x_reshaped = np.reshape(x, (-1, n_features))
#    x_test_reshaped = np.reshape(x_test, (-1, n_features))
#    transformer = Normalizer().fit(x_reshaped)
#    x = transformer.transform(x_reshaped)
#    x_test = transformer.transform(x_test_reshaped)
#    
#    x = np.reshape(x, (-1, n_timesteps, n_features))
#    x_test = np.reshape(x_test, (-1, n_timesteps, n_features))
#    return x, x_test
    
            
