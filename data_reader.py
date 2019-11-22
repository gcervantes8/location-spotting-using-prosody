'''
Author: Diego Aguirre
Last Modified: Sep 12, 2017
Description: Definition of a simple function that
            generates random data to test the implementation of
            Skanteze's LSTM network
'''
import scipy.io as sio
import numpy as np
from random import randint
#from sklearn.preprocessing import Normalizer

def get_data(mat_path):
    
    mat = sio.loadmat(mat_path)

    x_train = mat['trainFeatures']
    y_train = mat['trainLabels']
    x_test = mat['testFeatures']
    y_test = mat['testLabels']
    
    return np.float32(x_train), y_train, np.float32(x_test), y_test

def get_data_extra(mat_path):
    
    mat = sio.loadmat(mat_path)
    train_is_speaking = mat['trainIsSpeaking']
    test_is_speaking = mat['testIsSpeaking']
    
    train_audios = mat['trainAudioInfo']
    test_audios = mat['testAudioInfo']
#    train_audios = mat['trainAudioFile']
#    test_audios = mat['testAudioFile']
    return train_is_speaking, test_is_speaking, train_audios, test_audios

def get_data_extra_entities(mat_path):
    
    mat = sio.loadmat(mat_path)
    train_is_named_entity = mat['trainIsNamedEntity']
    test_is_named_entity = mat['testIsNamedEntity']
    
    train_is_function_word = mat['trainIsFunctionWord']
    test_is_function_word = mat['testIsFunctionWord']
    return train_is_named_entity, test_is_named_entity, train_is_function_word, test_is_function_word

def get_loc_sample_indices(y_labels):
    indices_with_loc = []
    
    for i, y_sample in enumerate(y_labels):
        if np.any(y_sample.flatten()):
            indices_with_loc.append(i)
            
    return np.array(indices_with_loc)


def filter_data(x_train, y_train):
    filtered_x = []
    filtered_y = []
    no_loc_samples = 0
    for i, y_sample in enumerate(y_train):
        #If contains 1, then keep
        if np.any(y_sample.flatten()):
            filtered_x.append(x_train[i])
            filtered_y.append(y_sample)
        else:            
            no_loc_samples += 1
#            rng = randint(0, 10)
#            #Take 10% of data when label is 0
#            if rng == 5:                
#                filtered_x.append(x_train[i])
#                filtered_y.append(y_sample)
                
    print('Number samples with no locations', no_loc_samples)
    print('Number samples:', len(y_train))
    return np.array(filtered_x), np.array(filtered_y)

def filter_data_with_extra(x_train, y_train, is_speaking, audios):
    filtered_x = []
    filtered_y = []
    filtered_is_speaking = []
    filtered_audios = []
    no_loc_samples = 0
    for i, y_sample in enumerate(y_train):
        #If contains 1, then keep
        if np.any(y_sample.flatten()):
            filtered_x.append(x_train[i])
            filtered_y.append(y_sample)
            filtered_is_speaking.append(is_speaking[i])
            filtered_audios.append(audios[i])
        else:            
            no_loc_samples += 1
#            rng = randint(0, 10)
#            #Take 10% of data when label is 0
#            if rng == 5:                
#                filtered_x.append(x_train[i])
#                filtered_y.append(y_sample)
                
    print('Number samples with no locations', no_loc_samples)
    print('Number samples:', len(y_train))
    return np.array(filtered_x), np.array(filtered_y), np.array(filtered_is_speaking), np.array(filtered_audios)


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
    
            
