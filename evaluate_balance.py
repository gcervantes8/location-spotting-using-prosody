# -*- coding: utf-8 -*-
"""
Created on ~

@author: Jerry C
"""

import numpy as np
from eval_metrics import threshold_evaluate, precision_recall_curve, evaluate

#If threshold is None then varies the threshold and gets the best one, if has value, then gets f1 for that
def evaluation_on_balanced(predictions, labels, is_speaking, threshold):
    predictions = predictions.flatten()
    labels = labels.flatten()
    is_speaking = is_speaking.flatten()
    
    
    location_indices = np.where(labels == 1)[0]
    
    is_speaking[location_indices] = 0 #Turns places with locations to 0
    is_speaking_indices = np.where(is_speaking == 1)[0] #Indices where they are speaking, doesn't have locations
    
    num_location_frames = len(location_indices)
    
    downsampled_is_speaking_indices = np.random.choice(is_speaking_indices, num_location_frames)
    
    
    balanced_indices = np.concatenate((location_indices, downsampled_is_speaking_indices))
    if threshold == None:
        precision, recall, threshold, f1 = threshold_evaluate(predictions[balanced_indices], labels[balanced_indices], True)
    else:
        precision, recall,  f1 = evaluate(predictions[balanced_indices], labels[balanced_indices], threshold)
    return precision, recall, threshold, f1
    
    #precision_recall_curve(predictions[balanced_indices], labels[balanced_indices])
    
    
    
    
