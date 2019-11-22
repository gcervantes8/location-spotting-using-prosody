# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:09:48 2018

@author: Jerry C
"""

import numpy as np
from matplotlib.pyplot import plot

def threshold_evaluate(predictions, y_test, will_print):
    predictions = predictions.flatten()
    y_test = y_test.flatten()
    end_point = 1000 #1000
    thresholds = [x / end_point for x in range(0, end_point)]
    best_precision, best_recall, best_threshold, best_f1 = 0, 0, 0, 0
    for threshold in thresholds:
        precision, recall, f1 = evaluate(predictions, y_test, threshold)
        
        if f1 != 0:
            if will_print:
                print_results(precision, recall, f1, threshold)
        if f1 > best_f1:
            best_precision, best_recall, best_threshold, best_f1 = precision, recall, threshold, f1
            
    if will_print:
        print('Best results:')
        print_results(best_precision, best_recall, best_f1, best_threshold)
    return best_precision, best_recall, best_threshold, best_f1
        
def evaluate(predictions, y_test, threshold):
    predictions = predictions.flatten()
    y_test = y_test.flatten()
    bool_preds = predictions > threshold
    tp, fn, fp = find_confusion_matrix(bool_preds, y_test)
    precision, recall, f1_measure = get_metrics(tp, fn, fp)
    return precision, recall, f1_measure
		
def precision_recall_curve(predictions, y_test):
    predictions = predictions.flatten()
    y_test = y_test.flatten()
    end_point = 1000
    thresholds = [x / end_point for x in range(0, end_point)]
    precision_list = []
    recall_list = []
    
    for threshold in thresholds:
        precision, recall, f1 = evaluate(predictions, y_test, threshold)
        precision_list.append(precision)
        recall_list.append(recall)
        if f1 != 0:
            pass
#            print_results(precision, recall, f1_measure, threshold)
#        if f1_measure > best_f1:
#            best_precision, best_recall, best_threshold, best_f1 = precision, recall, threshold, f1_measure
#    print(recall_list)
#    print(precision_list)
    
    plot(thresholds, precision_list)

def print_results(precision, recall, f1_measure, threshold):
    print('Precision: %.4f, Recall: %.3f, f1: %.4f threshold %3f' % (precision, recall, f1_measure, threshold))

def random_baseline(y_test):
    y_test = y_test.flatten()
    baseline_predictions = np.random.randint(2, size = len(y_test))
    
    #Could be any threshold between 0 and 1
    precision, recall, f1 = evaluate(baseline_predictions, y_test, 0.5)
    return precision, recall, f1
    
def speech_only_baseline(y_test, is_speaking):
    y_test = y_test.flatten()
    is_speaking = is_speaking.flatten()
    baseline_predictions = np.random.randint(2, size = len(y_test))
    is_speaking = np.logical_or(y_test, is_speaking) #Probably not needed
    
    baseline_predictions = np.logical_and(baseline_predictions, is_speaking)
    
	#Could be any threshold between 0 and 1
    precision, recall, f1 = evaluate(baseline_predictions, y_test, 0.5)
    return precision, recall, f1
    
            
def content_words_baseline(y_test, is_speaking, is_function_word):
    y_test = y_test.flatten()
    is_speaking = is_speaking.flatten()
    is_function_word = is_function_word.flatten()
    
    baseline_predictions = np.random.randint(2, size = len(y_test))
    is_speaking = np.logical_or(y_test, is_speaking) #Probably not needed
    
    baseline_predictions = np.logical_and(baseline_predictions, is_speaking)
    is_not_function_word = np.logical_not(is_function_word)
    baseline_predictions = np.logical_and(baseline_predictions, is_not_function_word)
    
    #Could be any threshold between 0 and 1
    precision, recall, f1 = evaluate(baseline_predictions, y_test, 0.5)
    return precision, recall, f1

#Takes the predictions done over locations, and the predictions done
#over named entities that aren't locations and runs a 2 sample t-test
def t_test_2_sample(preds_at_locations, preds_at_nonloc_NE):
    from scipy.stats import ttest_ind

    #That was easy
    t, p = ttest_ind(preds_at_locations, preds_at_nonloc_NE, equal_var=False)
    return t, p
    

    
def get_metrics(tp, fn, fp):
    precision, recall, f1_measure = 0, 0, 0
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    
    if (precision + recall) != 0:
        f1_measure = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1_measure
    

#Takes 1D np array of predictions, y test    
def find_confusion_matrix(pred, y_test):
    
    matrix = np.zeros((2,2))
    
    for i in range(len(pred)):
        matrix[int(y_test[i]), int(pred[i])] += 1
    
    fp = matrix[0, 1] #False positive count
    fn = matrix[1, 0] #False negative count
    tp = matrix[1, 1] #True positive count
    return tp, fn, fp
    
