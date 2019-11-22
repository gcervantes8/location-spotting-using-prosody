# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:11:51 2019

@author: Jerry C
"""

import numpy as np

def failure_analysis(y_test, predictions, audios_info, audio_times):
    
    y_test = y_test.flatten()
    predictions = predictions.flatten()
    audios_info = audios_info.flatten()
    audio_times = audio_times.flatten()

    #Converts from numpy array of objects to np arr of strings
    audios_info = np.array([f[0] for f in audios_info])
    
    dist = np.abs(y_test - predictions)
    
    sorted_indices = np.argsort(dist)
    
    best_dist = dist[sorted_indices]
    best_audio_names = audios_info[sorted_indices]
    best_audio_times = audio_times[sorted_indices]
    best_preds = predictions[sorted_indices]
    best_labels = y_test[sorted_indices]
    
    non_loc_idx = best_labels == 0 #Indices without locations
    loc_idx = best_labels == 1 #Indices with locations
    
    print_best_and_worst_dists('location_analysis', best_dist[loc_idx], best_audio_names[loc_idx], best_audio_times[loc_idx], best_preds[loc_idx], best_labels[loc_idx])
    print_best_and_worst_dists('nonlocation_analysis', best_dist[non_loc_idx], best_audio_names[non_loc_idx], best_audio_times[non_loc_idx], best_preds[non_loc_idx], best_labels[non_loc_idx])
    
    
    
#    for i in range(print_top_n):
#        print('label %.0f prediction %.3f dist %.3f audio_file %s' % (best_labels[i], best_preds[i], best_dist[i], best_audio_names[i]))
    
    
def print_best_and_worst_dists(file_name, best_dist, best_audio_names, best_times, best_preds, best_labels):
    n_to_write = 200
    best_file_name = file_name + '_best.txt'
    
    audio_to_times = dict()
    with open(best_file_name, 'a') as file:
        for i in range(n_to_write):
            info_str = 'label %.0f prediction %.3f dist %.3f audio_file %s' % (best_labels[i], best_preds[i], best_dist[i], best_audio_names[i])
            is_closeby_time = False
            sw_audio = best_audio_names[i]
            sw_time = best_times[i]
            if sw_audio in audio_to_times:
                times_in_audio = audio_to_times[sw_audio]
                for time in times_in_audio:
                    #If time is within 10 seconds
                    if abs(sw_time-time) < 10:
                        is_closeby_time = True
                        break
                if is_closeby_time:
                    continue
                audio_to_times[sw_audio].append(sw_time)
            else:
                audio_to_times[sw_audio] = [sw_time]
                
                
            
            m, s = divmod(best_times[i], 60)
            minutes_str = str(int(m))
            if m < 10:
                minutes_str = '0' + minutes_str
            time_str = minutes_str + ':' + str(round(s, 1))
            info_str = info_str + ' ' + time_str

            file.write(info_str + '\n')
                
    n = len(best_dist)
    audio_to_times = dict()
    worst_file_name = file_name + '_worst.txt'
    with open(worst_file_name, 'a') as file:
        #Iterate backwards
        for i in range(n-1, n-n_to_write, -1):
            info_str = 'label %.0f prediction %.3f dist %.3f audio_file %s' % (best_labels[i], best_preds[i], best_dist[i], best_audio_names[i])
            is_closeby_time = False
            sw_audio = best_audio_names[i]
            sw_time = best_times[i]
            if sw_audio in audio_to_times:
                times_in_audio = audio_to_times[sw_audio]
                for time in times_in_audio:
                    #If time is within 10 seconds
                    if abs(sw_time-time) < 10:
                        is_closeby_time = True
                        break
                if is_closeby_time:
                    continue
                audio_to_times[sw_audio].append(sw_time)
            else:
                audio_to_times[sw_audio] = [sw_time]
            
            
            m, s = divmod(best_times[i], 60)
            minutes_str = str(int(m))
            if m < 10:
                minutes_str = '0' + minutes_str
            time_str = minutes_str + ':' + str(round(s, 1))
            info_str = info_str + ' ' + time_str
            file.write(info_str + '\n')
            
        
        
    

    
    
    
    
