# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:37:23 2019

@author: Jerry C
"""

import numpy as np

def lang_fail_analysis(predictions, times, files, output_file_name):
    predictions = np.array(predictions).flatten()
    times = np.array(times).flatten()
    files = np.array(files).flatten()

    #Turn form numpy array of objects to numpy array of strings
    files = np.array([f[0] for f in files])
    
    sorted_indices = np.argsort(predictions)
    
    s_predictions = predictions[sorted_indices]
    s_times = times[sorted_indices]
    s_files = files[sorted_indices]
    
    #fail_analysis_strs = filter_close_preds(s_predictions, s_times, s_files, 300)
    #fail_analysis_strs = []
    #j = len(s_predictions) -1
    #for i in range(len(s_predictions)):

#        analysis_str = 'File: ' + str(s_files[j]) + ' Time: ' + str(s_times[j]) + ' Pred: ' + str(s_predictions[j])
#        fail_analysis_strs.append(analysis_str)
#        j -= 1

#    i = len(s_predictions)-1
#    fail_analysis_strs = []
#    while len(fail_analysis_strs) < 300:
#        fail_analysis_strs.append('File: ' + str(s_files[i][0]) + ' Time: ' + str(s_times[i]) + ' Pred: ' + str(s_predictions[i]))    
#        i -= 1
    
#    best_file_name = 'Spanish_high_preds.txt'
    audio_to_times = dict()
    n_outputted = 0
    print('Number predictions' + str(len(s_predictions)))
    with open(output_file_name, 'a') as file:
        j = len(s_predictions) - 1
        for i in range(len(s_predictions)):
            is_closeby_time = False
            analysis_str = 'File: ' + str(s_files[j]) + ' Time: ' + str(s_times[j]) + ' Pred: ' + str(s_predictions[j])
            sw_audio = s_files[j]
            sw_time = s_times[j]
            j -= 1
            if sw_audio in audio_to_times:
                times_in_audio = audio_to_times[sw_audio]
                for time in times_in_audio:
                    if abs(sw_time-time) < 10:
                        is_closeby_time = True
                        break
                if is_closeby_time:
                    continue
                audio_to_times[sw_audio].append(sw_time)
            else:
                audio_to_times[sw_audio] = [sw_time]
#            info_str = 'label %.0f prediction %.3f dist %.3f audio_file %s' % (best_labels[i], best_preds[i], best_dist[i], best_audio_names[i])
            file.write(analysis_str + '\n')
            n_outputted += 1
            if n_outputted > 200:
                print('Reached output limit')
                break
    
    
def random_points_lang(files, times, n):
    import random
    times = np.array(times).flatten()
    files = np.array(files).flatten()
    
    #Turn form numpy array of objects to numpy array of strings
    files = np.array([f[0] for f in files])
    fail_analysis_strs = []
    for i in range(n):
        idx = random.randint(1, len(times))
        minutes, seconds = divmod(times[idx], 60)
        minutes_str = str(int(minutes))
        if minutes < 10:
            minutes_str = '0' + minutes_str
        time_str = minutes_str + ':' + str(round(seconds, 1)) 
        #time_str = "%d:%03d"%(minutes, seconds)
        fail_analysis_strs.append('File: ' + str(files[idx]) + ' Time: ' + str(time_str))
        
    best_file_name = 'Random_points.txt'
    
    with open(best_file_name, 'a') as file:
        for analysis_str in fail_analysis_strs:
            file.write(analysis_str + '\n')
        
    
            
#If predictions are within 1 second, then only takes the first one.
def filter_close_preds(predictions, times, files, n):
    
    
    file_times = {}
    
    n_preds = len(predictions)
    
    fail_analysis_strs = []
    
    idx = n_preds-1
    while n > len(fail_analysis_strs) and (idx >= 0):
        file = files[idx]
        time = times[idx]
        
        
        is_sec_apart = True
        try:
            times_for_file = file_times[file]
            insert_index = np.searchsorted(times_for_file, time)
            #Since the times index if sorted, items before and after only need to checked
            
            
            if insert_index > 0:
                time_before = times_for_file[insert_index-1]
                if (time - time_before) <= 1:
                    is_sec_apart = False
            
            if insert_index < len(times_for_file):
                time_after = times_for_file[insert_index]
                if (time_after - time) <= 1:
                    is_sec_apart = False
            
            if is_sec_apart:
                times_for_file = np.insert(times_for_file, insert_index, time)
#                bisect.insort(times_for_file, time)
                file_times[file] = times_for_file
            
        except KeyError:
            file_times[file] = np.array([time])
            is_sec_apart = True
    
        if is_sec_apart:
            minutes, seconds = divmod(time, 60)
            minutes_str = str(int(minutes))
            if minutes < 10:
                minutes_str = '0' + minutes_str
            time_str = minutes_str + ':' + str(round(seconds, 1)) 
#            time_str = "%d:%03d"%(minutes, seconds)
            fail_analysis_strs.append('File: ' + str(file) + ' Time: ' + str(time_str) + ' Pred: ' + str(predictions[idx]))
        idx -= 1
            
    return fail_analysis_strs
            
        
        
