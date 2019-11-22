from __future__ import print_function
import tensorflow
import tensorflow as tf
import data_reader
import numpy as np
from tensorflow.contrib import rnn
from random import shuffle
from eval_metrics import threshold_evaluate, t_test_2_sample
import eval_metrics
import scipy.io as sio
from evaluate_balance import evaluation_on_balanced
from lang_fail_analysis import lang_fail_analysis, random_points_lang
from failure_analysis import failure_analysis
import keras
# Hyper-params
learning_rate = 0.001
#training_steps = 1200
training_steps = 2000
batch_size = 32
display_step = 1
beta = 0.00001

num_input = 10  # Prosody
timesteps = 200  # 60 sec * 20 frames/sec = 1200
#keep_prob_value = 0.6
num_layers = 2
train_mode = False
single_track_train = False
balanced_eval = False
eval_other_lang = True

print('Read params')
tf.reset_default_graph()
num_output_units = 1

# Reading data
print("Reading data...")
if not single_track_train:
    model_path = 'Location_full_model_180_2_track_2720-14-7-7-4.h5'
else:
    model_path = 'Location_full_model_270_1_track_2759-14-7-7-4.h5' #270
    num_input = 5


if eval_other_lang:
    output_file_name = 'English-SBC.txt'
    #mat_path = './RNN-jp-callhome-10s.mat'
    #mat_path = './RNN-Spanish-callhome-10s.mat'
    #mat_path = './RNN-2500-to-4940-spacy.mat'
    #mat_path = './RNN-EnglishNewsBroadcasts-Cleaned.mat'
    mat = sio.loadmat(mat_path)
    feats = mat['testFeatures']
    lang_files = mat['testAudioFile']
    lang_times = mat['testTimes']
	
    random_points_lang(lang_files, lang_times, 300)
else:
    mat_path = './RNN-2500-to-4940-spacy.mat'
    mat = sio.loadmat(mat_path)

    y_dev = mat['devLabels']
    if (not balanced_eval) or train_mode:
        indices = data_reader.get_loc_sample_indices(y_dev)
    else:
        indices = np.array(range(np.size(y_dev, 0)))
    y_dev = y_dev[indices]
    x_dev = mat['devFeatures'][indices]
    is_speaking_dev = mat['devIsSpeaking'][indices]
    if single_track_train:
        x_dev = x_dev[:,:,:5]

    if train_mode:
        y_train = mat['trainLabels']
        indices = data_reader.get_loc_sample_indices(y_train)
        y_train = y_train[indices]
        x_train = mat['trainFeatures'][indices]

        y_dev = np.expand_dims(y_dev, axis = 2)
        y_train = np.expand_dims(y_train, axis = 2)

        if single_track_train:
            x_train = x_train[:,:,:5]

    else:
        y_labels = mat['testLabels']
        if not balanced_eval:
            indices = data_reader.get_loc_sample_indices(y_labels)
        else:
            indices = np.array(range(np.size(y_labels, 0)))
        y_labels = y_labels[indices]
        print('Got x_features')
        x_features = mat['testFeatures'][indices]
        is_speaking_labels = mat['testIsSpeaking'][indices]
        is_function_word_labels = mat['testIsFunctionWord'][indices]
        test_audio_files = mat['testAudioFile'][indices]
        test_times = mat['testTimes'][indices]
        ne_labels = mat['testIsNamedEntity'][indices]
        if single_track_train:
            x_features = x_features[:,:,:5]



print(str(num_hidden) + ' hidden units')

from tensorflow.keras import layers
from tensorflow.keras import regularizers

if train_mode:
    model = tensorflow.keras.Sequential()
    model.add(layers.Bidirectional(layers.CuDNNLSTM(16, kernel_regularizer = regularizers.l2(beta), recurrent_regularizer = regularizers.l2(beta), bias_regularizer = regularizers.l2(beta), return_sequences=True),
                               input_shape = (timesteps, num_input))) #+ num samples
    model.add(layers.Bidirectional(layers.CuDNNLSTM(8, kernel_regularizer = regularizers.l2(beta), recurrent_regularizer = regularizers.l2(beta), bias_regularizer = regularizers.l2(beta), return_sequences=True)))
    model.add(layers.Bidirectional(layers.CuDNNLSTM(8, kernel_regularizer = regularizers.l2(beta), recurrent_regularizer = regularizers.l2(beta), bias_regularizer = regularizers.l2(beta), return_sequences=True)))
    model.add(layers.Bidirectional(layers.CuDNNLSTM(4, kernel_regularizer = regularizers.l2(beta), recurrent_regularizer = regularizers.l2(beta), bias_regularizer = regularizers.l2(beta), return_sequences=True)))
    model.add(layers.Dense(1, kernel_regularizer = regularizers.l2(beta), bias_regularizer = regularizers.l2(beta)))

    model.compile(loss = 'binary_crossentropy', optimizer = 'RMSprop')
    model.summary()


          
print("Building graph...")

print("\nStarting session... Num output units: ", num_output_units)


nParams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print(nParams)


if eval_other_lang:
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    preds = model.predict(feats)
    lang_fail_analysis(preds, lang_times, lang_files, output_file_name)
    
elif not train_mode:
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    if balanced_eval:
        predictions = model.predict(x_dev)
        precision, recall, threshold, f1 = evaluation_on_balanced(predictions, y_dev, is_speaking_dev, None)
        print('Balanced dev results: Prec ' + str(precision) + ' Recall ' + str(recall) + ' f1 ' + str(f1))

        predictions = model.predict(x_features)
        precision, recall, threshold, f1 = evaluation_on_balanced(predictions, y_labels, is_speaking_labels, threshold)
        print('Balanced test results: Prec ' + str(precision) + ' Recall ' + str(recall) + ' f1 ' + str(f1))
        print('Threshold ' + str(threshold))
    else:
        dev_preds = model.predict(x_dev)
        precision, recall, threshold, f1 = eval_metrics.threshold_evaluate(dev_preds, y_dev, False)
        print('Results on Dev: Prec ' + str(precision) + ' Recall ' + str(recall) + ' f1 ' + str(f1) + ' threshold ' + str(threshold))
        predictions = model.predict(x_features)
        precision, recall, f1 = eval_metrics.evaluate(predictions, y_labels, threshold)
        print('Results: Prec ' + str(precision) + ' Recall ' + str(recall) + ' f1 ' + str(f1))
        precision, recall, f1 = eval_metrics.random_baseline(y_labels)
        print('Random baseline: Prec ' + str(precision) + ' Recall ' + str(recall) + ' f1 ' + str(f1))
        precision, recall, f1 = eval_metrics.speech_only_baseline(y_labels, is_speaking_labels)
        print('Speech only baseline: Prec ' + str(precision) + ' Recall ' + str(recall) + ' f1 ' + str(f1))
        precision, recall, f1 = eval_metrics.content_words_baseline(y_labels, is_speaking_labels, is_function_word_labels)
        print('Content word baseline: Prec ' + str(precision) + ' Recall ' + str(recall) + ' f1 ' + str(f1))
        failure_analysis(y_labels, predictions, test_audio_files, test_times)

        location_idx = y_labels == 1
        ne_idx = ne_labels == 1
        non_locations = np.logical_not(location_idx)
        non_location_named_entities = np.logical_and(ne_idx, non_locations)
            
        t_stat, p_val = t_test_2_sample(predictions[location_idx], predictions[non_location_named_entities])
        print('NE-T-test, t: ' + str(t_stat) + ' p: ' + str(p_val))
        loc_mean = np.mean(predictions[location_idx].flatten())
        nonloc_ne_mean = np.mean(predictions[non_location_named_entities].flatten())
        print('Location Mean ' + str(loc_mean) + ' nonloc ne mean' + str(nonloc_ne_mean))
            
        
else:
    print('Training start')
    class Metrics(keras.callbacks.Callback):
        def __init__(self, validation_data=()):
            self.x_dev, self.y_dev = validation_data
            
        def on_train_begin(self, logs={}):
            self._data = []
            self.epoch = 0
            self.best_f1_scores = np.array([0.0, 0, 0])
        def on_epoch_end(self, batch, logs={}):
            self.epoch += 1
            if ((self.epoch % 30) == 0):
                
                predictions = model.predict(self.x_dev)
                epoch_mae = 0 #TODO COMPUTE MAE
                precision, recall, threshold, f1 = threshold_evaluate(predictions, self.y_dev, False)
                print("Epoch " + str(self.epoch) + ' f1: ' + str(f1))
                print("Precision " + str(precision) + ", Recall= " + str(recall) + ' Threshold= ' + str(threshold))
                print()
                if(np.any(f1 > self.best_f1_scores)):
                    smallest_index = self.best_f1_scores.argmin()
                    self.best_f1_scores[smallest_index] = f1
                    print('Saved top 3 model, epoch', str(self.epoch))
                    n_tracks = 2
                    if single_track_train:
                        n_tracks = 1
                    model.save('./Location_Full_Model_' + str(self.epoch) + '_'  + str(n_tracks) + '_track.h5')
                    
        def get_data(self):
            return self._data
            
    # Run the initializer
    metrics = Metrics((x_dev, y_dev))
    model.fit(x_train, y_train, epochs = 400, validation_data=(x_dev, y_dev), callbacks=[metrics])
    print("Optimization Finished!")

