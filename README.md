# Location Spotting Using Prosody

## Running

The file to run is bidirectional_keras.py file


The implementation of the LSTM model requires that you use a GPU
To make the model CPU compatible, the call of CuDNNLSTM should be changed to LSTM, this will make the model run much slower on GPU but compatible on GPU

tensorflow-gpu

There are 4 parameters that should be adjusted based on what you are using the model for
1. train_mode
2. single_track_train
3. balanced_eval
4. eval_other_lang


1. If train mode is True then the model will be trained.

2. If single_track is True, then only the prosodic features of the user speaking will be used.

3. If balanced_eval is True, then the model will evaluated on a balanced dataset where half of the frames are location frames, and the other half are frames with speech that don't have a location mention.

4. If eval_other_lang is True, then the model will be evaluated over data from another language, as specified in the mat_path variable

