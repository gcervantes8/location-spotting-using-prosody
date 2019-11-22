# Location Spotting Using Prosody

## Running

To run, you need to run the bidirectional_keras.py file


The implementation of the LSTM model requires that you use a GPU
To make the model CPU compatible, the call of CuDNNLSTM should be changed to LSTM, this will make the model run much slower on GPU but compatible on GPU

tensorflow-gpu

There is 4 parameters that should be adjusted based on what you are using the model for
1. train_mode
2. single_track_train
3. balanced_eval
4. eval_other_lang