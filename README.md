
# Location Spotting Using Prosody


## Installing and Running
Training and evaluation is done through the bidirectional_keras.py file
Python 3 and GPU are required to run code as is.

#### GPU Requirement
The code requires that you use a GPU, for the LSTM.

#### CPU Compatibility
To make CPU compatible:
the Tensorflow.keras.layers call of CuDNNLSTM should be changed to Tensorflow.keras.layers.LSTM,  This will make the code run much slower on GPU



### Required Python packages
* tensorflow-gpu (1.13 tested)
* keras
* scipy
* numpy


## Training and Evaluating
Depending on what the code is being used for, there are 4 parameters that should be adjusted 

1. train_mode
2. single_track_train
3. balanced_eval
4. eval_other_lang
 
**train mode** if is True, then the model will be trained.

**single_track** if is True, then only the prosodic features of the user speaking will be used.

**balanced_eval** if is True, then the model will evaluated on a balanced dataset where half of the frames are location frames, and the other half are frames with speech that don't have a location mention.

**eval_other_lang** if is True, then the model will be evaluated over data from another language, as specified in the mat_path variable

### Authors
Authors for the research paper are:
Gerardo Cervantes
Nigel G. Ward