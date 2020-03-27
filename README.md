

# Location Spotting Using Prosody


## Installing and Running
Training and evaluation is done through the *bidirectional_keras.py* file
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


### Computing features
The prosodic features used in the research paper were computed using the Mid-level prosodic feature toolkit, found [here](https://github.com/nigelgward/midlevel)

Labels are found using *spaCy* (Python library).  *SpaCy* can find whether a word is a location mention.  Matlab can run python code.

Features and labels are saved into a *.mat* file with variables

* trainFeatures
* trainLabels
* testFeatures
* testLabels

**Features** are 3-dimensional of size *\[num. of samples, num. of timesteps, num. of features\]*

**Labels** are 2-dimensional of size *\[num. of samples, num. of timesteps\]*, value 1 if location mention, 0 otherwise


* testIsSpeaking
* testIsFunctionWord

To compute the three baselines from the research paper, you'll need these 2 variables set.  

Both variables resized to be 2-dimensional of size *\[num. of samples, num. of timesteps\]* with value 1 or 0

**IsSpeaking** is 1 if there is speech, 0 otherwise.  These can be gotten from the word-aligned transcripts

**IsFunctionWord** is 1 if is a function word, 0 otherwise.  Using word-aligned transcripts and NLTK stopword list. 



## Training and Evaluating
There are 4 parameters that should be adjusted, depending on what the code is being used for


1. train_mode
2. single_track_train
3. balanced_eval
4. eval_other_lang
 
**train_mode** will train the model when true and do evaluation when false

**single_track** will use only prosodic features of speaker when true, otherwise adds interlocutor features as well

**balanced_eval** will evaluate on balanced dataset where half the frames are location frames, and other half are speech frames without location mentions

**eval_other_lang** will evaluate over data from another language, as specified in the *mat_path* variable


### Research paper authors
Gerardo Cervantes

Nigel G. Ward