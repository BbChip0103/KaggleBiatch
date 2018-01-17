This is part of 4th place solution of our team for TensorFlow Speech Recognition Challenge - https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/.

# Prerequirements
* python 2.7
* pytorch

# Stucture
- dataset - dataloaders
- utils - blending utils; pseudo, submission and double words generation
- nets - folder with networks
- img - augmentations
- stacker.py - xgboost L2 model 
- neural_stacker.py - neural net L2 model
- main - training and predicting with TTA

# Silence preparation
I cut all the noise files by sliding window of length = 1 sec by constant step for all files. The step size was that the number of 1sec silence clips were around 2300 - to make dataset balanced.

# Cross Validation
I just used 5 kfolds by person ids for all classes, except silence. 6 silence files were also divided into 5 folds (in one fold there were both type of noise - pink and other.)

# Solution overview
I trained L1 models on 31 class on spectrograms with TTA. I didn't do any preprocessing. For augmentations I used: random volume, random noise - normalized by power energy, random shift and random speed.

I didn't play much with L2 models. I tried xgboost stacking and neural net stacking. After teaming up I didn't spend a lof of time with that - my teammates were doing that better :) 

I also trained separate model for unknown unknowns.

# Unknown unknowns 

The idea behind it was that some "unknown" test predictions from ordinary L1 models sometimes confuse "unknown unknowns" with core words. So we may want to try adding additional "unknown unknowns" class. So I tried to simulate it, generating double-words.

So double-words are 2 random words from different classes from one person, that were concatenated together. I took the first part of first word and second part of second word. The words are divided my index of max amplitude. Then each part is normalized by energy power and after that concatenated with random volumes. So I generated around 770k+ of double-words.

Here we have words "stop" and "dog" - and generating the word "stog". (In "resource" folder you can find additional "stog.wav" file.) 

![Alt text](/resources/stog.png?raw=true)

After that I trained 2 networks (Densenet201 and VGGbn19) on 32 classes (31 original class and 1 class for double words) - Model A. After that I took ~3100 most confident test predictions and added them to train. Trained 3 networks (Densenet201, VGGbn19 and InceptionResnetV2) on the new dataset - 31 original class + pseudo "unknown unknowns" - Model B.

Because I finished doing that 2 days before the end of competition there were no time to retrained all other model of ensemble - so we just replaced all predictions from our ensemble by "unknown" where the Model B was giving label for "unknown unknown".

If replaced by Model A - the private LB goes up by 0.01. If replaced by Model B - the increase is 0.24 on private LB.



