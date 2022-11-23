



# Session 1.1 -  Speech Classification (Acoustic modeling)

***In order to have common framework and to have opportunity to exchange  on possible coming errors.  The usage of Linux OS is strongly recommended during the tutorials.***


## Technical requirecoment :

- python 
- anaconda  (miniconda) or any other virtual environnement manager
- Jupyter Notebooks
    
## Machine learning libraries :

-  Scikit-learn
-  Pytorch

## Audio and speech Processing

- soundfile
- torchaudio
- librosa
- pydub

## Step up anconda env

Create  a virtual environnement the following command line :

```conda create -n    m2aidn_env python=3.9```


launch jupyter notebook and create your own id and tutorial


## Dataset

Please download the speech command dataset using the following command line :

```
from torchaudio.datasets import SPEECHCOMMANDS
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


!# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


```

-   Make	statistics on the database
-   USE AUDIO TORCHAUDIO LIBRAIRY TO EXTRACT FEATURES (MFCC, PITCH (F0))
-   Visualize samples using 
-  Flatting the features on single feature vector.

## Acoustic Model (classification)

- Build classifier using scikit-learn
- Build classifier using pythorch



    




## References

This tutorials was mainly inspired from  :

1.  [AUDIO MANIPULATION WITH TORCHAUDIO](https://gitlab.inria.fr/asini/m2aidn_ubs)
2.  [Speech Command Classification with torchaudio](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html#speech-command-classification-with-torchaudio)
3.  [4 Types of Classification Tasks in Machine Learning
](https://machinelearningmastery.com/types-of-classification-in-machine-learning/)

## Bibliography 

