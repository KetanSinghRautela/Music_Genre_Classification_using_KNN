# Muisc_Genre_Classificatio_Using_KNN

## Introduction
Music genre classification forms a basic step for building a strong recommendation system.
The idea behind this project is to see how to handle sound files in python, compute sound and audio features from them, run Machine Learning Algorithms on them, and see the results.
In a more systematic way, the main aim is to create a machine learning model, which classifies music samples into different genres. It aims to predict the genre using an audio signal as its input.
The objective of automating the music classification is to make the selection of songs quick and less cumbersome. If one has to manually classify the songs or music, one has to listen to a whole lot of songs and then select the genre. This is not only time-consuming but also difficult. 
Automating music classification can help to find valuable data such as trends, popular genres, and artists easily. Determining music genres is the very first step towards this direction.

## Dataset

For this project, the dataset that we will be working with is GTZAN Genre Classification dataset which consists of 1,000 audio tracks, each 30 seconds long. It contains 10 genres, each represented by 100 tracks.
The 10 genres are as follows:
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

The dataset has the following folders:
Genres original — A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
Images original — A visual representation for each audio file. One way to classify data is through neural networks because NN’s usually take in some sort of image representation.
2 CSV files — Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs are split before into 3 seconds audio files.

## KNN Through Graph
<img width="678" alt="image" src="https://github.com/KetanSinghRautela/Music_Genre_Classification_using_KNN/assets/129218488/d1fe007a-3389-4c9a-ada1-10bac7f6aaa3">

## Library used are
Libraries used in the code are-
```
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tempfile import TemporaryFile
import os
import math
import pickle
import random
import operator
```

## Accuracy
![image](https://github.com/KetanSinghRautela/Music_Genre_Classification_using_KNN/assets/129218488/923eaa4e-9f31-4d05-9cae-eff04f03e008)



