import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import sounddevice as sd
from scipy.io.wavfile import write
import os
from numpy import argmax

Model1DConvNet = keras.models.load_model('KerasModels/1DConvNet.h5')
Model1DConvNet.summary()

ModelDeepDenseNet = keras.models.load_model('KerasModels/DeepDenseNet.h5')
ModelDeepDenseNet.summary()

ModelLSTM = keras.models.load_model('KerasModels/LSTM.h5')
ModelLSTM.summary()

ModelSimpleDenseNet = keras.models.load_model('KerasModels/SimpleDenseNet.h5')
ModelSimpleDenseNet.summary()

clear = lambda: os.system('clear')

def Encode(mode, input):
	labels = ["1 Lehuzva", "2 Lehuzva", "Alapjarat"]
	print(mode + " " + labels[argmax(input)])
	
while(True):
#	clear()
	record = sd.rec(int((1/2.94) * 44100), samplerate=44100, channels=1)

	
	Encode("SimpleDense ",  ModelSimpleDenseNet.predict(record[:1500].reshape(1,1500))   )
	Encode("DeepDense   ",  ModelDeepDenseNet.predict(record[:1500].reshape(1,1500))   )
	Encode("LSTM        ",  ModelLSTM.predict(record.reshape(1,15000))   )
	Encode("1DConvNet   ",  Model1DConvNet.predict(record.reshape(1,15000))   )
	K.clear_session()


