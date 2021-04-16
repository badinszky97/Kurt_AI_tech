import Commons

Commons.Log("Importálás")

import random, scipy.signal, scipy.io.wavfile
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.regularizers import l1
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from tensorflow.keras.layers import Dropout



Commons.Log("Adatok beolvasása")

ablakmeret = 15000

Adatgyujto = Commons.DataKeeper()

Adatgyujto.AddNewType(Commons.WavToOverlappedArray("sounds/1lehuzva_mono.wav", window_size=ablakmeret, offset=500), 0)
Adatgyujto.AddNewType(Commons.WavToOverlappedArray("sounds/2lehuzva_mono.wav", window_size=ablakmeret, offset=500), 1)
Adatgyujto.AddNewType(Commons.WavToOverlappedArray("sounds/alapjarat_mono.wav", window_size=ablakmeret, offset=3000),2)


#Adatgyujto.AddNewType(Commons.WavToSplittedArray("sounds/1lehuzva_mono.wav", window_size=ablakmeret), 0)
#Adatgyujto.AddNewType(Commons.WavToSplittedArray("sounds/2lehuzva_mono.wav", window_size=ablakmeret), 1)
#Adatgyujto.AddNewType(Commons.WavToSplittedArray("sounds/alapjarat_mono.wav", window_size=ablakmeret),2)




Commons.Log("Train - Test - Valid szétválasztása")

Adatgyujto.GenerateTrainTestValid()
print(str(type(Adatgyujto.x_test)))
print(str(Adatgyujto.x_test.shape))
Commons.Log("Modell létrehozása")


#Modell létrehozása

tf.compat.v1.reset_default_graph()
clear_session            

# Model
#######

input_shape=(ablakmeret, 1)
x = Input(shape=input_shape)

# Hidden layers
#d0_layer = Dropout(.3)(x)
conv1 = Conv1D(filters=10,
                kernel_size=5,
                activation="relu", input_shape=input_shape[1:])(x)

pool1 = MaxPool1D(pool_size=20)(conv1)

#dropout1 = Dropout(rate=0.3)(pool1)
conv2 = Conv1D(filters=5,
               kernel_size=3,
               activation="relu")(pool1)

pool2 = MaxPool1D(pool_size=10, strides=1)(conv2)

fllayer = Flatten()(pool2)

dense1 = Dense(units=50, activation="relu")(fllayer)
#dropout3 = Dropout(rate=0.1)(dense1)
dense2 = Dense(units=20, activation="relu")(dense1)
#dropout4 = Dropout(rate=0.3)(dense2)
predictions = Dense(units = 3, activation='softmax')(dense2)

model = Model(inputs=x, outputs=predictions)

#10,5,64    5,3,20    50 20 3  ~90%

model.summary()

Commons.Log("Model Train")

#Train szakasz

loss = sparse_categorical_crossentropy
optimizer = Adam()
 
# Compilation
#############

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
print("tanitunk: " + str(type(Adatgyujto.x_train)) + str(type(Adatgyujto.y_train)))
print("tanitunk: " + str((Adatgyujto.x_train.shape)) + str((Adatgyujto.y_train.shape)))


history = model.fit(x=Adatgyujto.x_train, y=Adatgyujto.y_train,
                    validation_data=(Adatgyujto.x_valid, Adatgyujto.y_valid),
                    epochs=100,
                    batch_size=50)

Commons.Log("Mentés")

Commons.display_history(history)
kerdes = input("Mentsuk a modelt? [Igen/...]")
if(kerdes == "Igen"):
    model.save("KerasModels/1DConvNet.h5")
    Commons.save_history(history, "KerasModels/1DConvNet")


