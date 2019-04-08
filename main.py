"""

MGU - Projekt nr 2
Klasyfikacja obrazów przy użyciu konwolucyjnych sieci neuronowych

"""

## Ładowanie bibliotek i pakietów
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import time


## Ładowanie zbiorów danych
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

n_train, img_row, img_col, img_ch = x_train.shape
n_test = x_test.shape[0]
categories = np.unique(y_train)
n_categories = len(categories)
number_of_layers=1
batch_size=64
epochs=125






## Data pre-processing
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#one-hot-encoding
# convert class labels to binary class labels
y_train = np_utils.to_categorical(y_train, n_categories)
y_test = np_utils.to_categorical(y_test, n_categories)

## Convolutional Neural Network for CIFAR-10 dataset
# Define the model
model = Sequential()
#Add layers
for i in range(number_of_layers):
    if i ==0 :
        #layer0
        model.add(Convolution2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
    
    #next layers
    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

#flatten layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_categories))
model.add(Activation('softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()
model_info = model.fit(x_train, y_train, batch_size=batch_size, \
    epochs=epochs, validation_data=(x_test, y_test))
end = time.time()

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

print("Model took %0.2f seconds to train" % (end - start))
# compute test accuracy
print("Accuracy on test data is: %0.2f" % accuracy(x_test, y_test, model))
f = open("batch_size"+str(batch_size)+"_epochow_"+str(epochs),"w")
f.write(model_info.history['acc'])
f.write(model_info.history['val_acc'])
f.close()
#model_info.history[]

