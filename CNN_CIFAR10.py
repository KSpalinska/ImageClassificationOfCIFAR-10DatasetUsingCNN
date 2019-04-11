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
import matplotlib.pyplot as plt
import time


## Funkcje pomocnicze
def accuracy(test_x, test_y, model):
    """ 
    Funkcja licząca precyzję (accuracy) 
    dla danego zbioru danych i danego modelu 
    """
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis = 1)
    true_class = np.argmax(test_y, axis = 1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


## Ładowanie zbiorów danych
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

n_train, img_row, img_col, img_ch = x_train.shape
n_test = x_test.shape[0]
categories = np.unique(y_train)
n_categories = len(categories)


## Pre-processing danych
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# one-hot-encoding - konwersja etykiet kategorii na binarne etykiety kategorii
y_train = np_utils.to_categorical(y_train, n_categories)
y_test = np_utils.to_categorical(y_test, n_categories)


## Parametry sieci
number_of_layers = 1
batch_size = 64
epochs = 5


## CNN dla zbioru CIFAR-10

# Zdefiniwanie modelu
model = Sequential()

# Dodawanie warstw
for i in range(number_of_layers):
    if i == 0:
        # layer0
        model.add(Convolution2D(32, (3, 3), padding = 'same', input_shape = x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

    # next layers
    model.add(Convolution2D(64, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

# flatten layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_categories))
model.add(Activation('softmax'))


## Konfiguracja procesu uczenia
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


## Trenowanie modelu
start = time.time()
model_info = model.fit(x_train, y_train, batch_size=batch_size, epochs = epochs, validation_data = (x_test, y_test))
end = time.time()

print("Model took %0.2f seconds to train" % (end - start))


## Precyzja na zbiorze testowym
print("Accuracy on test data is: %0.2f" % accuracy(x_test, y_test, model))


## Zapis wyników do pliku
f = open("results/batch_size_" + str(batch_size) + ";epochs_" + str(epochs), "w")

for i in range(len(model_info.history['acc'])):
    f.write("\nacc: " + str(model_info.history['acc'][i]))
    f.write("\nval_acc: " + str(model_info.history['val_acc'][i]))

f.close()

#model_info.history[]

# wykres accuracy
plt.plot(model_info.history['acc'])
plt.plot(model_info.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/accuracy_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.png')
plt.show()


# wykres loss
plt.plot(model_info.history['loss'])
plt.plot(model_info.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/loss_epochs_'+str(epochs)+'_batchsize_'+str(batch_size)+'.png')
plt.show()

