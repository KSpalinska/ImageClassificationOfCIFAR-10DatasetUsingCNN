## Image classification of CIFAR-10 dataset using Convolutional Neural Networks (CNN).


Proponowane parametry do przetestowania (wszystkiego ze wszystkim możemy nie zdążyć; możemy wybrać niektóre 
i np podzielić się tak że każda z nas będzie miała inny batch_size i będzie robić eksperymenty :P ):

epochs = ( tyle ile damy radę xd ) 125, 250 ?
batch_size: 64, 128, 256, 512
rozmiary w warstwach:
- 16, 32
- 32, 64
- 64, 128
- 128, 256
- 16, 32, 64
- 32, 64, 128
- 64, 128, 256 

pool_size: (3,2), (2,2)


Inspiracje: 

https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

http://cs231n.github.io/convolutional-networks/



# Wersja z dwiema warstwami (tak jak jest teraz): 

#Convolutional Neural Network for CIFAR-10 dataset
#Define the model
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


# Wersja z trzema warstwami - propozycja od pana 

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
    
    #next layers
    model.add(Convolution2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3)))
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


