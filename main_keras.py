from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop,Adam
from keras.datasets import cifar10,mnist
import numpy as np
from PIL import Image
import argparse
import math
import matplotlib.pyplot as plt
import keras.backend as K
from keras import constraints,initializers
from Chinese_inputs import CommonChar, ImageChar
import os

def generator_model(im_size, output_channel = 3):
    initializer = initializers.truncated_normal(stddev=0.1)
    model = Sequential()
    model.add(Dense(input_dim=100, units=512*4*4,kernel_initializer=initializer))
    model.add(Activation('linear'))

    model.add(Reshape((4,4,512)))
    model.add(Conv2DTranspose(256,(5,5),strides=(2,2),padding='same',kernel_initializer=initializer))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',kernel_initializer=initializer))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializer))
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Conv2DTranspose(output_channel,(5,5),strides=(2,2),padding='same',kernel_initializer=initializer))
    model.add(Activation('tanh'))
    return model

def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def discriminator_model(im_size, input_channel = 3):
    initializer = initializers.truncated_normal(stddev=0.1)
    model = Sequential()
    model.add(Convolution2D(
                32,(5, 5),
                padding='same',
                input_shape=(im_size, im_size, input_channel),strides=(2,2),
                kernel_initializer=initializer))
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(64,(5, 5), padding='same', strides=(2,2),
                            kernel_initializer=initializer))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(128, (5, 5), padding='same', strides=(2, 2),
                            kernel_initializer=initializer))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    #model.add(Convolution2D(512,(5, 5), padding='same', strides=(2,2),
    #                        kernel_initializer=initializer))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def combine_images(generated_images):
    num = 100 #generated_images.shape[0]
    width = 10 #int(math.sqrt(num))
    height = 10 #int(math.ceil(float(num)/width))
    depth = generated_images.shape[-1]
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1],depth),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images[:num]):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:,:,:]
    return image

def train(BATCH_SIZE):
    d_losses =[]
    g_losses = []
    cc =CommonChar()
    ic =ImageChar()
    X_all = []
    for c in cc.chars:
        ic.drawText(c)
        X_all.append((ic.toArray()-127.5)/127.5)
    X_train = np.array(X_all)

    if len(X_train.shape)==3:
        X_train = X_train.reshape(X_train.shape + (1,))

    optim = Adam(lr=0.0002,beta_1=0.5)

    discriminator = discriminator_model(im_size=64,input_channel=1)
    generator = generator_model(im_size=64,output_channel=1)
    discriminator.compile(loss='binary_crossentropy', optimizer=optim)
    generator.compile(loss='binary_crossentropy', optimizer=optim)
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=optim)

    if not os.path.exists("keras_samples/"):
        os.mkdir("keras_samples/")

    for epoch in range(150):
        print("Epoch is", epoch)
        nob = int(X_train.shape[0]/BATCH_SIZE)
        for index in range(nob):
            noise = np.random.uniform(-1, 1, (BATCH_SIZE,100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)


            combined_X = np.concatenate((image_batch,generated_images),axis=0)
            combined_Y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)

            d_loss = discriminator.train_on_batch(combined_X, combined_Y)

            noise = np.random.uniform(-1, 1, (BATCH_SIZE,100))
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)
            discriminator.trainable = True

            if index == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                if image.shape[-1] ==1:
                    image = image[:,:,0]
                Image.fromarray(image.astype(np.uint8)).save(
                    "keras_samples/"+str(epoch)+"_"+str(index)+".png")

        print("Epoch %d Step %d d_loss : %f" % (epoch, index, d_loss))
        print("Epoch %d Step %d g_loss : %f" % (epoch, index, g_loss))
        d_losses.append(d_loss)
        g_losses.append(g_loss)
    return d_losses,g_losses

if __name__ == "__main__":
    d_losses,g_losses = train(BATCH_SIZE=128)
    print(len(d_losses))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d_losses,label='d_loss')
    ax.plot(g_losses,label='g_loss')
    ax.legend()
    plt.show()