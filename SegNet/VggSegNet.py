from keras.models import *
from keras.layers import *

import os

file_path = os.path.dirname(os.path.abspath(__file__))
VGG_Weights_path = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation\FNN\data\vgg16_weights_th_dim_ordering_th_kernels.h5"
IMAGE_ORDERING = 'channels_last'


def VGGSegnet(nb_classes, input_height=240, input_width=240, vgg_level=3):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    model = Sequential()
    model.add(Layer(input_shape=(input_height, input_width, 3)))

    # encoder
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # decoder
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())

    model.add(Convolution2D(nb_classes, 1, 1, border_mode='valid', ))

    model.outputHeight = model.output_shape[-2]
    model.outputWidth = model.output_shape[-1]

    print(model.summary())
    model.add(Reshape((nb_classes, model.output_shape[2] * model.output_shape[1])))

    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    m = VGGSegnet(101, 255, 255)

    from keras.utils import plot_model

    plot_model(m, show_shapes=True, to_file='model.png')
