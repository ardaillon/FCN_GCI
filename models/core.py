"""
adapted from original and FCN-f0 repositories at:
https://github.com/marl/crepe
https://github.com/ardaillon/FCN-f0/

original articles:
"CREPE: A Convolutional Representation for Pitch Estimation", 2018, (Kim, Jong Wook; Salamon, Justin; Li, Peter; Bello, Juan Pablo)
"Fully-Convolutional Network for Pitch Estimation of Speech Signals", Interspeech 2019, (Ardaillon, Luc; Roebel, Axel)

modified by Luc Ardaillon: 04/11/2019
"""

import numpy  as np

# the model is trained on 16kHz audio
model_srate = 16000

def build_model(learning_rate=0.0002, weightsFile=None, inputSize=993, dropout = 0, training = False):
    '''
    :param learning_rate:
    :param weightsFile:
    :param inputSize:
    :param dropout:
    :param training:
    :return:
    '''

    from keras.layers import Input, Reshape, Conv2D, BatchNormalization, MaxPool2D, Dropout
    from keras.layers import Permute, Flatten
    from keras.models import Model
    from keras import optimizers

    layers = [1, 2, 3, 4, 5, 6]
    capacity_coeff = 2
    filters = capacity_coeff * np.array([256, 32, 32, 128, 256, 512])
    widths = [32, 32, 32, 32, 32, 32]
    strides = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    if(inputSize is not None):
        x = Input(shape=(inputSize,), name='input', dtype='float32')
        y = Reshape(target_shape=(inputSize, 1, 1), name='input-reshape')(x)
    else:
        x = Input(shape=(None,1,1), name='input', dtype='float32')
        y = x

    for l, f, w, s in zip(layers, filters, widths, strides):
        y = Conv2D(f, (w, 1), strides=s, padding='valid', activation='relu', name="conv%d" % l)(y)
        if(l<4):
            y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid', name="conv%d-maxpool" % l)(y)

        y = BatchNormalization(name="conv%d-BN" % l)(y)
        if(dropout and training):
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

    # here replaced the fully-connected layer by a convolutional one:
    y = Conv2D(1, (4, 1), strides=(1, 1), padding='valid', activation='sigmoid', name="classifier")(y)

    if(training):
        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)

    model = Model(inputs=x, outputs=y)

    if(weightsFile is not None):  # if restarting learning from a checkpoint
        model.load_weights(weightsFile)

    if(training):
        for layer in model.layers:
            layer.trainable = True

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mean_squared_error')

    return model

if __name__ == '__main__':
    # model = build_model()
    # inputSize = None
    inputSize = 993
    model = build_model(learning_rate=0.0002, weightsFile=None, inputSize=inputSize, dropout = 0, training = False)
    model.summary()
