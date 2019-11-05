"""
adapted from original CREPE and FCN-f0 repositories at:
https://github.com/marl/crepe
https://github.com/ardaillon/FCN-f0/

original articles:
"CREPE: A Convolutional Representation for Pitch Estimation", 2018, (Kim, Jong Wook; Salamon, Justin; Li, Peter; Bello, Juan Pablo)
"Fully-Convolutional Network for Pitch Estimation of Speech Signals", Interspeech 2019, (Ardaillon, Luc; Roebel, Axel)

Code for building, loading, and viewing model

modified by Luc Ardaillon: 04/11/2019
"""

import numpy  as np
import os

# the model is trained on 16kHz audio
model_srate = 16000

def build_model(learning_rate=0.0002, weightsFile=None, inputSize=993, dropout = 0, training = False):
    '''
    Build the FCN model (both for training or inference, though at inference the model could be loaded from a json file instead)
    :param learning_rate: initial learning rate. Used only for training
    :param weightsFile: file containing the weights of the model (hdf5 format), when loading a pre-trained model
    :param inputSize: minimum input size for the model to predict something
    :param dropout: if 1, use dropout in training (not used in our work, but kept here as an option)
    :param training: In our code for training, data shape is handled a bit differently than for inference, so need to specify if we want to train or use the model
    :return: compiled model (possibly loaded with weights of pre-trained model if weightsFile is specified)
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


def load_model(modelName = 'FCN_synth_GF', from_json = False):
    '''
    build model or load it from json file and load corresponding weights from hdf5 file, according to given model name
    :param modelName: name of the model to be used (default is FCN_synth_GF)
    :param from_json: If true, load the model from a json file instead of using the build_model function
    :return: compiled model loaded with weights corresponding to given pre-trained model name
    '''

    curDir = os.path.dirname(os.path.abspath(__file__))
    print(os.path.isdir(curDir))
    if modelName == 'FCN_synth_tri':
        modelDir = os.path.join(curDir, 'FCN_synth_tri')
        weightsFile = os.path.join(modelDir, 'weights.h5')
    elif modelName == 'FCN_synth_GF':
        modelDir = os.path.join(curDir, 'FCN_synth_GF')
        weightsFile = os.path.join(modelDir, 'weights.h5')
    elif modelName == 'FCN_CMU__10_90':
        modelDir = os.path.join(curDir, 'FCN_CMU_10_90')
        weightsFile = os.path.join(modelDir, 'weights.h5')
    elif modelName == 'FCN_CMU__60_20_20':
        modelDir = os.path.join(curDir, 'FCN_CMU__60_20_20')
        weightsFile = os.path.join(modelDir, 'weights.h5')
    else:
        raise("Model doesn't exist. Available options are ['FCN_synth_tri', 'FCN_synth_GF', 'FCN_CMU__10_90', 'FCN_CMU__60_20_20']")

    modelFile = os.path.join(curDir, 'model.json')

    if(from_json):
        json_file = open(modelFile, 'r')
        loaded_model_json = json_file.read()
        from keras.models import model_from_json
        model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights(weightsFile)
    else:
        # for FULLCONV mode, input size is not defined
        model = build_model(learning_rate=0.0002, weightsFile=weightsFile, inputSize=None, training=False)

    return model


if __name__ == '__main__':
    '''
    View model summary
    '''
    # model = build_model()
    # inputSize = None
    inputSize = 993
    model = build_model(learning_rate=0.0002, weightsFile=None, inputSize=inputSize, dropout = 0, training = False)
    model.summary()
