
import os
from models.core import build_model

def load_model(modelName, from_json = False):
    '''
    load model from json file and corresponding weights from hdf5 file
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

    from models.core import build_model
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
