from dnn_mods.gputils import lock_gpu
from dnn_mods.pulse_pred.models.testModel_993.input_pipeline.mainPipeline import get_files_lists
import os
import time as timeModule
import sys
import numpy as np
from pysndfile import sndio
from fileio import iovar
from dnn_mods.pulse_pred.models.testModel_993.core import build_model
from numpy.lib.stride_tricks import as_strided
import warnings

def sliding_norm(audio, frame_sizes = 993):
    n_frames = len(audio)
    audio = np.pad(audio, frame_sizes//2, mode = 'wrap')

    hop_length = 1
    frames = as_strided(audio, shape=(frame_sizes, n_frames), strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose()

    # normalize each frame -- this is expected by the model
    mean = np.mean(frames, axis=1)[:, np.newaxis]
    std = np.std(frames, axis=1)[:, np.newaxis]

    startId = frame_sizes//2
    stopId = startId + len(mean)
    audio = audio[startId:stopId]
    mean = mean.flatten()
    std = std.flatten()

    audio -= mean
    audio /= std

    return np.array(audio)

def segment_sound(snd, segDuration = 60., model_sr = 16000., frame_sizes = 993, totalNetworkPooling = 8):

    maxSndSegLenSamp = int(segDuration * model_sr)
    maxSndSegLenSamp -= maxSndSegLenSamp%totalNetworkPoolingFactor
    maxSndSegLenSamp += frame_sizes

    sndSegments = []
    i = 0
    while(i < len(snd) - frame_sizes):
        startIdx = i
        endIdx = i+maxSndSegLenSamp
        sndSegments.append(snd[startIdx:endIdx])
        i = endIdx - frame_sizes + totalNetworkPooling

    return sndSegments

if __name__ == '__main__':
    ####################################################################################################################
    ############################################# Command-line interface : #############################################
    ####################################################################################################################
    from argparse import ArgumentParser
    parser = ArgumentParser(description="run analysis of a given model on a given database")

    global_args = parser.add_argument_group('global')
    global_args.add_argument('-w', "--weights_file", default='weights.h5', help='file containing the weights of the model')
    global_args.add_argument('-p', "--paramsFile", default=None, help='parameters dictionary stored as pickle file providing the parameters to be used for the training')
    global_args.add_argument('-i', "--input", default='/data2/anasynth_nonbp/ardaillon/f0_analysis/corpus/val_set-orig_snds/audioCorrected', help='input sound data to be analysed (either a directory or a file')
    global_args.add_argument('-o', "--output", default='~/deepF0/test', help='output f0 analysis (either a directory or a file with sdif extension)')
    global_args.add_argument('-norm', "--norm", type=int, default=0, help='normalize input with mean and std')
    parser.add_argument("--cpu", action="store_true", help="run on CPU instead of GPU")

    args = parser.parse_args()

    ####################################################################################################################
    ############################################# Get Input files : ####################################################
    ####################################################################################################################

    weights_file = args.weights_file
    input = args.input
    output = args.output

    ####################################################################################################################
    ############################################# Get params : #########################################################
    ####################################################################################################################
    inputStride = 1

    paramsFile = args.paramsFile
    try:
        paramsDict = iovar.load_var(paramsFile)
    except:
        print("paramsFile not found")

    try:
        model_name = paramsDict['model_name']
    except:
        model_name = 'testModel_993'
    print("model_name = "+model_name)

    try:
        model_sr = paramsDict['model_sr']
    except:
        model_sr = 16000.
    print("model_sr = "+str(model_sr))

    try:
        input_size = paramsDict['num_data_samples']
    except:
        input_size = 993
    print("input_size = "+str(input_size))

    totalNetworkPoolingFactor = 8

    ####################################################################################################################
    ############################################# Get a lock on a GPU : ################################################
    ####################################################################################################################

    try:
        if(not args.cpu):
            gpu_id_locked = lock_gpu()
    except:
        print("unable to lock a gpu")

    ####################################################################################################################
    ########################################### run network on audio files : ###########################################
    ####################################################################################################################

    maxSndSegLenSec = 60 # analyse segments of a maximum duration of maxSndSegLenSec seconds to avoid running out of memory

    print("Run prediction from trained model on audio files")
    listFilesTest = []
    num_test_files = 0

    # Get list of all input files for evaluation
    if input:
        print("gather test examples", file=sys.stderr)
        sys.stderr.flush()

        listFilesTest = get_files_lists(input)

    num_test_files = np.sum([len(l) for l in listFilesTest])
    if num_test_files == 0:
        raise RuntimeError("rootDataDirVal dir empty" + str(input))
    print("number of test files", [len(ll) for ll in listFilesTest], file=sys.stderr)

    print("### Build and load pre-trained model ###")
    model = build_model(learning_rate=0.0002, weightsFile=weights_file, inputSize=None, training=0)

    startAnaTime = timeModule.time()

    file_nb = 0

    for ll in listFilesTest:
        for f in ll:
            file_nb += 1
            print("running prediction on file "+f)
            print("file nb "+str(file_nb)+" / "+str(num_test_files))
            if(f.endswith('.wav')): # read audio file
                snd, snd_sr, enc = sndio.read(f)
            elif(f.endswith('.p')): # get audio from pickled data file
                data = iovar.load_var(f)
                snd = data['data'][0]
                enc = 'pcm16'
                try:
                    if(data['sr'] is not None):
                        snd_sr = data['sr']
                    else:
                        snd_sr = model_sr
                except:
                    snd_sr = model_sr

            if len(snd.shape) == 2: # make mono
                snd = snd.mean(1)
                snd = snd.astype(np.float32)

            if snd_sr != model_sr: # resample audio if necessary
                from resampy import resample
                snd = resample(snd, snd_sr, model_sr)

            if(args.norm):
                snd = sliding_norm(snd, frame_sizes = input_size) # normalize sound

            snd = np.pad(snd, input_size//2, mode='wrap') # pad sound at extremities

            sndSegments = segment_sound(snd, segDuration=maxSndSegLenSec, model_sr=model_sr, frame_sizes=input_size, totalNetworkPooling=totalNetworkPoolingFactor) # segment sound in chunks for analysis

            activations = []

            for iseg, sndSeg in enumerate(sndSegments):

                ####################################################################################################################
                ############################################# Load pre-trained model : #############################################
                ####################################################################################################################

                # print("### Build and load pre-trained model ###")
                # winSize = len(sndSeg)
                # model = build_model(learning_rate=0.0002, weightsFile=weights_file, inputSize=winSize, training = 0)

                # run prediction and convert the frequency bin weights to Hz
                audio = np.array([np.reshape(sndSeg, (len(sndSeg),1,1))])
                start_predictionTime = timeModule.time()
                activationsSeg = model.predict(audio, verbose=1)
                stop_predictionTime = timeModule.time()

                confidence = activationsSeg.max(axis=3)

                activationsSeg = np.reshape(activationsSeg, (np.shape(activationsSeg)[1]))

                activations = np.concatenate((activations, activationsSeg))

            time_src = np.arange(len(activations))*totalNetworkPoolingFactor
            time_target = np.arange(len(snd)-input_size+1)
            interpType = 'cubicSpline'
            if(interpType == 'linear'):
                activations = np.interp(time_target, time_src, activations)
            elif(interpType=='cubicSpline'):
                from scipy.interpolate import CubicSpline
                if(np.isnan(activations).any()):
                    warnings.warn("activations contains nan values")
                    noNaN_ids = np.where(np.isnan(activations)==False)[0]
                    time_src = time_src[noNaN_ids]
                    activations = activations[noNaN_ids]
                interpolator = CubicSpline(time_src, activations)
                activations = interpolator(time_target)

            outputDir = os.path.dirname(output)
            if(not os.path.isdir(outputDir)):
                os.makedirs(outputDir)
            if(output.endswith('.wav')):
                outF0AnalysisFileName = os.path.basename(output)
            else:
                outF0AnalysisFileName = os.path.splitext(os.path.split(f)[-1])[0]+'.targetPred.wav'
            outf0file = os.path.join(outputDir, outF0AnalysisFileName)
            print("saving GCI analysis to file "+outf0file)
            sndio.write(outf0file, activations, model_sr, 'wav', enc)

            anaTime_prediction = stop_predictionTime - start_predictionTime
            print("file analyzed in "+str(anaTime_prediction)+"s")

    stopAnaTime = timeModule.time()
    anaTime = stopAnaTime - startAnaTime

    print("analysis on test data run in "+str(anaTime)+"s")
