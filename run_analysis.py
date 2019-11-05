"""
adapted from original CREPE and FCN-f0 repositories at:
https://github.com/marl/crepe
https://github.com/ardaillon/FCN-f0/

original articles:
"CREPE: A Convolutional Representation for Pitch Estimation", 2018, (Kim, Jong Wook; Salamon, Justin; Li, Peter; Bello, Juan Pablo)
"Fully-Convolutional Network for Pitch Estimation of Speech Signals", Interspeech 2019, (Ardaillon, Luc; Roebel, Axel)

Code for running analysis on files

modified by Luc Ardaillon: 05/11/2019
"""

import os
import sys
import numpy as np
from pysndfile import sndio
import warnings
from models.core import load_model
import re
from target_to_GCI import get_gci_times
from fileio.sdif import mrk

# model is trained for a sampling rate of 16000Hz
model_sr = 16000.

def db2lin(vec) :
    '''
    Convert values from dB to linear
    :param vec: input values on a linear scale
    :return: output values on a dB scale
    '''
    return 10**(vec/20.)

def normalize_snd_file(inSnd, level = -3):
    '''
    Normalize input sound to have its maximum absolute value at the given level in dB
    :param inSnd: input sound
    :param level: maximum absolute value in dB
    :return: normalized sound
    '''
    maxValIn = np.max(np.abs(inSnd))
    maxValOut = db2lin(level)
    scaleFactor = maxValOut / maxValIn
    normSnd = inSnd * scaleFactor
    return normSnd

def segment_sound(snd, segDuration = 60., frame_sizes = 993, totalNetworkPoolingFactor = 8):
    '''
    Segment input sound file into smaller chunks to avoid memory issues during inference
    :param snd: input sound
    :param segDuration: maximum duration of segments (in s)
    :param frame_sizes: minimal size used by the network for prediction (necessary to have a correct connection of predicted values when reassembling the predicted vectors in the end)
    :param totalNetworkPoolingFactor: total pooling factor (downsampling) applied by the max pooling layers in the network (necessary to have a correct connection of predicted values when reassembling the predicted vectors in the end)
    :return: sound segments
    '''
    maxSndSegLenSamp = int(segDuration * model_sr)
    maxSndSegLenSamp -= maxSndSegLenSamp%totalNetworkPoolingFactor
    maxSndSegLenSamp += frame_sizes

    sndSegments = []
    i = 0
    while(i < len(snd) - frame_sizes):
        startIdx = i
        endIdx = i+maxSndSegLenSamp
        sndSegments.append(snd[startIdx:endIdx])
        i = endIdx - frame_sizes + totalNetworkPoolingFactor

    return sndSegments

def run_prediction(filename, output = None, modelTag = 'FCN_synth_tri', verbose = True):
    '''
    Collect the sound files to process and run the prediction on each file
    :param filename: List
        List containing paths to sound files (wav or aiff) or folders containing sound files to be analyzed.
    :param output: str or None
        Path to directory for saving output files. If None, output files will be saved to the directory containing the input file.
    :param modelTag: str
        name of the pre-trained model to be used for inference
    :param verbose: bool
        Print status messages and keras progress (default=True).
    :return: nothing
    '''

    # load model:
    load_from_json = False

    if(load_from_json):
        model = load_model(modelTag, from_json=True)
    else:
        model = load_model(modelTag)

    if(modelTag=='FCN_synth_GF'):
        mode = 'GF'
    else:
        mode = 'triangle'

    files = []
    for path in filename:
        if os.path.isdir(path):
            found = ([file for file in os.listdir(path) if
                      (file.lower().endswith('.wav') or file.lower().endswith('.aiff') or file.lower().endswith('.aif'))])
            if len(found) == 0:
                print('FCN_GCI: No sound files (only wav or aiff supported) found in directory {}'.format(path),
                      file=sys.stderr)
            files += [os.path.join(path, file) for file in found]
        elif os.path.isfile(path):
            if not (path.lower().endswith('.wav') or path.lower().endswith('.aiff') or path.lower().endswith('.aif')):
                print('FCN_GCI: Expecting sound file(s) (only wav or aiff supported) but got {}'.format(path),
                      file=sys.stderr)
            else:
                files.append(path)
        else:
            print('FCN_GCI: File or directory not found: {}'.format(path),
                  file=sys.stderr)

    if len(files) == 0:
        print('FCN_GCI: No sound files found in {} (only wav or aiff supported), aborting.'.format(filename))
        sys.exit(-1)

    for i, file in enumerate(files):
        if verbose:
            print('FCN_GCI: Processing {} ... ({}/{})'.format(
                file, i+1, len(files)), file=sys.stderr)
        run_prediction_on_file(file, output=output, model=model, mode=mode, verbose=verbose)
    return


def get_audio(sndFile, model_input_size = 993):
    '''
    Load and pre-process (make mono, resample, normalize, and pad) audio from input sound file
    :param sndFile: input sound file
    :param model_input_size: minimum input size of the model (necessary for padding input sound to have correct duration in output)
    :return: audio, sound encoding tag
    '''

    # read sound :
    # from scipy.io import wavfile
    # (sr, audio) = wavfile.read(sndFile)
    from pysndfile import sndio
    (audio, sr, enc) = sndio.read(sndFile)

    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)

    sndDuration = len(audio)/sr
    print("duration of sound is "+str(sndDuration))

    if sr != model_sr: # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_sr)

    audio = normalize_snd_file(audio)

    # pad so that frames are centered around their timestamps (i.e. first frame is zero centered).
    audio = np.pad(audio, int(model_input_size//2), mode='constant', constant_values=0)

    return (audio, enc)


def get_output_path(file, suffix, output_dir):
    '''
    Return the path of an output file corresponding to a wav file
    :param file: input file
    :param suffix: suffixe to be used for the output file
    :param output_dir: output directory where to store output file
    :return: full path of output file
    '''

    if((output_dir is not None) and ((suffix.endswith('.sdif') and output_dir.endswith('sdif')) or (suffix.endswith('.csv') and output_dir.endswith('csv')))):
        path = output_dir
    else:
        (filePath, ext) = os.path.splitext(file)
        path = re.sub(r"(?i)"+ext+"$", suffix, file)
        if output_dir is not None:
            path = os.path.join(output_dir, os.path.basename(path))
            if(not os.path.isdir(output_dir)):
                os.makedirs(output_dir)
    return path


def run_prediction_on_file(inFile, output=None, model=None, mode = 'GF', maxSndSegLenSec = 60, model_input_size = 993, totalNetworkPoolingFactor = 8, verbose=True):
    '''
    Run the prediction of target waveform on input file
    :param file: full path to the file to be analyzed
    :param output: output directory
    :param model: prebuilt model with preloaded weights to be used for inference
    :param maxSndSegLenSec: default is 60s. Analyse segments of a maximum duration of maxSndSegLenSec seconds to avoid running out of memory
    :param verbose: print some infos
    :return: don't return anything
    '''

    if(model==None):
        raise('FCN_GCI: model is None')

    # read and pad the audio from file :
    (snd, enc) = get_audio(inFile, model_input_size)

    sndSegments = segment_sound(snd, segDuration=maxSndSegLenSec, frame_sizes=model_input_size, totalNetworkPoolingFactor=totalNetworkPoolingFactor) # segment sound in chunks for analysis

    activations = []

    for iseg, sndSeg in enumerate(sndSegments):

        audio = np.array([np.reshape(sndSeg, (len(sndSeg),1,1))])
        activationsSeg = model.predict(audio, verbose=verbose)
        activationsSeg = np.reshape(activationsSeg, (np.shape(activationsSeg)[1]))
        activations = np.concatenate((activations, activationsSeg))

    time_src = np.arange(len(activations))*totalNetworkPoolingFactor
    time_target = np.arange(len(snd)-model_input_size+1)
    interpType = 'cubicSpline'
    if(interpType == 'linear'):
        targetPred = np.interp(time_target, time_src, activations)
    elif(interpType=='cubicSpline'):
        from scipy.interpolate import CubicSpline
        if(np.isnan(activations).any()):
            warnings.warn("activations contains nan values")
            noNaN_ids = np.where(np.isnan(activations)==False)[0]
            time_src = time_src[noNaN_ids]
            activations = activations[noNaN_ids]
        interpolator = CubicSpline(time_src, activations)
        targetPred = interpolator(time_target)

    targetFile = get_output_path(inFile, ".targetPred.wav", output)
    outputDir = os.path.dirname(targetFile)
    if(not os.path.isdir(outputDir)):
        os.makedirs(outputDir)
    print("saving target shape analysis to file "+targetFile)
    sndio.write(targetFile, targetPred, model_sr, 'wav', enc)

    GCI_times = get_gci_times(targetPred, model_sr, mode=mode)

    GCIFile = targetFile.replace('.targetPred.wav', '.GCI.sdif')
    print("saving GCI markers in file "+GCIFile)
    mrk.store(GCIFile, GCI_times, ['GCI'] * len(GCI_times))

    return


if __name__ == '__main__':
    ####################################################################################################################
    ############################################# Command-line interface : #############################################
    ####################################################################################################################
    from argparse import ArgumentParser
    parser = ArgumentParser(description="run prediction of target shape using the given model on given files or directory")

    global_args = parser.add_argument_group('global')
    parser.add_argument('-i', "--input", nargs='+', default='./examples/speech16k-norm', help='path to one ore more WAV file(s) to analyze OR can be a directory')
    global_args.add_argument('-o', "--output", default='./examples/pred_target_triangle', help='output f0 analysis (either a directory or a file with sdif extension)')
    global_args.add_argument('-m', "--modelTag", default='FCN_synth_tri', help='name model to be used for prediction (default is FCN_synth_tri)')

    args = parser.parse_args()

    input = args.input
    output = args.output
    modelTag = args.modelTag

    ####################################################################################################################
    ############################################# Get a lock on a GPU : ################################################
    ####################################################################################################################

    try:
        from dnn_mods.gputils import lock_gpu
        gpu_id_locked = lock_gpu()
    except:
        print(" ")

    ####################################################################################################################
    ########################################### run network on audio files : ###########################################
    ####################################################################################################################

    run_prediction(input, output=output, modelTag=modelTag, verbose=True)

