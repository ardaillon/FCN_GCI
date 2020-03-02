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
try:
    from pysndfile import sndio
except:
    from scipy.io import wavfile
from models.core import load_model
import re
from target_to_GCI import get_gci_times
try:
    from fileio.sdif import mrk # ircam's library for our own use, not available from github
except:
    from file_utils.fileio import write_csv_file
from predict_target import speech_to_target
import numpy as np


# model is trained for a sampling rate of 16000Hz
model_sr = 16000.


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

    targetPred = speech_to_target(inFile, model, model_input_size=model_input_size, maxSndSegLenSec=maxSndSegLenSec, totalNetworkPoolingFactor=totalNetworkPoolingFactor, verbose=verbose, model_sr=model_sr)

    targetFile = get_output_path(inFile, ".targetPred.wav", output)
    outputDir = os.path.dirname(targetFile)
    if(not os.path.isdir(outputDir)):
        os.makedirs(outputDir)
    print("saving target shape analysis to file "+targetFile)
    try:
        sndio.write(targetFile, targetPred, model_sr, 'wav', 'pcm16')
    except:
        wavfile.write(targetFile, int(model_sr), targetPred)

    GCI_times = get_gci_times(targetPred, model_sr, mode=mode)

    try:
        GCIFile = targetFile.replace('.targetPred.wav', '.GCI.sdif')
        mrk.store(GCIFile, GCI_times, ['GCI'] * len(GCI_times))
        print("saved GCI markers in file "+GCIFile)
    except:
        GCIFile = targetFile.replace('.targetPred.wav', '.GCI.csv')
        write_csv_file(GCIFile, GCI_times, ['GCI'] * len(GCI_times))
        print("saved GCI markers in file "+GCIFile)

    return


if __name__ == '__main__':
    ####################################################################################################################
    ############################################# Command-line interface : #############################################
    ####################################################################################################################
    from argparse import ArgumentParser
    parser = ArgumentParser(description="run prediction of target shape using the given model on given files or directory")

    global_args = parser.add_argument_group('global')
    parser.add_argument('-i', "--input", nargs='+', default='./examples/speech16k-norm', help='path to one ore more WAV file(s) to analyze OR can be a directory')
    global_args.add_argument('-o', "--output", default=None, help='output GCI analysis (either a directory or a file with sdif extension)')
    global_args.add_argument('-m', "--modelTag", default='FCN_synth_GF', help='name model to be used for prediction (default is FCN_synth_GF)')

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

