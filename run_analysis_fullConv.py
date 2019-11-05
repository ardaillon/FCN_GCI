from dnn_mods.gputils import lock_gpu
import os
import sys
import numpy as np
from pysndfile import sndio
import warnings
from models.load_model import load_model
import re

# model is trained for a sampling rate of 16000Hz
model_sr = 16000.

def db2lin(vec) :
    return 10**(vec/20.)

def normalize_snd_file(inSnd, level = -3):
    maxValIn = np.max(np.abs(inSnd))
    maxValOut = db2lin(level)
    scaleFactor = maxValOut / maxValIn
    normSnd = inSnd * scaleFactor
    return normSnd

def segment_sound(snd, segDuration = 60., frame_sizes = 993, totalNetworkPoolingFactor = 8):
    '''
    Segment input sound file into smaller chunks to avoid memory issues
    :param snd: input sound
    :param segDuration: maximum duration of segments (in s)
    :param frame_sizes:
    :param totalNetworkPoolingFactor:
    :return:
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

def run_prediction(filename, output = None, modelTag = 'FCN_synth_tri', verbose = True, plot = False):
    """
    Collect the sound files to process and run the prediction on each file
    Parameters
    ----------
    filename : list
        List containing paths to sound files (wav or aiff) or folders containing sound files to
        be analyzed.
    output : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    verbose : bool
        Print status messages and keras progress (default=True).
    """

    # load model:
    load_from_json = False

    if(load_from_json):
        model = load_model(modelTag, from_json=True)
    else:
        model = load_model(modelTag)

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
        run_prediction_on_file(file, output=output, model=model, plot=plot, verbose=verbose)
    return


def get_audio(sndFile, model_input_size = 993):

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
    """
    return the output path of an output file corresponding to a wav file
    """

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


def run_prediction_on_file(inFile, output=None, model=None, maxSndSegLenSec = 60, model_input_size = 993, totalNetworkPoolingFactor = 8, plot=False, verbose=True):
    '''

    :param file:
    :param output:
    :param model:
    :param maxSndSegLenSec: default is 60s. Analyse segments of a maximum duration of maxSndSegLenSec seconds to avoid running out of memory
    :param plot:
    :param verbose:
    :return:
    '''

    if(model==None):
        raise('FCN_GCI: model is None')

    # read and pad the audio from file :
    (snd, enc) = get_audio(inFile, model_input_size)

    sndSegments = segment_sound(snd, segDuration=maxSndSegLenSec, frame_sizes=model_input_size, totalNetworkPoolingFactor=totalNetworkPoolingFactor) # segment sound in chunks for analysis

    activations = []

    for iseg, sndSeg in enumerate(sndSegments):

        audio = np.array([np.reshape(sndSeg, (len(sndSeg),1,1))])
        activationsSeg = model.predict(audio, verbose=1)
        activationsSeg = np.reshape(activationsSeg, (np.shape(activationsSeg)[1]))
        activations = np.concatenate((activations, activationsSeg))

    time_src = np.arange(len(activations))*totalNetworkPoolingFactor
    time_target = np.arange(len(snd)-model_input_size+1)
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


    targetFile = get_output_path(inFile, ".targetPred.wav", output)

    outputDir = os.path.dirname(targetFile)
    if(not os.path.isdir(outputDir)):
        os.makedirs(outputDir)

    print("saving target shape analysis to file "+targetFile)
    sndio.write(targetFile, activations, model_sr, 'wav', enc)

    return


if __name__ == '__main__':
    ####################################################################################################################
    ############################################# Command-line interface : #############################################
    ####################################################################################################################
    from argparse import ArgumentParser
    parser = ArgumentParser(description="run analysis of a given model on a given database")

    global_args = parser.add_argument_group('global')
    parser.add_argument('-i', "--input", nargs='+', default='./examples/speech16k-norm', help='path to one ore more WAV file(s) to analyze OR can be a directory')
    global_args.add_argument('-o', "--output", default='./examples/pred_target', help='output f0 analysis (either a directory or a file with sdif extension)')
    global_args.add_argument('-m', "--modelTag", default='FCN_synth_tri', help='name model to be used for prediction (default is FCN_synth_tri)')

    args = parser.parse_args()

    input = args.input
    output = args.output
    modelTag = args.modelTag

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

    run_prediction(input, output=output, modelTag=modelTag, verbose=True, plot=False)

