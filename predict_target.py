
import numpy as np
import warnings
# from pysndfile import sndio
from scipy.io import wavfile


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


def get_audio(sndFile, model_input_size = 993, model_sr=16000.):
    '''
    Load and pre-process (make mono, resample, normalize, and pad) audio from input sound file
    :param sndFile: input sound file
    :param model_input_size: minimum input size of the model (necessary for padding input sound to have correct duration in output)
    :return: audio, sound encoding tag
    '''

    # read sound :
    (sr, audio) = wavfile.read(sndFile)
    # (audio, sr, enc) = sndio.read(sndFile)

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

    return audio


def segment_sound(snd, segDuration = 60., frame_sizes = 993, totalNetworkPoolingFactor = 8, model_sr=16000.):
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


def speech_to_target(inFile, model, model_input_size=993, maxSndSegLenSec=60., totalNetworkPoolingFactor=8, verbose=True, model_sr=16000.):

    # read and pad the audio from file :
    snd = get_audio(inFile, model_input_size, model_sr=model_sr)

    sndSegments = segment_sound(snd, segDuration=maxSndSegLenSec, frame_sizes=model_input_size,
                                totalNetworkPoolingFactor=totalNetworkPoolingFactor, model_sr=model_sr)  #  segment sound in chunks for analysis

    activations = []

    for iseg, sndSeg in enumerate(sndSegments):
        audio = np.array([np.reshape(sndSeg, (len(sndSeg), 1, 1))])
        activationsSeg = model.predict(audio, verbose=verbose)
        activationsSeg = np.reshape(activationsSeg, (np.shape(activationsSeg)[1]))
        activations = np.concatenate((activations, activationsSeg))

    time_src = np.arange(len(activations)) * totalNetworkPoolingFactor
    time_target = np.arange(len(snd) - model_input_size + 1)
    interpType = 'cubicSpline'
    if (interpType == 'linear'):
        targetPred = np.interp(time_target, time_src, activations)
    elif (interpType == 'cubicSpline'):
        from scipy.interpolate import CubicSpline
        if (np.isnan(activations).any()):
            warnings.warn("activations contains nan values")
            noNaN_ids = np.where(np.isnan(activations) == False)[0]
            time_src = time_src[noNaN_ids]
            activations = activations[noNaN_ids]
        interpolator = CubicSpline(time_src, activations)
        targetPred = interpolator(time_target)
    return targetPred

