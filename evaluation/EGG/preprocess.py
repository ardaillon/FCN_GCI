
from pysndfile import sndio
import os

def low_pass_butter(sig, order = 5, cutOff = 500, fs = 44100):
    '''
    low-pass filtering using a butterworth filter
    '''
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    cutOff = cutOff / nyq
    (b,a) = butter(order, cutOff, btype='lowpass')
    sig_lp = filtfilt(b, a, sig, axis=0)
    return sig_lp

def high_pass_butter(sig, order = 5, cutOff = 8000, fs = 44100):
    '''
    low-pass filtering using a butterworth filter
    '''
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    cutOff = cutOff / nyq
    (b,a) = butter(order, cutOff, btype='highpass')
    sig_hp = filtfilt(b, a, sig, axis=0)
    return sig_hp

if __name__ == '__main__':

    inEggDir = '../../examples/EGG'
    outCleanEggDir = '../../examples/EGG_processed'

    if(not os.path.isdir(outCleanEggDir)):
        os.makedirs(outCleanEggDir)

    eggFiles = os.listdir(inEggDir)

    nbFiles = len(eggFiles)
    i = 0

    for f in eggFiles:
        if(f.endswith('.wav')):
            i += 1
            print("processing file "+str(i)+" / "+str(nbFiles))
            inEggFile = os.path.join(inEggDir, f)
            outEggFile = os.path.join(outCleanEggDir, f)

            # load egg signal
            (egg, sr, enc) = sndio.read(inEggFile)

            # low-pass signal to remove noise
            egg_lp = low_pass_butter(egg, order = 5, cutOff = 500, fs = sr)

            # high-pass filter signal to remove slow fluctuations
            egg_filt = high_pass_butter(egg_lp, order = 5, cutOff = 30, fs = sr)

            sndio.write(outEggFile, egg_filt, sr, 'wav', enc)

