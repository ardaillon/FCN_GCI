# from pysndfile import sndio
from scipy.io import wavfile
from scipy.signal import find_peaks
from fileio.sdif import mrk
import numpy as np
from matplotlib import pyplot as plt


def get_gci_triangle(targetPred, sr, thresh=0.5):
    '''
    Detect GCI positions using peak-picking on predicted triangle target curve
    :param targetPred: predicted triangle target curve
    :param sr: samling rate of predicted target curve (should be 16kHz)
    :return: positions of detected GCIs, in s
    '''
    fmax = 1000.
    minPeriodSamples = int(sr / fmax)

    prominence = 0.25 * thresh
    width = int(minPeriodSamples / 4)
    (peaks_pos, peaks_properties) = find_peaks(targetPred, height=thresh, distance=minPeriodSamples, prominence=prominence, width=width)
    GCI_times = peaks_pos / sr

    return GCI_times


def filter_peaks_for_no_sign_change_in_dgf_between_peaks(peaks, dgf, posDgfThresh = 0.005):
    '''

    :param peaks:
    :param dgf:
    :param posDgfThresh:
    :return:
    '''
    group = []
    peakGroups = []
    group.append(peaks[0])
    for i,p in enumerate(peaks[1:]):
        i += 1
        dgf_seg = dgf[peaks[i-1]:p]
        if(np.max(dgf_seg)<posDgfThresh):
            group.append(p)
        else:
            peakGroups.append(group)
            group = [p]

        if(i==len(peaks[1:])):
            peakGroups.append(group)

        filtered_peaks = []

    for g in peakGroups:
        filtered_peaks.append(int(np.round(np.mean(g))))

    return np.array(filtered_peaks)


def get_gci_GF(targetPred, sr, PLOT=False):
    '''
    Detect GCI positions using peak-picking on the negative derivative of the predicted glottal flow target curve
    :param targetPred: predicted glottal flow target curve
    :param sr: samling rate of predicted target curve (should be 16kHz)
    :param PLOT: plot glottal flow and derivative along with detected GCIs
    :return: positions of detected GCIs, in s
    '''

    maxF0 = 500.
    # margin_fact = 0.9
    margin_fact = 1.
    min_period_sec = margin_fact * (1 / maxF0)

    minGFHeightAtGCI = 0.2

    dgf = np.diff(targetPred)

    distance_samp = int(np.floor(min_period_sec * sr))
    minPeaksWidth = int(round((sr / maxF0) / 8.))
    peaks_res = find_peaks(-1 * dgf * (targetPred[:-1] > minGFHeightAtGCI), height=0.01, distance=distance_samp, width=minPeaksWidth)[0]

    peaks_res = filter_peaks_for_no_sign_change_in_dgf_between_peaks(peaks_res, dgf)

    if (PLOT):
        plt.figure()
        plt.plot(targetPred, 'g')
        plt.plot(dgf, 'r')
        plt.plot(dgf * (targetPred[:-1] > minGFHeightAtGCI), 'b')
        plt.plot(peaks_res, dgf[peaks_res], 'xk')
        plt.show()

    sndTimes = np.arange(len(dgf)) / sr
    GCI_times = sndTimes[peaks_res]

    return GCI_times


def get_gci_times(targetPred, sr, mode='GF', thresh=0.5):

    if(mode=='triangle'):
        GCI_times = get_gci_triangle(targetPred, sr, thresh=thresh)

    elif(mode=='GF'):
        GCI_times = get_gci_GF(targetPred, sr)

    return GCI_times


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="run analysis of a given model on a given database")

    global_args = parser.add_argument_group('global')
    global_args.add_argument('-i', "--input", default='/data/anasynth/ardaillon/pulse_pred/results/BDD_pulse_pred_16k/testModel_497-triangle_target-patience_64-use_zones-red_lr_patience_10-capacity_coeff_2/analysis/manual', help='input sound data to be analysed (either a directory or a file')
    global_args.add_argument('-o', "--output", default='~/deepF0/test', help='output (either a directory or a file with sdif extension)')
    global_args.add_argument('-t', "--thresh", default=0.5, type=float, help='peak heights threshold for determining GCIs positions from predicted target curve')
    global_args.add_argument('-m', "--mode", default='GF', help='mode for GCI detection from target curve, either triangle or GF (as the process is different depending on the type of target curve predicted by the model)')

    args = parser.parse_args()

    inputFile = args.input
    outputFile = args.output
    mode = args.mode
    thresh = args.thresh

    if(inputFile.endswith('.wav')):
        print("processing file "+inputFile)

        # (targetPred, sr, enc) = sndio.read(inputFile)
        (sr, targetPred) = wavfile.read(inputFile)

        GCI_times = get_gci_times(targetPred, sr, mode=mode, thresh=thresh)

        mrk.store(outputFile, GCI_times, ['GCI'] * len(GCI_times))

