'''
Code for evaluation of GCI detection, using the "custom" metrics that consider all GCIs in ground truth
(regardless of the pitch and voiced/unvoiced detection)
These are the metrics presented in table 2 of the paper
"GCI detection from raw speech using a fully-convolutional network" (https://arxiv.org/abs/1910.10235)

The metrics used in table 1 are based on the code from https://github.com/VarunSrivastavaIITD/DCNN, (file "metrics.py")
following the definitions of : Patrick A Naylor, Anastasis Kounoudes, Jon Gudnason, and
Mike Brookes, “Estimation of Glottal Closure Instants in Voiced Speech Using the DYPSA Algorithm,”
'''

import os
from fileio.sdif import mrk
import numpy as np
from fileio.iovar import save_var

def get_false_alarm_rate(gci_gt, pairing_idx):
    '''
    False alarm rate (FAR = % of detected GCI that don't correspond to a ground truth GCI)
    :param gci_gt: ground truth GCIs
    :param pairing_idx: index of the ground truth GCI matched with the detected GCI
    :return: Computed False Alarm Rate (FAR)
    '''
    nb_gci_gt = len( gci_gt)
    nb_duplicates = len(pairing_idx) - len(np.unique(pairing_idx))
    false_alarm_rate = nb_duplicates * 100 / nb_gci_gt
    return false_alarm_rate

def get_identification_rate(gci_gt, pairing_idx):
    '''
    Identification Rate (IDR = % of ground truth GCIs that are uniquely paired with detected GCIs)
    :param gci_gt: ground truth GCIs
    :param pairing_idx: index of the ground truth GCI matched with the detected GCI
    :return: Computed Identification Rate (IDR)
    '''
    # TODO : Should this measure be corrected to incorporate all the closest paired GCIs (and not only the one that are uniquely paired?), and to take into account a maximal distance based on the f0?
    nb_gci_gt = len(gci_gt)
    nb_unique_id = len([pi for pi in pairing_idx if list(pairing_idx).count(pi) == 1])
    identification_rate = nb_unique_id * 100 / nb_gci_gt
    return identification_rate

def get_miss_rate(gci_gt, pairing_idx):
    '''
    Miss Rate (MR = % of ground truth GCI that are not paired with any detected GCI)
    # TODO : Should this measure be corrected to take into account a maximal distance based on the f0?
    :param gci_gt: ground truth GCIs
    :param pairing_idx: index of the ground truth GCI matched with the detected GCI
    :return: Computed Miss Rate (MR)
    '''
    miss_idx = np.delete(gci_gt, pairing_idx)
    nb_miss = len(miss_idx)
    nb_gci_gt = len(gci_gt)
    miss_rate = nb_miss * 100 / nb_gci_gt
    return miss_rate

def get_identification_errors(gci_gt, gci_ana, pairing_idx):
    '''
    Identification error (distance, in ms, between a truly detected GCI and the corresponding ground truth GCI)
    :param gci_gt: ground truth GCIs
    :param gci_ana: detected GCIs
    :param pairing_idx: index of the ground truth GCI matched with the detected GCI
    :return: Computed identification error for each truly detected GCI, in ms
    '''
    unique_ids = [pi for pi in pairing_idx if list(pairing_idx).count(pi) == 1]
    identification_errors = [np.min(np.abs(gci_ana - gci_gt[ui]))*1000 for ui in unique_ids]
    return identification_errors

def get_identification_accuracy(identification_errors):
    '''
    Identification Accuracy (IDA = standard deviation of the identification errors)
    Also compute the mean, but not used in presented results
    :param identification_errors: computed identification errors
    :return: Computed Identification Accuracy (IDA = std of identification errors. Also return the mean)
    '''
    identification_accuracy = {}
    identification_accuracy['mean'] = np.mean(identification_errors)
    identification_accuracy['std'] = np.std(identification_errors)
    return identification_accuracy

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Get the evaluation metrics for the detected GCIs, comparing to the ground truth for a given database")

    global_args = parser.add_argument_group('global')
    global_args.add_argument('-gt', "--ground_truth", default='/data2/anasynth/ardaillon/corpus/CMU_artic/cmu_us_bdl_arctic/GCI/GCI_gt', help='directory with ground truth gci files')
    global_args.add_argument('-i', "--input", default=None, help='directory with analyzed gci files')
    global_args.add_argument('-o', "--output", default=None, help='output pickle file containing results')

    args = parser.parse_args()

    suffixe = '.gci.sdif'

    groundTruthDir = args.ground_truth
    analysisDir = args.input
    results_file = args.output

    ana_files = os.listdir(analysisDir)
    nbFiles = len(ana_files)

    results = {'files': [], # pairs of ground truth and analyzed gci files
                'IDR': [], # identification rate
                'MR': [], # miss rate
                'FAR': [], # false alarm rate
                'IDE': [], # identification errors
                'IDA_mean': [], # identification accuracy (mean of identification errors, in ms)
                'IDA_std': [], # identification accuracy (std of identification errors, in ms)
                'global_IDR': None, # global identification rate (average on all files)
                'global_MR': None, # global miss rate (average on all files)
                'global_FAR': None, # global false alarm rate (average on all files)
                'global_IDA': None # global identification accuracy, in ms (average on all files)
                }

    file_id = 0
    for f in ana_files:
        file_id += 1
        print("processing file " + str(file_id) + " / " + str(nbFiles))
        print(f)
        if(f.endswith(suffixe)):
            ana_f = os.path.join(analysisDir, f)
            (gci_ana, labs_ana) = mrk.load_new(ana_f)
            gci_ana = np.array(gci_ana)
            gt_f = os.path.join(groundTruthDir, f)
            (gci_gt, labs_gt) = mrk.load_new(gt_f)
            gci_gt = np.array(gci_gt)

            pairing_idx = np.ones(len(gci_ana), dtype=int)*-1

            for i,t in enumerate(gci_ana):
                closest_gt_ind = np.argmin(np.abs(gci_gt - t))
                pairing_idx[i] = closest_gt_ind

            false_alarm_rate = get_false_alarm_rate(gci_gt, pairing_idx)
            identification_rate = get_identification_rate(gci_gt, pairing_idx)
            miss_rate = get_miss_rate(gci_gt, pairing_idx)
            identification_errors = get_identification_errors(gci_gt, gci_ana, pairing_idx)
            identification_accuracy = get_identification_accuracy(identification_errors)

            results['files'].append([gt_f, ana_f])
            results['IDR'].append(identification_rate)
            results['MR'].append(miss_rate)
            results['FAR'].append(false_alarm_rate)
            results['IDE'].append(identification_errors)
            results['IDA_mean'].append(identification_accuracy['mean'])
            results['IDA_std'].append(identification_accuracy['std'])

    global_IDR = np.mean(results['IDR'])
    results['global_IDR'] = global_IDR
    print('global_IDR = '+str(global_IDR))
    global_MR = np.mean(results['MR'])
    results['global_MR'] = global_MR
    print('global_MR = '+str(global_MR))
    global_FAR = np.mean(results['FAR'])
    results['global_FAR'] = global_FAR
    print('global_FAR = '+str(global_FAR))
    global_IDA = np.mean(results['IDA_std'])
    results['global_IDA'] = global_IDA
    print('global_IDA = '+str(global_IDA))

    save_var(results_file, results)

