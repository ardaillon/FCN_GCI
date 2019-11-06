
import csv


def write_csv_file(filename, times, labels):
    '''
    Save GCI markers in csv format
    :param filename: file name where to store the detected GCIs
    :param times: times of detected GCIs
    :param labels: labels (not really useful, just use 'GCI')
    :return: nothing
    '''
    with open(filename, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['times', 'labels'])
        for (t,l) in zip(times, labels):
            filewriter.writerow([str(t), l])
    return

