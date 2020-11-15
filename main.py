import csv
import numpy as np
from mlxtend.data import loadlocal_mnist
from Visualization import Visualization as V
from ARTNetwork import ARTNetwork
from Art2Network import ART2


def mix_data(data, n_class, samples_per_class, have_to_mix):
    if not have_to_mix:
        return data
    data_reordered = list()
    for i in range(samples_per_class):
        for j in range(n_class):
            data_reordered.append(data[i + j * samples_per_class])

    return data_reordered


def modify_data_for_clustering(raw_data, move):
    if move:
        raw_data = np.transpose(raw_data)

        for i in range(len(raw_data)):
            if len(means) > 0:
                mean = (np.max(raw_data[i]) - np.min(raw_data[i])) / 2
                means.append(mean)
            else:
                mean = means[i]
            raw_data[i] = np.add(raw_data[i], -mean)


        raw_data = np.transpose(raw_data)
    new_data = list()

    for data in raw_data:

        for i in range(len(data)):
            if data[i] >= 0:
                data = np.append(data, 0)
            else:
                data = np.append(data, -data[i])
                data[i] = 0
        new_data.append(data)
    return new_data


def load_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        _data = list(reader)

    for index in range(1, len(_data)):
        _data[index] = [float(item) for item in _data[index]]

    return _data


def data_to_compare(path, move):
    raw_data = load_csv(path)[1:]

    classification = list()
    for i in range(len(raw_data)):
       classification.append(raw_data[i].pop())

    data = modify_data_for_clustering(raw_data, move=move)

    return data, classification


if __name__ == '__main__':
    train_x_path = 'HAPT/Train/X_train.txt'
    train_y_path = 'HAPT/Train/y_train.txt'
    train_subjects_path = 'HAPT/Train/subject_id_train.txt'

    test_x_path = 'HAPT/Test/X_test.txt'
    test_y_path = 'HAPT/Test/y_test.txt'
    test_subjects_path = 'HAPT/Test/subject_id_test.txt'

    train_x = load_csv(train_x_path)
    train_y = load_csv(train_y_path)






