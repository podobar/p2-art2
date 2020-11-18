import csv
import numpy as np
from mlxtend.data import loadlocal_mnist
from Visualization import Visualization as V
from ARTNetwork import ARTNetwork
from Art2Network import ART2
from random import shuffle


def mix_data(data, n_class, samples_per_class, have_to_mix):
    if not have_to_mix:
        return data
    data_reordered = list()
    for i in range(samples_per_class):
        for j in range(n_class):
            data_reordered.append(data[i + j * samples_per_class])

    return data_reordered


def compute_means(raw_data):
    raw_data = np.transpose(raw_data)

    for i in range(len(raw_data)):
        mean = (np.max(raw_data[i]) - np.min(raw_data[i])) / 2
        means.append(mean)
    return means


def modify_data_for_clustering(raw_data, means, move):
    raw_data = np.transpose(raw_data)

    if move:
        for i in range(len(raw_data)):
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


if __name__ == '__main__':
    path = 'clustering/hexagon.csv'

    raw_data = load_csv(path)[1:]
    data_dim = len(raw_data[0]) - 1

    classification = list()
    predictions = list()

    shuffle(raw_data)
    #raw_data.reverse()
    #raw_data = mix_data(raw_data, 8, 150, True)
    for i in range(len(raw_data)):
       classification.append(raw_data[i].pop())

    means = list()
    means = compute_means(raw_data)

    new_data = raw_data #modify_data_for_clustering(raw_data, means, False)

    network = ART2(len(new_data[0]), 11)
    for i in range(50):
        for data in new_data:
            network.present(data, True)

    for data in new_data:
        predictions.append(network.present(data, False))

    if data_dim == 2:
        V.visual_2D_clusters(raw_data, classification, predictions)

    if data_dim == 3:
        V.visual_3D_clusters(raw_data, classification)
        V.visual_3D_clusters(raw_data, predictions)

    print("Classes created:", len(set(predictions)))
    print("Actual classes:", len(set(classification)))
