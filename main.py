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
            mean = (np.max(raw_data[i]) - np.min(raw_data[i])) / 2
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
    path1 = 'clustering/cube.csv'
    path2 = 'clustering/cube-notmatching.csv'

    # train_X, train_y = loadlocal_mnist(
    #     images_path='MNIST\\train-images.idx3-ubyte',
    #     labels_path='MNIST\\train-labels.idx1-ubyte')

    raw_data = load_csv(path1)[1:]
    data_dim = len(raw_data[0]) - 1

    classification = list()
    predictions = list()

    for i in range(len(raw_data)):
       classification.append(raw_data[i].pop())

    #raw_data = mix_data(raw_data, 8, 150, True)
    #network = ARTNetwork(_input_size=len(raw_data[0]), _init_output_size=2)
    #network.learn(data_set=raw_data, cycles=1200)

    new_data = modify_data_for_clustering(raw_data, move=False)

    network = ART2(len(new_data[0]), 10)
    for i in range(1):
        for data in new_data:
            network.present(data, True)

    compares, classification2 = data_to_compare(path2, False)

    for data in compares:
        predictions.append(network.present(data, False))

    if data_dim == 2:
        V.visual_2D_clusters(raw_data, classification, predictions)
    if data_dim == 3:
        V.visual_3D_clusters(compares, classification2)
        V.visual_3D_clusters(compares, predictions)

    set1 = set(predictions)
    print("Classes created:", len(set1))
    print("Actual classes:", len(set(classification)))
