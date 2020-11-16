import csv
import numpy as np
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


def modify_data_for_clustering(raw_data):
    # raw_data = np.transpose(raw_data)
    #
    # for i in range(len(raw_data)):
    #     mean = (np.max(raw_data[i]) - np.min(raw_data[i])) / 2
    #     raw_data[i] = np.add(raw_data[i], -mean)
    #
    # raw_data = np.transpose(raw_data)
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
    path = 'clustering/mnist_test.csv'

    # train_X, train_y = loadlocal_mnist(
    #     images_path='MNIST\\train-images.idx3-ubyte',
    #     labels_path='MNIST\\train-labels.idx1-ubyte')

    raw_data = load_csv(path)
    data_dim = len(raw_data[0]) - 1
    raw_data = sorted(raw_data, key=lambda x: int(x[0]))
    classification = list()
    predictions = list()

    for i in range(len(raw_data)):
        classification.append(int(raw_data[i].pop(0)))
        raw_data[i] = list(map(int, raw_data[i]))
    #raw_data = mix_data(raw_data, 8, 150, True)
    #network = ARTNetwork(_input_size=len(raw_data[0]), _init_output_size=2)
    #network.learn(data_set=raw_data, cycles=1200)

    #new_data = modify_data_for_clustering(raw_data)
    new_data=raw_data
    network = ART2(len(new_data[0]), 10)
    # for i in range(50):
    #     for data in new_data:
    #         network.present(data, True)

    for data in new_data:
        predictions.append(network.present(data, False))

    if data_dim == 2:
        V.visual_2D_clusters(raw_data, classification, predictions)
    if data_dim == 3:
        V.visual_3D_clusters(raw_data, classification)
        V.visual_3D_clusters(raw_data, predictions)
    if data_dim == 28*28:
        V.show_MNIST_bitmap(raw_data[0])
        V.show_LTM_state(network.T)
        V.show_LTM_state(network.B)

    print("Classes created:", np.max(predictions)+1)
    print("Actual classes:", np.max(classification)+1)
    print("End")
