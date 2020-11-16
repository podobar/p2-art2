import csv
import numpy as np
from Visualization import Visualization as V
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
    train_path = 'clustering/mnist_train.csv'
    test_path = 'clustering/mnist_test.csv'

    #Optional sorting of learning data
    # raw_data = sorted(load_csv(train_path), key=lambda x: int(x[0]))
    train_data = load_csv(train_path)
    test_data = load_csv(test_path)
    data_dim = len(train_data[0]) - 1
    classification = list()
    predictions = list()

    for i in range(len(train_data)-1, -1, -1):
        # if train_data[i][0] == 8 or train_data[i][0] == 9:  #Optional deleting classes of '8' and '9' for further testing on unknown data
        # if train_data[i][0] == 0 or train_data[i][0] == 1:  # Optional deleting classes of '0' and '1' for further testing on unknown data
        # if train_data[i][0] == 5 or train_data[i][0] == 6:  # Optional deleting classes of '5' and '6' for further testing on unknown data
        # if train_data[i][0] == 1 or train_data[i][0] == 2: # Optional deleting classes of '1' and '2' for further testing on unknown data
        #     train_data.pop(i)
        # else:
        train_data[i].pop(0)
        train_data[i] = list(map(int, train_data[i]))
    for i in range(len(test_data)):
        test_data[i] = list(map(int, test_data[i]))
        classification.append(test_data[i].pop(0))

    network = ART2(data_dim, 10)
    #Learn
    for data in train_data:
        network.present(data, learn=True)
    #Test
    for data in test_data:
        predictions.append(network.present(data, learn=False))

    if data_dim == 2:
        V.visual_2D_clusters(train_data, classification, predictions)
    if data_dim == 3:
        V.visual_3D_clusters(train_data, classification)
        V.visual_3D_clusters(train_data, predictions)
    if data_dim == 28*28:
        V.show_LTM_state(network.T)
        V.show_clusterization_results(classification, predictions)

    print("Classes created:", np.max(predictions)+1)
    print("Actual classes:", np.max(classification)+1)
    print("End")
