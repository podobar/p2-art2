import csv
import numpy as np

from ARTNetwork import ARTNetwork
from Art2Network import ART2


def load_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        _data = list(reader)

    for index in range(1, len(_data)):
        _data[index] = [float(item) for item in _data[index]]

    return _data


if __name__ == '__main__':
    path = 'clustering/cube.csv'
    raw_data = load_csv(path)[1:]
    classification = list()
    predictions = list()
    for i in range(len(raw_data)):
        classification.append(raw_data[i].pop())
    #network = ARTNetwork(_input_size=len(raw_data[0]), _init_output_size=2)
    #network.learn(data_set=raw_data, cycles=1200)

    network = ART2(len(raw_data[0]), 10)
    for i in range(10):
        for data in raw_data:
            network.present(data, True)

    for data in raw_data:
        predictions.append(network.present(data, False))

    print("Classes created:", np.max(predictions))
    print("Actual classes:", np.max(classification))
