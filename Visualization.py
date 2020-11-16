import matplotlib.pyplot as plt
import numpy as np

class Visualization:

    @staticmethod
    def visual_3D_clusters(data3D, classes):
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        PointX = list()
        PointY = list()
        PointZ = list()

        for data in data3D:
            PointX.append(data[0])
            PointY.append(data[1])
            PointZ.append(data[2])

        ax.scatter3D(PointX, PointY, PointZ
                     , c=classes  # color value of individual points is taken from their heights
                     , cmap="hsv"  # the color mapping to be used. Other example options: winter, autumn, ...
                     )
        plt.show()

    @staticmethod
    def visual_2D_clusters(data2D, classes, predicted):
        fig, axs = plt.subplots(1, 2)
        fig.tight_layout(pad=3.0)

        axs[0].set_title('Predictions')
        axs[1].set_title('Test set')

        PointX = list()
        PointY = list()

        for data in data2D:
            PointX.append(data[0])
            PointY.append(data[1])

        colors_predicted = predicted
        colors_test = classes

        axs[0].scatter(PointX, PointY, c=colors_predicted, cmap='cool', marker=".")
        axs[1].scatter(PointX, PointY, c=colors_test, cmap='cool', marker=".")
        plt.show()

    @staticmethod
    def show_MNIST_bitmap(data: list):
        plt.imshow(np.reshape(np.asarray(data),(28,28)), cmap='gray')
        plt.show()

    @staticmethod
    def show_LTM_state(LTM: list):
        N = len(LTM[0])
        M = len(LTM)
        fig=plt.figure(figsize=(28,28))
        for i in range(M):
            img = np.reshape(np.asarray(LTM[i]), (28,28))
            fig.add_subplot(1, M, i+1)
            plt.imshow(img,cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()

    @staticmethod
    def show_clusterization_results(classification: list, predictions: list):
        MAX_PREDICTION = max(predictions)
        matches = np.zeros(MAX_PREDICTION+2)
        for i in range(len(classification)):
            if predictions[i] == -1: #Correctly classified
                matches[predictions[MAX_PREDICTION+1]] += 1
            else:
                matches[predictions[i]] += 1
        plt.bar(range(MAX_PREDICTION+2), matches)
        for index,value in enumerate(matches ):
            plt.text(index-0.36, value+10, '{:5.0f}'.format(value))
        plt.title('Podsumowanie klastryzacji')
        plt.xlabel('Numer klastra')
        plt.ylabel('Ilość dopasowań')
        plt.show()