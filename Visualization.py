import matplotlib.pyplot as plt


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