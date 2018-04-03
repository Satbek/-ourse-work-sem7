import numpy as np
from ReportTools.Experiment import Experiment
from lib import data
from lib.haar_wawelet_method import haar, gradients as grad
import os
import progressbar


class WaweletMethodFrequencyCharackteristickExperiment(Experiment):
    path = os.path.abspath(os.getcwd() + "../../../ExperimentResults/WaweletMethodFrequencyCharackteristickExperiment")
    print(path)

    def __init__(self, name, reported=True, grid_degree=7):
        super().__init__(name, reported)
        x, y = data.get_plane(-np.pi, np.pi, -np.pi, np.pi, grid_degree)
        self._plane = {'x': x, 'y': y}
        self.exp = lambda x, y, w1, w2: np.exp(w1 * 1j * x + w2 * 1j * y)
        self.M = grid_degree
        self.KotelmikovLimit = 2 ** self.M // 10 + 1

    def _get_one_charackteristick(self, i, j, noised=False, photons=100):
        im = self.exp(self._plane['x'], self._plane['y'], i, j)
        grad_X, grad_Y = grad.fried_model_gradient(im)
        X_H, Y_H = grad.Hudgin_gradien_model(im)
        if (noised):
            grad_X = data.get_Poisson_noise(grad_X.real, photons) \
                     + 1j * data.get_Poisson_noise(grad_X.imag, photons)
            grad_Y = data.get_Poisson_noise(grad_Y.real, photons) \
                     + 1j * data.get_Poisson_noise(grad_Y.imag, photons)
            X_H = data.get_Poisson_noise(X_H.real, photons) \
                  + 1j * data.get_Poisson_noise(X_H.imag, photons)
            Y_H = data.get_Poisson_noise(Y_H.real, photons) \
                  + 1j * data.get_Poisson_noise(Y_H.imag, photons)
        LH, HL, HH = haar.analyze(grad_X, grad_Y, X_H, Y_H)
        res = haar.syntesis(
            {0: np.array([[np.mean(im) * 2 ** self.M]])}, LH, HL, HH, self.M)
        return np.abs(np.fft.fft2(im)[j, i]) / \
               np.abs(np.fft.fft2(res[self.M])[j, i])

    # todo decorators for noised and for ideal and difference case
    @Experiment._execute_decorator
    def execute(self, noised=False, photons=100):
        """
        выполняет эксперимент
        :param noised: зашумленное изображение или нет
        :param photons: количество фотонов, "сила шума"
        :return:
        """
        size = self.KotelmikovLimit
        z = np.zeros(size ** 2).reshape(size, size)
        with progressbar.ProgressBar(max_value=size ** 2) as bar:
            for i in range(size):
                for j in range(size):
                    z[i, j] = self._get_one_charackteristick(
                        i, j, noised, photons)
                    bar.update(i * size + j)
        return z

    # todo try, catch
    def save(self):
        """
        saves expreriment in the path directory
        """

        date = str(self.Datetime.timestamp())
        # it should be a property
        path_to_dir = self.__class__.path + "/" + self.name + date
        print(path_to_dir)
        os.mkdir(path_to_dir)
        with open(
                path_to_dir + "/" + self.name + ":" + date + ".txt", 'w'
        ) as f:
            f.write("Name:" + self.Name + "\n")
            f.write("Description:" + self.Description + "\n")
            f.write("Report:" + self.Report + "\n")
            f.write("Grid degree:" + str(self.M) + "\n")
            f.write("Kotelmikov Limit:" + str(self.KotelmikovLimit) + "\n")
            f.write("Date:" + self.Datetime.__str__() + "\n")
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # turn off summarization, line-wrapping
        with open(path_to_dir + "/" + self.name + "Result.txt", 'w') as f:
            f.write(np.array2string(self.Result, separator=', '))
        self.figure.savefig(path_to_dir + "/" + "3dplot.png")

    def draw_plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        from mpl_toolkits.mplot3d import Axes3D
        """
        draws default 3D - plot
        :return:
        """
        size = self.Result.shape[0]
        x, y = np.meshgrid(range(0, size), range(0, size))
        fig = plt.figure(figsize=(30, 30))
        self.figure = fig
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(x, y, self.Result)
        ax.set_zlim(0, 2)
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        major_locator = MultipleLocator(2)
        major_formatter = FormatStrFormatter('%d')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_formatter)
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_major_formatter(major_formatter)
