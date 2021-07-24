import tensorflow as tf
import numpy as np
from tensorflow.nn import convolution
from scipy.special import factorial
from matplotlib import pyplot as plt

from celluloid import Camera
from IPython.display import HTML, display
from enum import Enum


class DifferenceType(Enum):
    """
    Different difference types.
    """
    CENTRAL_DIFFERENCE = "central difference"
    FORWARD_DIFFERENCE = "forward difference"


class Maxwell3DFiniteDifference:
    """
    Solver for the 3 dimensional Maxwell equations.
    """

    def __init__(self, mesh_size, step_size, geometry=None, conductivity=None, permitivity=None, permeability=None,
                 frame_rate=8, diff_type: DifferenceType = DifferenceType.CENTRAL_DIFFERENCE):
        self.mesh_size = mesh_size
        self.step_size = step_size
        self.geometry = geometry
        self._number_divisions_per_axis = int(1 / self.mesh_size)
        self._E = np.zeros(shape=(
        self._number_divisions_per_axis, self._number_divisions_per_axis, self._number_divisions_per_axis, 3))
        self._I = np.zeros(shape=(
        self._number_divisions_per_axis, self._number_divisions_per_axis, self._number_divisions_per_axis, 3))
        self._B = np.zeros(shape=(
        self._number_divisions_per_axis, self._number_divisions_per_axis, self._number_divisions_per_axis, 3))
        self._p = np.zeros(shape=(
        self._number_divisions_per_axis, self._number_divisions_per_axis, self._number_divisions_per_axis, 1))

        axis_diskrete = np.linspace(0, 1, self._number_divisions_per_axis)
        mesh = np.stack(np.meshgrid(axis_diskrete, axis_diskrete))

        if conductivity:
            self._g = np.vectorize(conductivity)(*mesh)
        else:
            self._g = np.zeros(
                shape=(
                self._number_divisions_per_axis, self._number_divisions_per_axis, self._number_divisions_per_axis, 1))

        if permitivity:
            self._eps = np.vectorize(permitivity)(*mesh)
        else:
            self._eps = np.ones(shape=(
            self._number_divisions_per_axis, self._number_divisions_per_axis, self._number_divisions_per_axis, 1))

        if permeability:
            self._mu = np.vectorize(permeability)(*mesh)
        else:
            self._mu = np.ones(shape=(
            self._number_divisions_per_axis, self._number_divisions_per_axis, self._number_divisions_per_axis, 1))

        if diff_type == DifferenceType.CENTRAL_DIFFERENCE:
            pass
        else:
            Y = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])

            Z1 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            Y = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])

            Z2 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            X = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            Y = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])

            Z3 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            self._continuity_filter = np.concatenate(
                [Z1.reshape(3, 3, 3, 1), Z2.reshape(3, 3, 3, 1), Z3.reshape(3, 3, 3, 1)],
                axis=3).reshape(3, 3, 3, 3, 1)

            Y = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
            Z = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

            Z1 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), Z.reshape(3, 3, 1)], axis=2)

            Y = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

            Z2 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            F1 = np.concatenate([np.zeros((3, 3, 3, 1)), Z1.reshape(3, 3, 3, 1), Z2.reshape(3, 3, 3, 1)], axis=3)

            Y = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            Z = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])

            Z1 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), Z.reshape(3, 3, 1)], axis=2)

            Y = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
            Z = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

            Z2 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), Z.reshape(3, 3, 1)], axis=2)

            F2 = np.concatenate([Z1.reshape(3, 3, 3, 1), np.zeros((3, 3, 3, 1)), Z2.reshape(3, 3, 3, 1)], axis=3)

            Y = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

            Z1 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            Y = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])

            Z2 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            F3 = np.concatenate([Z1.reshape(3, 3, 3, 1), Z2.reshape(3, 3, 3, 1), np.zeros((3, 3, 3, 1))], axis=3)

            self._faraday_filter = np.concatenate(
                [F1.reshape(3, 3, 3, 3, 1), F2.reshape(3, 3, 3, 3, 1), F3.reshape(3, 3, 3, 3, 1)], axis=4) / self.mesh_size ** 2

            X = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
            Y = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

            Z1 = np.concatenate([X.reshape(3, 3, 1), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            Y = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])

            Z2 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            F1 = np.concatenate([np.zeros((3, 3, 3, 1)), Z1.reshape(3, 3, 3, 1), Z2.reshape(3, 3, 3, 1)], axis=3)

            X = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            Y = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])

            Z1 = np.concatenate([X.reshape(3, 3, 1), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            Y = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])

            Z2 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            F2 = np.concatenate([Z1.reshape(3, 3, 3, 1), np.zeros((3, 3, 3, 1)), Z2.reshape(3, 3, 3, 1)], axis=3)

            Y = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])

            Z1 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            Y = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])

            Z2 = np.concatenate([np.zeros((3, 3, 1)), Y.reshape(3, 3, 1), np.zeros((3, 3, 1))], axis=2)

            F3 = np.concatenate([Z1.reshape(3, 3, 3, 1), Z2.reshape(3, 3, 3, 1), np.zeros((3, 3, 3, 1))], axis=3)

            self._ampere_filter = np.concatenate(
                [F1.reshape(3, 3, 3, 3, 1), F2.reshape(3, 3, 3, 3, 1), F3.reshape(3, 3, 3, 3, 1)], axis=4) / self.mesh_size ** 2

        self.fig, self.axs = (None, None)
        self.camera = None
        self.frame_rate = frame_rate
        self._video_constant = 1 / frame_rate

    def evolve(self, initial_B, initial_E, initial_charge, integration_period, order=1, video=False):
        """
        Evolves an initial state of the electromagnetic field over a time interval "integration_period".

        :param initial_B: initial magnetic field, callable (x,y,z) -> (Bx,By,Bz).
        :param initial_E: initial electric field, callable (x,y,z) -> (Ex,Ey,Ez)
        :param initial_charge: initial charge density, callable (x,y,z) -> p.
        :param integration_period: period of time to evolve the electromagnetic field.
        :param order: order of the ODE solver.
        :param video: if True a video is generated in the ipyhton notebook.
        :return: final magnetic field, final electric field, final charge density
        """
        if order < 0:
            raise ValueError("Order must be non negative!")
        N = self._number_divisions_per_axis
        faraday_filter = tf.constant(self._faraday_filter, name='faraday_filter', dtype=tf.float64)
        ampere_filter = tf.constant(self._ampere_filter, name='ampere_filter', dtype=tf.float64)
        continuity_filter = tf.constant(self._continuity_filter, name='continuity_filter', dtype=tf.float64)

        axis_diskrete = np.linspace(0, 1, self._number_divisions_per_axis)
        mesh = np.stack(np.meshgrid(axis_diskrete, axis_diskrete, axis_diskrete))

        self._E[:, :, :, 0] = np.vectorize(lambda x, y, z: initial_E(x, y, z)[0])(*mesh)
        self._E[:, :, :, 1] = np.vectorize(lambda x, y, z: initial_E(x, y, z)[1])(*mesh)
        self._E[:, :, :, 2] = np.vectorize(lambda x, y, z: initial_E(x, y, z)[2])(*mesh)

        self._B[:, :, :, 0] = np.vectorize(lambda x, y, z: initial_B(x, y, z)[0])(*mesh)
        self._B[:, :, :, 1] = np.vectorize(lambda x, y, z: initial_B(x, y, z)[1])(*mesh)
        self._B[:, :, :, 2] = np.vectorize(lambda x, y, z: initial_B(x, y, z)[2])(*mesh)

        self._p = np.vectorize(initial_charge)(*mesh) * self.mesh_size ** 3

        eps = tf.constant(self._eps.reshape((1, N, N, N, 1)), name='dielectricity', dtype=tf.float64)
        mu = tf.constant(self._mu.reshape((1, N, N, N, 1)), name='permitivity', dtype=tf.float64)
        g = tf.constant(self._g.reshape((1, N, N, N, 1)), name='conductivity', dtype=tf.float64)
        E = tf.Variable(self._E.reshape((1, N, N, N, 3)), name='e_field', dtype=tf.float64)
        B = tf.Variable(self._B.reshape((1, N, N, N, 3)), name='b_field', dtype=tf.float64)
        p = tf.Variable(self._p.reshape((1, N, N, N, 1)), name='charge_density', dtype=tf.float64)

        if video:
            self.fig, self.axs = plt.subplots(2, 2)
            self.camera = Camera(self.fig)
            video_period = max(1, int(1 / (self.frame_rate * self.step_size)))
        else:
            video_period = np.inf

        for jter in range(int(integration_period / self.step_size)):
            dB = B
            dE = E
            for ord in range(1, order + 1):
                j = g * dE
                dB, dE = (convolution(dE, filters=faraday_filter, padding='SAME'),
                          convolution(dB / eps / mu, filters=ampere_filter, padding='SAME') - j / eps)
                dp = convolution(j * self.mesh_size, filters=continuity_filter, padding='SAME')

                E.assign_add(dE * self.step_size ** ord / factorial(ord))
                B.assign_add(dB * self.step_size ** ord / factorial(ord))
                p.assign_add(dp * self.step_size ** ord / factorial(ord))
            if video and (jter % video_period == 0):
                self._record(p / self.mesh_size ** 3, g * E * self.mesh_size, E, B)

        if video:
            self.generate_mp4()
        return tf.squeeze(B).numpy(), tf.squeeze(E).numpy(), tf.squeeze(g * E * self.mesh_size).numpy(), tf.squeeze(
            p).numpy() / self.mesh_size ** 3

    def _record(self, charge_density, current, electric_field, magnetic_field):
        """
        Records a state of the electromagnetic field.
        
        :param charge_density: tensorflow array.
        :param current: tensorflow array.
        :param electric_field: tensorflow array.
        :param magnetic_field: tensorflow array.
        :return: None
        """
        N = charge_density.shape[2] // 2
        charge_density = charge_density[:,:, N]
        current = current[:, :, N, :2]
        electric_field = electric_field[:, :, N, :2]
        magnetic_field = magnetic_field[:, :, N, :2]

        charge_density = np.flipud(tf.squeeze(charge_density).numpy())
        current = tf.squeeze(current).numpy()
        e_field = tf.squeeze(electric_field).numpy()
        magnetic_field = np.flipud(tf.squeeze(magnetic_field).numpy())
        arrow_num = int(current.shape[0] // 20)

        self.axs[0, 0].imshow(charge_density, vmin=-33, vmax=33)
        self.axs[0, 0].set_title('charge density')
        self.axs[0, 1].quiver(np.linspace(0, 1, int(current.shape[0] / arrow_num)),
                              np.linspace(0, 1, int(current.shape[0] / arrow_num)),
                              current[::arrow_num, ::arrow_num, 0],
                              current[::arrow_num, ::arrow_num, 1],
                              scale=0.2)
        self.axs[0, 1].set_title('current')
        self.axs[1, 0].imshow(magnetic_field, vmin=-1, vmax=1)
        self.axs[1, 0].set_title('magnetic field')
        self.axs[1, 1].quiver(np.linspace(0, 1, int(current.shape[0] / arrow_num)),
                              np.linspace(0, 1, int(current.shape[0] / arrow_num)),
                              e_field[::arrow_num, ::arrow_num, 0],
                              e_field[::arrow_num, ::arrow_num, 1],
                              scale=3)
        self.axs[1, 1].set_title('electric field')
        self.fig.set_figheight(10)
        self.fig.set_figwidth(10)
        for ax in self.fig.axes:
            ax.set_xticks([])
            ax.set_yticks([])
        self.camera.snap()

    def generate_mp4(self, ):
        anim = self.camera.animate(blit=True, interval=self._video_constant * 1e3)
        display(HTML(anim.to_html5_video()))
