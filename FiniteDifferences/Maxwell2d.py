import tensorflow as tf
import numpy as np
from tensorflow.nn import convolution
from scipy.special import factorial

class Maxwell2DFiniteDifference:
    """
    Solver for the 2 dimensional Maxwell equations.
    """

    def __init__(self, mesh_size, step_size, geometry=None, resistance=None, permitivity=None, permeability=None):
        self.mesh_size = mesh_size
        self.step_size = step_size
        self.geometry = geometry
        self._number_divisions_per_axis = int(1 / self.mesh_size)
        self._U = np.zeros(shape=(self._number_divisions_per_axis, self._number_divisions_per_axis, 2))
        self._I = np.zeros(shape=(self._number_divisions_per_axis, self._number_divisions_per_axis, 2))
        self._B = np.zeros(shape=(self._number_divisions_per_axis, self._number_divisions_per_axis, 1))
        self._p = np.zeros(shape=(self._number_divisions_per_axis, self._number_divisions_per_axis, 1))
        self._R = np.inf * np.ones(shape=(self._number_divisions_per_axis, self._number_divisions_per_axis, 1))
        self._eps = np.ones(shape=(self._number_divisions_per_axis, self._number_divisions_per_axis, 1))
        self._mu = np.ones(shape=(self._number_divisions_per_axis, self._number_divisions_per_axis, 1))
        X = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        Y = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        self._continuity_filter = np.stack([X, Y]).reshape((2, 3, 3, 1)).transpose(1, 2, 0, 3)
        X = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        Y = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        self._faraday_filter = np.stack([X, Y]).reshape((2, 3, 3, 1)).transpose(1, 2, 0, 3) / self.mesh_size ** 2
        X = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        Y = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        self._ampere_filter = np.stack([X, Y]).reshape((2, 3, 3, 1)).transpose(1, 2, 3, 0) * self.mesh_size

    def evolve(self, initial_B, initial_E, initial_charge, time, order=1):
        """
        Evolves an initial state of the electromagnetic field over a time interval.

        :param initial_B: initial magnetic field, callable (x,y) -> B.
        :param initial_E: initial electric field, callable (x,y) -> (Ex,Ey)
        :param initial_charge: initial charge density, callable (x,y) -> p.
        :param time: period of time to evolve the electromagnetic field.
        :param order: order of the ODE solver.
        :return: final magnetic field, final electric field, final charge density
        """
        if order < 0:
            raise ValueError("Order must be non negative!")
        N = self._number_divisions_per_axis
        faraday_filter = tf.constant(self._faraday_filter, name='faraday_filter', dtype=tf.float64)
        ampere_filter = tf.constant(self._ampere_filter, name='ampere_filter', dtype=tf.float64)
        continuity_filter = tf.constant(self._continuity_filter, name='continuity_filter', dtype=tf.float64)

        axis_diskrete = np.linspace(0, 1, self._number_divisions_per_axis)
        mesh = np.stack(np.meshgrid(axis_diskrete, axis_diskrete))

        self._U[:, :, 0] = np.vectorize(lambda x, y: initial_E(x, y)[0])(*mesh) * self.mesh_size
        self._U[:, :, 1] = np.vectorize(lambda x, y: initial_E(x, y)[1])(*mesh) * self.mesh_size
        self._I = self._U / self._R
        self._B = np.vectorize(initial_B)(*mesh)
        self._p = np.vectorize(initial_charge)(*mesh) * self.mesh_size ** 2

        eps = tf.constant(self._eps.reshape((1, N, N, 1)), name='dielectricity', dtype=tf.float64)
        mu = tf.constant(self._mu.reshape((1, N, N, 1)), name='permitivity', dtype=tf.float64)
        R = tf.constant(self._R.reshape((1, N, N, 1)), name='resistance', dtype=tf.float64)
        U = tf.Variable(self._U.reshape((1, N, N, 2)), name='e_field', dtype=tf.float64)
        I = tf.Variable(self._I.reshape((1, N, N, 2)), name='current', dtype=tf.float64)
        B = tf.Variable(self._B.reshape((1, N, N, 1)), name='b_field', dtype=tf.float64)
        p = tf.Variable(self._p.reshape((1, N, N, 1)), name='charge_density', dtype=tf.float64)

        for jter in range(int(time / self.step_size)):
            dB = B
            dU = U
            for ord in range(1, order + 1):
                I = dU / R
                dB, dU = (convolution(dU, filters=faraday_filter, padding='SAME'),
                          convolution(dB, filters=ampere_filter, padding='SAME') / eps / mu - I / eps)
                dp = convolution(I, filters=continuity_filter, padding='SAME')

                U.assign_add(dU * self.step_size ** ord / factorial(ord))
                B.assign_add(dB * self.step_size ** ord / factorial(ord))
                p.assign_add(dp * self.step_size ** ord / factorial(ord))
        return tf.squeeze(B).numpy(), tf.squeeze(U).numpy() / self.mesh_size, tf.squeeze(I).numpy, tf.squeeze(
            p).numpy() / self.mesh_size ** 2
