import numpy as np
from dataclasses import dataclass


def get_common_faces(vertex_1, vertex_2, faces):
    return faces[np.max(np.isin(faces, vertex_1), axis=1) * np.max(np.isin(faces, vertex_2), axis=1), :]


def compute_face_gradient(face, values):
    return np.linalg.solve(np.concatenate([face, np.ones((3, 1))], axis=1), values)


def compute_area(face):
    return np.abs(np.linalg.det(face[1:, :] - face[0, :])) / 2


def get_boundary(N):
    return list(range(N + 1)) + list(range(N + 1, (N + 1) ** 2, N + 1)) + list(
        range(2 * N + 1, (N + 1) ** 2, N + 1)) + list(range(N * (N + 1) + 1, (N + 1) ** 2))


@dataclass
class ShapeFunction:
    """
    Container for implementing shape-functions.
    """
    value: callable
    gradient: callable


class Mesh:
    """
    Class containing data structures and functionality for
    building a mesh for finite element computation in two dimensions.
    """

    def __init__(self, number_of_divisions, geometry=None):
        self.geometry = geometry
        self._vertices = self._generate_vertices(number_of_divisions)
        self.number_of_elements = self._vertices.shape[0]
        self._faces = self._get_faces(number_of_divisions)
        self.number_of_faces = self._faces.shape[0]

    @staticmethod
    def _get_faces(N):
        up = np.array([[j + i * (N + 1),
                        j + i * (N + 1) + 1,
                        j + i * (N + 1) + N + 1]
                       for i in range(N) for j in range(N)])
        down = np.array(
            [[i + 1 + j * (N + 1),
              i + (N + 1) + j * (N + 1),
              i + (N + 2) + j * (N + 1)]
             for j in range(N) for i in range(N)])
        return np.concatenate(
            [np.stack([u, d]) for u, d in zip(up, down)], axis=0)

    @staticmethod
    def _generate_vertices(number_of_divisions):
        """
        Computes vertices of a mesh of equilateral triangles for a
        parallelogram (with (0, 0) - (1, 0) as base and 60° angle).

        :param number_of_divisions: number of subdivisions of the
        parallelogram per axis.
        :return: N x 2 matrix of vertices.
        """
        M = np.array([[1, 0], [1 / 2, np.sqrt(3) / 2]]).T / number_of_divisions
        line = np.arange(number_of_divisions + 1)
        return np.concatenate(
            np.tensordot(M, np.stack(np.meshgrid(line, line)), (1, 0)).T, axis=0)

    def get_shape_function(self, base_number):
        """
        Returns a function handle for the shape function, its gradient
        and its support.

        :param base_number: index of the desired base function.
        :return: callable, callable, support.
        """
        if base_number > self.number_of_elements - 1:
            raise ValueError(f'base_number needs to be between 0 '
                             f'and {self.number_of_elements - 1}')

        adjacent_faces = np.isin(self._faces, base_number)
        transformator = list()
        for position, face in zip(adjacent_faces, self._faces):
            if np.max(position):
                triangle = self._vertices[face[np.argsort(position)], :]
                b = triangle[-1, :]
                triangle = (triangle - b)[:-1, :]
                transformator.append((np.linalg.inv(triangle), b))

        def shape_function(x):
            result = np.zeros((x.shape[0],))
            for trafo, b in transformator:
                h = np.matmul(x - b, trafo)
                fun_val = 1 - np.sum(h, axis=1)
                result += np.heaviside(h[:, 0], 1) * \
                          np.heaviside(h[:, 1], 1) * \
                          np.heaviside(-result, 1) * \
                          np.heaviside(fun_val, 1) * fun_val
            return result

        def gradient(x):
            result = np.zeros(x.shape)
            for trafo, b in transformator:
                h = np.matmul(x - b, trafo)
                grad = np.sum(trafo, axis=1)
                fun_val = 1 - np.sum(h, axis=1)
                result += np.heaviside(h[:, :1], 1) * \
                          np.heaviside(h[:, 1:], 1) * \
                          np.heaviside(-result, 1) * \
                          np.heaviside(fun_val.reshape(-1, 1), 1) * grad
            return result

        return ShapeFunction(value=shape_function, gradient=gradient)
