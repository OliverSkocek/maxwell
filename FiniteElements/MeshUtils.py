import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def compute_face_gradient(face, values):
    return np.linalg.solve(np.concatenate([face, np.ones((3, 1))], axis=1), values)


def compute_area(face):
    return np.abs(np.linalg.det(face[1:, :] - face[0, :])) / 2


class ShapeFunction:
    """
    Class for implementing finite element shape functions
    """

    def __init__(self, transformator):
        self._transformator = transformator

    def value_and_gradient(self, x):
        """
        Returns the function values and the gradients for x.

        :param x: N x 2 array of positions.
        :return: Tuple of value array and gradient array.
        """
        value = np.zeros((x.shape[0],))
        gradient = np.zeros(x.shape)
        for trafo, b in self._transformator:
            h = np.matmul(x - b, trafo)
            grad = -np.sum(trafo, axis=1)
            fun_val = (1 - np.sum(h, axis=1))
            update = (h[:, 0] >= 0) * (h[:, 1] >= 0) * (fun_val >= 0)
            value[update] = fun_val[update]
            gradient[update, :] = grad.reshape(1, -1)
        return value, gradient


class Mesh:
    """
    Class containing data structures and functionality for
    building a mesh for finite element computation in two dimensions.
    """

    def __init__(self, number_of_divisions, geometry=None):
        self.geometry = geometry
        self.number_of_divisions = number_of_divisions
        self._vertices = self._generate_vertices(number_of_divisions)
        self.number_of_vertices = self._vertices.shape[0]
        self._elements = self._get_elements(number_of_divisions)
        self.number_of_elements = self._elements.shape[0]
        self.boundary_vertices = self._get_boundary()
        self._area = np.sqrt(3) / 4 * np.linalg.norm(self._vertices[1] - self._vertices[0]) ** 2

    @staticmethod
    def _get_elements(N):
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

        adjacent_elements = np.isin(self._elements, base_number)
        transformator = list()
        for position, element in zip(adjacent_elements, self._elements):
            if np.max(position):
                triangle = self._vertices[element[np.argsort(position)], :]
                b = triangle[-1, :]
                triangle = (triangle - b)[:-1, :]
                transformator.append((np.linalg.inv(triangle), b))

        return ShapeFunction(transformator=transformator)

    def _get_boundary(self,):
        """
        Returns the indexes of the boundary vertices.

        :return:
        """
        number_of_divisions = self.number_of_divisions
        return np.array(list(range(number_of_divisions + 1)) + list(
            range(number_of_divisions + 1, (number_of_divisions + 1) ** 2, number_of_divisions + 1)) + list(
            range(2 * number_of_divisions + 1, (number_of_divisions + 1) ** 2, number_of_divisions + 1)) + list(
            range(number_of_divisions * (number_of_divisions + 1) + 1, (number_of_divisions + 1) ** 2)))

    def get_common_elements(self, vertex_1, vertex_2):
        """
        Returns all elements that contain vertex_1 and vertex_2.

        :param vertex_1:
        :param vertex_2:
        :return:
        """
        return self._elements[
               np.max(np.isin(self._elements, vertex_1), axis=1) * np.max(np.isin(self._elements, vertex_2), axis=1), :]

    def get_overlap_integral(self, vertex_1, vertex_2):
        """
        Computes the overlap integral between the shape function located at vertex_1
        and vertex_2.

        :param vertex_1:
        :param vertex_2:
        :return:
        """
        common_elements = self.get_common_elements(vertex_1=vertex_1, vertex_2=vertex_2)
        return (2 if vertex_1 == vertex_2 else 1) * common_elements.shape[0] * self._area * (1 / 12)

    def get_gradient_overlap(self, vertex_1, vertex_2):
        """
        Computes the overlap integral between the gradients of the respective shape function located at vertex_0
        and vertex_1.

        :param vertex_1:
        :param vertex_2:
        :return:
        """
        common_elements = self.get_common_elements(vertex_1=vertex_1, vertex_2=vertex_2)
        phi = 0 if vertex_1 == vertex_2 else 2 / 3 * np.pi
        return np.cos(phi) * common_elements.shape[0] * self._area * 4 / 3

    def compute_source_vector(self, charge_density):
        """
        Computes the inhomogeneity for the Poisson Problem.

        :param charge_density: callable for the charge density.
        :return:
        """
        # TODO build faster version.
        source = np.vectorize(
            lambda n: sum(
                self.get_overlap_integral(n, m) * charge_density(*self._vertices[m, :].tolist()) for m in
                np.unique(self.get_common_elements(n, n)))
        )(np.arange(self.number_of_vertices))

        source[self.boundary_vertices] = 0.0
        return source

    def compute_finite_laplace(self, direct=False):
        """
        Computes the matrix elements for the Poisson Problem.

        :param direct: if True, laplacian is directly computed from get_gradient_overlap.
        :return:
        """
        if direct:
            index = np.arange(self.number_of_vertices)
            A = np.vectorize(self.get_gradient_overlap)(*np.meshgrid(index, index))
            eye = np.eye(A.shape[0])
            A[self.boundary_vertices.reshape(-1, 1), :] = eye[self.boundary_vertices.reshape(-1, 1), :]
            A[:, self.boundary_vertices.reshape(1, -1)] = eye[:, self.boundary_vertices.reshape(1, -1)]
            return A
        else:

            template = np.ones((self.number_of_divisions + 1,))

            diagonals = list()
            start = 3 * template
            start[[0, -1]] = np.array([1, 2])
            end = np.flip(start)
            middle = 6 * template
            middle[[0, -1]] = np.array([3, 3])
            d_0 = np.concatenate(
                [start] + [middle for _ in range(self.number_of_divisions + 1 - 2)] + [end]) * self._area * 4 / 3
            d_0[self.boundary_vertices] = 1.0
            diagonals.append(d_0)

            start = -0.5 * template
            start[-1] = 0.0
            end = np.flip(start)[1:]
            middle = -template
            middle[-1] = 0.0
            d_1 = np.concatenate(
                [start] + [middle for _ in range(self.number_of_divisions + 1 - 2)] + [end]) * self._area * 4 / 3
            b_exclude = self.boundary_vertices - 1
            b_exclude = b_exclude[b_exclude >= 0]
            d_1[b_exclude] = 0.0
            b_exclude = self.boundary_vertices[self.boundary_vertices < d_1.size]
            d_1[b_exclude] = 0.0
            diagonals += [d_1, d_1]

            start = -template
            start[0] = 0.0
            end = -np.ones((self.number_of_divisions + 2,))
            end[[0, -1]] = 0.0
            d_2 = np.concatenate([start for _ in range(self.number_of_divisions + 1 - 2)] + [end]) * self._area * 4 / 3
            b_exclude = self.boundary_vertices - self.number_of_divisions
            b_exclude = b_exclude[b_exclude >= 0]
            d_2[b_exclude] = 0.0
            b_exclude = self.boundary_vertices[self.boundary_vertices < d_2.size]
            d_2[b_exclude] = 0.0
            diagonals += [d_2, d_2]

            start = -template
            start[[0, -1]] = -0.5
            d_3 = np.concatenate([start for _ in range(self.number_of_divisions + 1 - 1)]) * self._area * 4 / 3
            b_exclude = self.boundary_vertices - self.number_of_divisions - 1
            b_exclude = b_exclude[b_exclude >= 0]
            d_3[b_exclude] = 0.0
            b_exclude = self.boundary_vertices[self.boundary_vertices < d_3.size]
            d_3[b_exclude] = 0.0
            diagonals += [d_3, d_3]

            return sparse.diags(diagonals, [0, 1, -1, self.number_of_divisions, -self.number_of_divisions,
                                            self.number_of_divisions + 1,
                                            -(self.number_of_divisions + 1)])

    def solve(self, charge_density, direct=False):
        """
        Solves the Poisson problem with zero dirichlet boundary condition given a charge density.

        :param charge_density: callable for the charge density.
        :param direct: if True, laplacian is directly computed from get_gradient_overlap.
        :return:
        """
        if direct:
            return np.linalg.solve(A=self.compute_finite_laplace(direct), b=self.compute_source_vector(charge_density))
        else:
            return spsolve(A=self.compute_finite_laplace(direct), b=self.compute_source_vector(charge_density))



        # TODO apply geometry
