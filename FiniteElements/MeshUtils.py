import numpy as np


def generate_vertices(N):
    M = np.array([[1, 0], [1 / 2, np.sqrt(3) / 2]]).T / N
    line = np.arange(N + 1)
    return np.concatenate(
        np.tensordot(M, np.stack(np.meshgrid(line, line)), (1, 0)).T, axis=0)


def get_faces(N):
    up = np.array([[j + i * (N + 1), j + i * (N + 1) + 1, j + i * (N + 1) + N + 1]
                   for i in range(N) for j in range(N)])
    down = np.array(
        [[i + 1 + j * (N + 1), i + (N + 1) + j * (N + 1), i + (N + 2) + j * (N + 1)]
         for j in range(N) for i in range(N)])
    return np.concatenate([np.stack([u, d]) for u, d in zip(up, down)], axis=0)


def get_common_faces(vertex_1, vertex_2, faces):
    return faces[np.max(np.isin(faces, vertex_1), axis=1) * np.max(np.isin(faces, vertex_2), axis=1), :]


def compute_face_gradient(face, values):
    return np.linalg.solve(np.concatenate([face, np.ones((3, 1))], axis=1), values)


def compute_area(face):
    return np.abs(np.linalg.det(face[1:, :] - face[0, :])) / 2


def get_boundary(N):
    return list(range(N + 1)) + list(range(N + 1, (N + 1) ** 2, N + 1)) + list(
        range(2 * N + 1, (N + 1) ** 2, N + 1)) + list(range(N * (N + 1) + 1, (N + 1) ** 2))
