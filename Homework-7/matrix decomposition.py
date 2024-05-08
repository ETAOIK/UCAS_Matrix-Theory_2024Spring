# Author: Chenkai GUO
# Student ID: 202328017515010
# Date: 2024.5.3

import numpy as np
import sympy
from sympy import Matrix

def full_rank_decomp(A: np.ndarray):
    # obtain Row-Echelon Form and rank of matrix A
    A_mat = Matrix(A)
    A_rref = np.array(A_mat.rref()[0])
    rank = np.linalg.matrix_rank(A)

    # obtain col_seq of matrix A
    col_seq = []
    for i in range(A_rref.shape[0]):
        for j in range(A_rref.shape[1]):
            if A_rref[i][j] == 1:
                col_seq.append(j)

    # obtain final matrix F and G satisfy A=F@G
    F = A[:, col_seq]
    G = A_rref[:rank, :]

    return F, G

def SVD(A: np.ndarray):
    # calculate eigenvalue and eigenvector of A.H@A
    # to determine the SVD of a matrix, we should obtain a U or V first
    # and then derivate the other one by this one.
    row, col = A.shape
    # here we derivate matrix V first
    B = np.mat(A).H @ np.mat(A)
    eigenvalue, eigenvector = np.linalg.eig(B)
    singular_argsort = np.argsort(np.round(eigenvalue, 6))[::-1]
    singulars = np.sqrt(np.sort(np.round(eigenvalue, 6))[::-1])
    singular_matrix = np.diag(singulars[singulars > 0])

    rank = np.linalg.matrix_rank(A)
    V = eigenvector[:, singular_argsort]

    V_1 = V[:, :rank]
    U_1 = A @ V_1 @ np.linalg.inv(singular_matrix)
    U = np.pad(U_1, pad_width=((0, 0), (0, row - rank)))
    singular_matrix = np.pad(singular_matrix, pad_width=((0, row - rank), (0, col - rank)))

    return U, singular_matrix, V


if __name__ == "__main__":
    matrix_1 = np.array([[-1,0,1,2],[1,2,-1,1],[2,2,-2,-1]])
    matrix_2 = np.array([[0, 0, 1], [2, 1, 1], [2j, 1j, 0]])
    matrix_3 = np.array([[1,2,1,0,1],[0,1,1,0,1],[1,3,2,0,2],[1,2,1,1,1]])
    print(full_rank_decomp(matrix_1))
    print(full_rank_decomp(matrix_2))
    print(SVD(matrix_3))
