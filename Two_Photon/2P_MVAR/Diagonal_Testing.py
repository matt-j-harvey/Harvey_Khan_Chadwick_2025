import numpy as np


matrix = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
]

matrix = np.array(matrix)
print("Shape", np.shape(matrix))


# Zeroing Diagonals
zeroed_matrix = np.zeros(np.shape(matrix))
np.fill_diagonal(a=zeroed_matrix, val=np.diag(matrix))
print("zeroed_matrix")
print(zeroed_matrix)
