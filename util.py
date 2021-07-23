import sys
import copy

def normalize_matrix(matrix):
	agg = 0.
	entries = len(matrix)
	for i in range(entries):
		for j in range(entries):
			agg += matrix[i][j]
	matrix_copy = copy.copy(matrix)
	for i in range(entries):
		for j in range(entries):
			matrix_copy[i][j] = matrix[i][j] / agg
	return matrix_copy
