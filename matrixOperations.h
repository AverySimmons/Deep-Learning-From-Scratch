#pragma once

#include <stdbool.h>
#include <stdio.h>


/*
adds two matrices in place.
*/
void matrixAdd(double* matrix1, double* matrix2, double* out_matrix, int matrix_size)
{
	for (int i = 0; i < matrix_size; i++) {
		out_matrix[i] = matrix1[i] + matrix2[i];
	}
}

/*
subtracts two matrices in place.
*/
void matrixSubtract(double* matrix1, double* matrix2, double* out_matrix, int matrix_size)
{
	for (int i = 0; i < matrix_size; i++) {
		out_matrix[i] = matrix1[i] - matrix2[i];
	}
}

/*
multiplies a matrix by a flat value in place.
*/
void matrixFlatMultiply(double* matrix, double scalar, double* out_matrix, int matrix_size)
{
	for (int i = 0; i < matrix_size; i++) {
		out_matrix[i] = matrix[i] * scalar;
	}
}

/*
multiplies two matricies in place.
*/
void matrixInPlaceMultiply(double* matrix1, double* matrix2, double* out_matrix, int matrix_size)
{
	for (int i = 0; i < matrix_size; i++) {
		out_matrix[i] = matrix1[i] * matrix2[i];
	}
}

/*
applies the ReLU function to a matrix.
*/
void matrixReLU(double* matrix, double* out_matrix, int matrix_size)
{
	for (int i = 0; i < matrix_size; i++) {
		out_matrix[i] = (matrix[i] > 0) ? matrix[i] : 0;
	}
}

/*
applies the ReLU prime function to a matrix.
*/
void matrixReLUPrime(double* matrix, double* out_matrix, int matrix_size)
{
	for (int i = 0; i < matrix_size; i++) {
		out_matrix[i] = (matrix[i] > 0) ? 1.0f : 0;
	}
}

/*
multiplies two matricies (not in place).
out_matrix = alpha * trans?(matrix1) * trans?(matrix2) + out_matrix * beta
transposition is applied after matricies have been created from row_first, size, and elements_per.
*/
void matrixMultiply(bool row_first_matrix1, bool row_first_matrix2, bool transpose_matrix1, bool transpose_matrix2, 
					double* matrix1, int matrix1_size, int matrix1_elements_per, double* matrix2, int matrix2_size, 
					int matrix2_elements_per, double* out_matrix, int out_matrix_size, bool row_first_out_matrix, double alpha, double beta)
{
	if (matrix1_size % matrix1_elements_per != 0 || matrix2_size % matrix2_elements_per != 0) {
		perror("Matrix size not divisible by elements per");
		return;
	}

	bool matrix1_flip = (row_first_matrix1 ^ transpose_matrix1);
	bool matrix2_flip = (row_first_matrix2 ^ transpose_matrix2);

	int matrix1_rows = (matrix1_flip) ? matrix1_size / matrix1_elements_per : matrix1_elements_per;
	int matrix1_columns = (matrix1_flip) ? matrix1_elements_per : matrix1_size / matrix1_elements_per;
	int matrix2_rows = (matrix2_flip) ? matrix2_size / matrix2_elements_per : matrix2_elements_per;
	int matrix2_columns = (matrix2_flip) ? matrix2_elements_per : matrix2_size / matrix2_elements_per;

	if (matrix1_columns != matrix2_rows || matrix1_rows * matrix2_columns != out_matrix_size) {
		perror("Tried matrix multiply with mismatched matricies");
		return;
	}

	for (int i = 0; i < matrix1_rows; i++) {
		for (int j = 0; j < matrix2_columns; j++) {
			double sum = 0;
			for (int k = 0; k < matrix1_columns; k++) {
				int matrix1_index = (matrix1_flip) ? i * matrix1_columns + k : k * matrix1_rows + i;
				int matrix2_index = (matrix2_flip) ? k * matrix2_columns + j : j * matrix2_rows + k;
				sum += matrix1[matrix1_index] * matrix2[matrix2_index];
			}
			int out_matrix_index = (row_first_out_matrix) ? i * matrix2_columns + j : j * matrix1_rows + i;
			out_matrix[out_matrix_index] = out_matrix[out_matrix_index] * beta + sum * alpha;
		}
	}
}