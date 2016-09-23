#include "matrix_mul.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef OPENMP
#include <omp.h>
#endif /* OPENMP */

#ifdef INTEL_SIMD_INSTRUCTIONS
#include <immintrin.h>
#endif

void initialize_single_dim_matrix(float *input_matrix, unsigned int length, unsigned int col_major)
{
	static unsigned char initializer = 0;
	if (col_major) {
		for (unsigned int i = 0; i<length; i++)
			for (unsigned int j = 0; j<length; j++, initializer %= 10)
				input_matrix[j * length + i] = initializer++;
	}
	else {
		for (unsigned int i = 0; i<length*length; i++, initializer %= 10)
			input_matrix[i] = initializer++;
	}
}

float *allocate_single_dim_matrix(int length)
{
	float *matrix = (float*)malloc(length * length * sizeof(float));
	if (matrix == NULL)
		printf("Error in matrix allocation : insufficient memory\n");
	return matrix;
}

float *free_single_dim_matrix(float *matrix)
{
	if (matrix == NULL)
		printf("Invalid argument to free a matrix\n");
	else
		free(matrix);
	return NULL;
}

void sequential_mat_mul_single_dim(float *mat_a, float *mat_b, float *mat_c, unsigned int length)
{
	for (unsigned int i = 0; i<length; i++)
		for (unsigned int j = 0; j<length; j++) {
			mat_c[i*length + j] = mat_a[i*length + j];
			for (unsigned int k = 0; k<length; k++)
#ifdef MATRIX_B_COLUMN_MAJOR_FORMAT
				mat_c[i*length + j] += mat_a[i*length + k] * mat_b[j*length + k];
#else
				mat_c[i*length + j] += mat_a[i*length + k] * mat_b[k*length + j];
#endif
		}
}