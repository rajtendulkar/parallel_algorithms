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

int test_sequential_single_dim_ptr(unsigned int matrix_length)
{	
	float *mat_a, *mat_b, *mat_c;
	clock_t t1, t2;
	int return_value = 0;

	/* Allocate matrix A. */
	mat_a = allocate_single_dim_matrix(matrix_length);
	if (mat_a == NULL) {
		printf("Error allocating memory for matrix A\n");
		return_value = -1;
		goto mat_a_alloc_fail;
	}

	/* Allocate matrix B. */
	mat_b = allocate_single_dim_matrix(matrix_length);
	if (mat_b == NULL) {
		printf("Error allocating memory for matrix B\n");
		return_value = -1;
		goto mat_b_alloc_fail;
	}

	/* Allocate matrix C. */
	mat_c = allocate_single_dim_matrix(matrix_length);
	if (mat_c == NULL) {
		printf("Error allocating memory for matrix C\n");
		return_value = -1;
		goto mat_c_alloc_fail;
	}

	/* Initialize A matrix in row-major format. */
	initialize_single_dim_matrix(mat_a, matrix_length, 0);

#ifdef MATRIX_B_COLUMN_MAJOR_FORMAT
	/* Initialize the B matrix in col-major format. */
	initialize_single_dim_matrix(mat_b, matrix_length, 1);
#else
	/* Initialize the B matrix in row-major format. */
	initialize_single_dim_matrix(mat_b, matrix_length, 0);
#endif

#ifdef PRINT_OUTPUT
	printf("\n*********************\n");
	printf("Matrix A : \n");
	print_matrix(mat_a, matrix_length);

	printf("\n*********************\n");
	printf("Matrix B : \n");
	print_matrix(mat_b, matrix_length);
#endif /* PRINT_OUTPUT */

	t1 = clock();

	sequential_mat_mul_single_dim(mat_a, mat_b, mat_c, matrix_length);

	t2 = clock();

	printf("Sequential : %f (seconds)\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

#ifdef PRINT_OUTPUT
	printf("\n*********************\n");
	printf("Seq Matrix C : \n");
	print_matrix(mat_c, matrix_length);
	printf("\n*********************\n");
#endif /* PRINT_OUTPUT */
	
	mat_c = free_single_dim_matrix(mat_c);

mat_c_alloc_fail:
	mat_b = free_single_dim_matrix(mat_b);

mat_b_alloc_fail:
	mat_a = free_single_dim_matrix(mat_a);

mat_a_alloc_fail:
	return return_value;
}

