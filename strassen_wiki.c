#include "matrix_mul.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>

/*
 * Source code taken from wikipedia, for comparison. 
 */
static void sum_two_dim(float **a, float **b, float **result, int tam);
static void subtract_two_dim(float **a, float **b, float **result, int tam);
static float **allocate_real_matrix_two_dim(int tam);
static float **free_real_matrix_two_dim(float **v, int tam);

static void strassen_two_dim(float **a, float **b, float **c, int tam) {

	// trivial case: when the matrix is 1 X 1:
	if (tam == 1) {
		c[0][0] = a[0][0] * b[0][0];
		return;
	}

	// other cases are treated here:
	else {
		int newTam = tam / 2;
		float **a11, **a12, **a21, **a22;
		float **b11, **b12, **b21, **b22;
		float **c11, **c12, **c21, **c22;
		float **p1, **p2, **p3, **p4, **p5, **p6, **p7;

		// memory allocation:
		a11 = allocate_real_matrix_two_dim(newTam);
		a12 = allocate_real_matrix_two_dim(newTam);
		a21 = allocate_real_matrix_two_dim(newTam);
		a22 = allocate_real_matrix_two_dim(newTam);

		b11 = allocate_real_matrix_two_dim(newTam);
		b12 = allocate_real_matrix_two_dim(newTam);
		b21 = allocate_real_matrix_two_dim(newTam);
		b22 = allocate_real_matrix_two_dim(newTam);

		c11 = allocate_real_matrix_two_dim(newTam);
		c12 = allocate_real_matrix_two_dim(newTam);
		c21 = allocate_real_matrix_two_dim(newTam);
		c22 = allocate_real_matrix_two_dim(newTam);

		p1 = allocate_real_matrix_two_dim(newTam);
		p2 = allocate_real_matrix_two_dim(newTam);
		p3 = allocate_real_matrix_two_dim(newTam);
		p4 = allocate_real_matrix_two_dim(newTam);
		p5 = allocate_real_matrix_two_dim(newTam);
		p6 = allocate_real_matrix_two_dim(newTam);
		p7 = allocate_real_matrix_two_dim(newTam);

		float **aResult = allocate_real_matrix_two_dim(newTam);
		float **bResult = allocate_real_matrix_two_dim(newTam);

		int i, j;

		//dividing the matrices in 4 sub-matrices:
		for (i = 0; i < newTam; i++) {
			for (j = 0; j < newTam; j++) {
				a11[i][j] = a[i][j];
				a12[i][j] = a[i][j + newTam];
				a21[i][j] = a[i + newTam][j];
				a22[i][j] = a[i + newTam][j + newTam];

				b11[i][j] = b[i][j];
				b12[i][j] = b[i][j + newTam];
				b21[i][j] = b[i + newTam][j];
				b22[i][j] = b[i + newTam][j + newTam];
			}
		}

		// Calculating p1 to p7:

		sum_two_dim(a11, a22, aResult, newTam); // a11 + a22
		sum_two_dim(b11, b22, bResult, newTam); // b11 + b22
		strassen_two_dim(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)

		sum_two_dim(a21, a22, aResult, newTam); // a21 + a22
		strassen_two_dim(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)

		subtract_two_dim(b12, b22, bResult, newTam); // b12 - b22
		strassen_two_dim(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)

		subtract_two_dim(b21, b11, bResult, newTam); // b21 - b11
		strassen_two_dim(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)

		sum_two_dim(a11, a12, aResult, newTam); // a11 + a12
		strassen_two_dim(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)

		subtract_two_dim(a21, a11, aResult, newTam); // a21 - a11
		sum_two_dim(b11, b12, bResult, newTam); // b11 + b12
		strassen_two_dim(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)

		subtract_two_dim(a12, a22, aResult, newTam); // a12 - a22
		sum_two_dim(b21, b22, bResult, newTam); // b21 + b22
		strassen_two_dim(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)

												// calculating c21, c21, c11 e c22:

		sum_two_dim(p3, p5, c12, newTam); // c12 = p3 + p5
		sum_two_dim(p2, p4, c21, newTam); // c21 = p2 + p4

		sum_two_dim(p1, p4, aResult, newTam); // p1 + p4
		sum_two_dim(aResult, p7, bResult, newTam); // p1 + p4 + p7
		subtract_two_dim(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7

		sum_two_dim(p1, p3, aResult, newTam); // p1 + p3
		sum_two_dim(aResult, p6, bResult, newTam); // p1 + p3 + p6
		subtract_two_dim(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6

											// Grouping the results obtained in a single matrix:
		for (i = 0; i < newTam; i++) {
			for (j = 0; j < newTam; j++) {
				c[i][j] = c11[i][j];
				c[i][j + newTam] = c12[i][j];
				c[i + newTam][j] = c21[i][j];
				c[i + newTam][j + newTam] = c22[i][j];
			}
		}

		// deallocating memory (free):
		a11 = free_real_matrix_two_dim(a11, newTam);
		a12 = free_real_matrix_two_dim(a12, newTam);
		a21 = free_real_matrix_two_dim(a21, newTam);
		a22 = free_real_matrix_two_dim(a22, newTam);

		b11 = free_real_matrix_two_dim(b11, newTam);
		b12 = free_real_matrix_two_dim(b12, newTam);
		b21 = free_real_matrix_two_dim(b21, newTam);
		b22 = free_real_matrix_two_dim(b22, newTam);

		c11 = free_real_matrix_two_dim(c11, newTam);
		c12 = free_real_matrix_two_dim(c12, newTam);
		c21 = free_real_matrix_two_dim(c21, newTam);
		c22 = free_real_matrix_two_dim(c22, newTam);

		p1 = free_real_matrix_two_dim(p1, newTam);
		p2 = free_real_matrix_two_dim(p2, newTam);
		p3 = free_real_matrix_two_dim(p3, newTam);
		p4 = free_real_matrix_two_dim(p4, newTam);
		p5 = free_real_matrix_two_dim(p5, newTam);
		p6 = free_real_matrix_two_dim(p6, newTam);
		p7 = free_real_matrix_two_dim(p7, newTam);
		aResult = free_real_matrix_two_dim(aResult, newTam);
		bResult = free_real_matrix_two_dim(bResult, newTam);
	} // end of else

} // end of Strassen function

  /*------------------------------------------------------------------------------*/
  // function to sum two matrices
static void sum_two_dim(float **a, float **b, float **result, int tam) {

	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			result[i][j] = a[i][j] + b[i][j];
		}
	}
}

/*------------------------------------------------------------------------------*/
// function to subtract two matrices
static void subtract_two_dim(float **a, float **b, float **result, int tam) {

	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			result[i][j] = a[i][j] - b[i][j];
		}
	}
}

/*------------------------------------------------------------------------------*/
// This function allocates the matrix using malloc, and initializes it. If the variable random is passed
// as zero, it initializes the matrix with zero, if it's passed as 1, it initializes the matrix with random
// values. If it is passed with any other int value (like -1 for example) the matrix is initialized with no
// values in it. The variable tam defines the length of the matrix.
static float **allocate_real_matrix_two_dim(int tam) {
	int i;
	float **v = NULL;         // pointer to the vector

					   // allocates one vector of vectors (matrix)
	v = (float**)malloc(tam * sizeof(float*));

	if (v == NULL) {
		printf("** Error in matrix allocation: insufficient memory **");		
	} else {

		// allocates each row of the matrix
		for (i = 0; i < tam; i++) {
			v[i] = (float*)malloc(tam * sizeof(float));

			if (v[i] == NULL) {
				printf("** Error: Insufficient memory **");
				free_real_matrix_two_dim(v, tam);
				return (NULL);
			}
		}
	}

	return (v);     // returns the pointer to the vector.
}

/*------------------------------------------------------------------------------*/
// This function unallocated the matrix (frees memory)
static float **free_real_matrix_two_dim(float **v, int tam) {

	int i;

	if (v == NULL) {
		return (NULL);
	}

	for (i = 0; i < tam; i++) {
		if (v[i]) {
			free(v[i]); // frees a row of the matrix
			v[i] = NULL;
		}
	}

	free(v);         // frees the pointer /
	v = NULL;

	return (NULL);   //returns a null pointer /
}

static void sequential_implementation_two_dim(float **mat_a, float **mat_b, float **mat_c, unsigned int length)
{
	for (unsigned int i = 0; i<length; i++)
		for (unsigned int j = 0; j<length; j++) {
			mat_c[i][j] = mat_a[i][j];
			for (unsigned int k = 0; k<length; k++)
				mat_c[i][j] += mat_a[i][k] * mat_b[k][j];
		}
}

static void initialize_matrix_two_dim(float **input_matrix, unsigned int length)
{
	static unsigned char initializer = 1;

	if (input_matrix == NULL || length == 0) {
		printf("%s : Invalid arguments\n", __func__);
		return;
	}

	for (unsigned int i = 0; i<length; i++)
		for (unsigned int j = 0; j<length; j++)
			input_matrix[i][j] = initializer++;
}

int test_strassen_wiki(unsigned int matrix_length)
{
	int return_value = 0;
	float **mat_a, **mat_b, **mat_c;
	clock_t t1, t2, t3;

	/* Allocate matrix A. */
	mat_a = allocate_real_matrix_two_dim(matrix_length);
	if (mat_a == NULL) {
		printf("Error allocating memory for matrix A\n");
		return_value = -1;
		goto mat_a_alloc_fail;
	}

	/* Allocate matrix B. */
	mat_b = allocate_real_matrix_two_dim(matrix_length);
	if (mat_b == NULL) {
		printf("Error allocating memory for matrix B\n");
		return_value = -1;
		goto mat_b_alloc_fail;
	}

	/* Allocate matrix C. */
	mat_c = allocate_real_matrix_two_dim(matrix_length);
	if (mat_c == NULL) {
		printf("Error allocating memory for matrix C\n");
		return_value = -1;
		goto mat_c_alloc_fail;
	}

	initialize_matrix_two_dim(mat_a, matrix_length);
	initialize_matrix_two_dim(mat_b, matrix_length);

#ifdef PRINT_OUTPUT
	printf("\n*********************\n");
	printf("Matrix A : \n");
	print_matrix_two_dim(mat_a, matrix_length);

	printf("\n*********************\n");
	printf("Matrix B : \n");
	print_matrix_two_dim(mat_b, matrix_length);
#endif /* PRINT_OUTPUT */

	t1 = clock();
	
	sequential_implementation_two_dim(mat_a, mat_b, mat_c, matrix_length);
	
	t2 = clock();

#ifdef PRINT_OUTPUT
	printf("\n*********************\n");
	printf("Matrix C : \n");
	print_matrix_two_dim(mat_c, matrix_length);
	printf("\n*********************\n");
#endif /* PRINT_OUTPUT */


	for (int i = 0; i<NUM_ITERATIONS; i++) {
		strassen_two_dim(mat_a, mat_b, mat_c, matrix_length);
		sum_two_dim(mat_c, mat_a, mat_c, matrix_length);
	}

	t3 = clock();

	printf("Sequential : %f (seconds) Strassen : %f (seconds)\n",
		(double)(t2 - t1) / CLOCKS_PER_SEC,
		(double)(t3 - t2) / (CLOCKS_PER_SEC * NUM_ITERATIONS));

#ifdef PRINT_OUTPUT
	printf("Matrix C : \n");
	print_matrix_two_dim(mat_c, matrix_length);
	printf("\n*********************\n");
#endif /* PRINT_OUTPUT */

	mat_c = free_real_matrix_two_dim(mat_c, matrix_length);

mat_c_alloc_fail:
	mat_b = free_real_matrix_two_dim(mat_b, matrix_length);

mat_b_alloc_fail:
	mat_a = free_real_matrix_two_dim(mat_a, matrix_length);

mat_a_alloc_fail:
	return return_value;
}

/*------------------------------------------------------------------------------*/