#include "matrix_mul.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>


#ifdef VERIFY_OUTPUT
int verify_quad_output(float*seq_matrix, MatQuad *quad_matrix, unsigned int matrix_length)
{
	float *quad_elements = get_root_element(quad_matrix);
	for (unsigned int i = 0; i < matrix_length; i++)
		for (unsigned int j = 0; j < matrix_length; j++)
			if (seq_matrix[i*matrix_length + j] != quad_elements[get_quad_index(i*matrix_length + j, matrix_length)]) {
				printf("Verification shows a mismatch at i%d j%d\n", i, j);
				return -1;
			}

	// Success
	printf("Output verified. Algorithm corect!\n");
	return 0;
}

int verify_output(float *seq_matrix, float *strassen_output, unsigned int matrix_length)
{
	int mismatch = 0;
	/* Verify the sequential vs. strassen algo output. */
	for (unsigned int i = 0; i<matrix_length * matrix_length; i++) {
		if (seq_matrix[i] != strassen_output[i]) {
			mismatch = 1;
			printf("Mismatch at index : %d %d %f %f\n",
				i / matrix_length, i%matrix_length, seq_matrix[i], strassen_output[i]);
			return -1;
		}
	}

	// Success
	printf("Output verified. Algorithm corect!\n");
	return 0;
}

#endif /* VERIFY_OUTPUT */