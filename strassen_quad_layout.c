#include "matrix_mul.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#ifdef OPENMP
#include <omp.h>
#endif /* OPENMP */

#ifdef INTEL_SIMD_INSTRUCTIONS
#include <immintrin.h>
#endif

static MatQuad *allocate_quad_matrix(unsigned int matrix_length)
{
	unsigned int num_quad_elements = 0; 	

	/* Check if matrix length is power of two. */
	if (matrix_length & (matrix_length - 1)) {
		printf("We are supporting only powers of two\n");
		return NULL;
	}

	/* Number of fine grain blocks */
	num_quad_elements = (matrix_length * matrix_length) / (MATQUAD_ROW_SIZE * MATQUAD_ROW_SIZE);
	if (num_quad_elements != 1) {
		unsigned int temp = num_quad_elements;
		while (temp != 0) {
			/* We require one super-block per 4 blocks. */
			temp /= 4;
			num_quad_elements += temp;
		}
	}	

	/* First allocate quad-elements. */
	MatQuad *quad_elements = malloc(num_quad_elements * sizeof(MatQuad));
	if (quad_elements == NULL) {
		printf("Out of memory : unable to allocate the matrix.");
		return NULL;
	}

	/* Allocate the matrix memory. */
	float *element_memory = malloc(matrix_length * matrix_length * sizeof(float));
	if (element_memory == NULL) {
		printf("out of memory for matrix elements\n");
		free(quad_elements);
		return NULL;
	}

	/* printf("Length : %d number quad blocks : %d quad:%p elem:%p\n",
		matrix_length, num_quad_elements, quad_elements, element_memory); */	

	/* First allocate all the basic blocks. */
	unsigned int basic_blocks = (matrix_length * matrix_length) / (MATQUAD_ROW_SIZE * MATQUAD_ROW_SIZE);
	for (int i = 0; i < basic_blocks; i++) {
		quad_elements[i].elements =
			&element_memory[i * MATQUAD_ROW_SIZE * MATQUAD_ROW_SIZE];
		
		/* Make the children NULL. */
		quad_elements[i].child_quad[0] = NULL;
		quad_elements[i].child_quad[1] = NULL;
		quad_elements[i].child_quad[2] = NULL;
		quad_elements[i].child_quad[3] = NULL;
	}

	/* Now we allocate all the super-blocks. */
	unsigned int remaining_blocks = num_quad_elements - basic_blocks;

	MatQuad *quad_ptr = &quad_elements[basic_blocks];
	unsigned int i = 0;
	while (remaining_blocks != 0) {
		quad_ptr->child_quad[0] = &quad_elements[i++];
		quad_ptr->child_quad[1] = &quad_elements[i++];
		quad_ptr->child_quad[2] = &quad_elements[i++];
		quad_ptr->child_quad[3] = &quad_elements[i++];
		
		quad_ptr->elements = NULL;		
		remaining_blocks--;
		quad_ptr++;
	}

	/* Last element will always be the root. */
	return &quad_elements[num_quad_elements - 1];
}

static MatQuad * free_quad_matrix(MatQuad *root)
{
	MatQuad *temp = root;
	
	/* Goto A0 element of the matrix. */
	while (temp->elements == NULL)
		temp = temp->child_quad[0];

	// printf("Free : quad:%p elem:%p\n", temp, temp->elements);

	free(temp->elements);
	free(temp);
	return NULL;
}

static inline uint32_t expand(uint16_t t)
{
	uint32_t u, v, w;
	u = ((t & 0x0000FF00) << 8) | (t & 0x000000FF);
	v = ((u & 0x00F000F0) << 4) | (u & 0x000F000F);
	w = ((v & 0x0C0C0C0C) << 2) | (v & 0x03030303);
	return(((w & 0x22222222) << 1) | (w & 0x11111111));
}

static inline uint16_t extract(uint32_t t)
{
	int u, v, w, x;
	u = (t & 0x55555555);
	v = ((u & 0x44444444) >> 1) | (u & 0x11111111);
	w = ((v & 0x30303030) >> 2) | (v & 0x03030303);
	x = ((w & 0x0F000F00) >> 4) | (w & 0x000F000F);
	return(((x & 0x00FF0000) >> 8) | (x & 0x000000FF));
}

static uint32_t calc_normal_order(uint32_t element, unsigned int row_length)
{
	return (extract(((element)) >> 1)) * row_length + (extract(((element))));
}

static uint32_t calc_z_order(uint16_t xPos, uint16_t yPos)
{
	return (expand(yPos) | (expand(xPos) << 1));
}

/* Converts a normal matrix index to quad layout index. */
unsigned int get_quad_index(unsigned int normal_index, unsigned int matrix_length)
{
	unsigned int normal_row = normal_index / matrix_length;
	unsigned int normal_col = normal_index % matrix_length;

	unsigned int normal_blk_i = normal_row / MATQUAD_ROW_SIZE;
	unsigned int normal_blk_j = normal_col / MATQUAD_ROW_SIZE;

	// Get the block number of the matrix
	unsigned int z_block_index = calc_z_order(normal_blk_i, normal_blk_j);

	return (z_block_index * MATQUAD_ROW_SIZE * MATQUAD_ROW_SIZE) + ((normal_row % MATQUAD_ROW_SIZE)*MATQUAD_ROW_SIZE) + (normal_col % MATQUAD_ROW_SIZE);
}

float *get_root_element(MatQuad *input_matrix)
{
	MatQuad *element = input_matrix;

	/* Goto A0 element of the matrix. */
	while (element->elements == NULL)
		element = element->child_quad[0];

	float *element_memory = element->elements;
	if (element_memory == NULL) {
		printf("Fatal error: we have came across a wrong pointer?\n");		
	}

	return element_memory;
}

/*
 * Pass a initialized_matrix to copy the data in the row-major format.
 */
static void initialize_quad_matrix(MatQuad *input_matrix, unsigned int matrix_length, float *initialized_matrix)
{
	static unsigned char initializer = 0;
	unsigned int basic_blocks = (matrix_length * matrix_length) / (MATQUAD_ROW_SIZE * MATQUAD_ROW_SIZE);

	float *element_memory = get_root_element(input_matrix);
	if (element_memory == NULL) {
		printf("Fatal error: NULL pointer?\n");
		return;
	}

	if (initialized_matrix == NULL) {

		/* Initialize with some random data. */	

		/* Remember this is not equal to [i][j] = initializer, since we have a different
		 * matrix layout.
		 */
		for (unsigned int i=0; i < matrix_length * matrix_length; i++, initializer %=10)
			element_memory[i] = initializer++;

	} else {
		/* Copy the data from the initialized matrix into quad layout. */
		for (unsigned int i = 0; i < matrix_length * matrix_length; i++)
			element_memory[get_quad_index(i, matrix_length)] = initialized_matrix[i];		
	}		
}

static void subtract(float *mat_a, float *mat_b, float *mat_c, unsigned int matrix_length)
{
#ifdef INTEL_SIMD_INSTRUCTIONS
	/* Assuming that the matrix_length is always in multiples of 8.*/
	unsigned int j;
#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
	for (unsigned int i = 0; i < matrix_length; i++) {
		for (j = 0; j < matrix_length; j += 8) {
			*((__m256*)(mat_c + i * matrix_length + j)) = _mm256_sub_ps(
				*((__m256*)(mat_a + i * matrix_length + j)),
				*((__m256*)(mat_b + i * matrix_length + j)));
		}		
	}
#else
#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
	for (unsigned int i = 0; i < matrix_length*matrix_length; i++)
			mat_c[i] = mat_a[i] - mat_b[i];
#endif /* INTEL_SIMD_INSTRUCTIONS */
}

static void sum(float *mat_a, float *mat_b, float *mat_c, unsigned int matrix_length)
{	
#ifdef INTEL_SIMD_INSTRUCTIONS
	/* Assuming that the matrix_length is always in multiples of 8.*/
	unsigned int j;
#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
	for (unsigned int i = 0; i < matrix_length; i++) {
		for (j = 0; j < matrix_length; j += 8) {
			*((__m256*)(mat_c + i * matrix_length + j)) = _mm256_add_ps(
				*((__m256*)(mat_a + i * matrix_length + j)),
				*((__m256*)(mat_b + i * matrix_length + j)));
		}		
	}
#else
#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
	for (unsigned int i = 0; i < matrix_length*matrix_length; i++)
		mat_c[i] = mat_a[i] + mat_b[i];
#endif /* INTEL_SIMD_INSTRUCTIONS */
}

static void sum_quad(MatQuad *mat_a, MatQuad *mat_b, MatQuad *mat_c)
{
	if (mat_a->elements == NULL) {
		/* Not using for-loop to avoid branching instructions. */
		sum_quad(mat_a->child_quad[0], mat_b->child_quad[0], mat_c->child_quad[0]);
		sum_quad(mat_a->child_quad[1], mat_b->child_quad[1], mat_c->child_quad[1]);
		sum_quad(mat_a->child_quad[2], mat_b->child_quad[2], mat_c->child_quad[2]);
		sum_quad(mat_a->child_quad[3], mat_b->child_quad[3], mat_c->child_quad[3]);
	} else {
		sum(mat_a->elements, mat_b->elements, mat_c->elements, MATQUAD_ROW_SIZE);
	}
}

static void subtract_quad(MatQuad *mat_a, MatQuad *mat_b, MatQuad *mat_c)
{
	if (mat_a->elements == NULL) {
		/* Not using for-loop to avoid branching instructions. */
		subtract_quad(mat_a->child_quad[0], mat_b->child_quad[0], mat_c->child_quad[0]);
		subtract_quad(mat_a->child_quad[1], mat_b->child_quad[1], mat_c->child_quad[1]);
		subtract_quad(mat_a->child_quad[2], mat_b->child_quad[2], mat_c->child_quad[2]);
		subtract_quad(mat_a->child_quad[3], mat_b->child_quad[3], mat_c->child_quad[3]);
	}
	else {
		subtract(mat_a->elements, mat_b->elements, mat_c->elements, MATQUAD_ROW_SIZE);
	}
}

static void sequential_mult(float *mat_a, float *mat_b, float *mat_c, unsigned int matrix_length)
{
	for (unsigned int i = 0; i < matrix_length; i++)
		for (unsigned int j = 0; j < matrix_length; j++) {
			/* Caching the sum of the matrix to a local value. */
			float sum = 0.0;
			for (unsigned int k = 0; k < matrix_length; k++)
				/* Multiplication considering B in the row-major format */
				sum += mat_a[i * matrix_length + k] * mat_b[k * matrix_length + j];

			mat_c[i * matrix_length + j] = sum;
		}
}

void strassen_quad(MatQuad *mat_a, MatQuad *mat_b, MatQuad *mat_c, unsigned int matrix_length)
{
	if (matrix_length <= MATQUAD_ROW_SIZE) {
		sequential_mult(mat_a->elements, mat_b->elements, mat_c->elements, matrix_length);
		return;
	}

	/* mat_a is split as :
	*
	*	-------------
	*	| A11 | A12 |
	*	|-----|-----|
	*   | A21 | A22 |
	*	-------------
	*/
	unsigned int n = matrix_length / 2;

	MatQuad *m1 = allocate_quad_matrix(n);
	MatQuad *m2 = allocate_quad_matrix(n);
	MatQuad *m3 = allocate_quad_matrix(n);
	MatQuad *m4 = allocate_quad_matrix(n);
	MatQuad *m5 = allocate_quad_matrix(n);
	MatQuad *m6 = allocate_quad_matrix(n);
	MatQuad *m7 = allocate_quad_matrix(n);

	MatQuad *temp1 = allocate_quad_matrix(n);
	MatQuad *temp2 = allocate_quad_matrix(n);
	MatQuad *temp3 = allocate_quad_matrix(n);
	MatQuad *temp4 = allocate_quad_matrix(n);
	MatQuad *temp5 = allocate_quad_matrix(n);
	MatQuad *temp6 = allocate_quad_matrix(n);
	MatQuad *temp7 = allocate_quad_matrix(n);
	MatQuad *temp8 = allocate_quad_matrix(n);
	MatQuad *temp9 = allocate_quad_matrix(n);
	MatQuad *temp10 = allocate_quad_matrix(n);
	MatQuad *temp11 = allocate_quad_matrix(n);
	MatQuad *temp12 = allocate_quad_matrix(n);
	MatQuad *temp13 = allocate_quad_matrix(n);

#ifdef OPENMP_SECTIONS
#pragma omp parallel sections
#endif /* OPENMP_SECTIONS */
	{

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M1 = (A11 + A22) * (B11 + B22)	
			sum_quad(mat_a->child_quad[0], mat_a->child_quad[3], temp1);
			sum_quad(mat_b->child_quad[0], mat_b->child_quad[3], temp2);
			strassen_quad(temp1, temp2, m1, n);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M2 = (A21 + A22) * B11
			sum_quad(mat_a->child_quad[2], mat_a->child_quad[3], temp3);
			strassen_quad(temp3, mat_b->child_quad[0], m2, n);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M3 = A11 * (B12 - B22)
			subtract_quad(mat_b->child_quad[1], mat_b->child_quad[3], temp4);
			strassen_quad(mat_a->child_quad[0], temp4, m3, n);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP */
		{
			// M4 = A22 * (B21 - B11)
			subtract_quad(mat_b->child_quad[2], mat_b->child_quad[0], temp5);
			strassen_quad(mat_a->child_quad[3], temp5, m4, n);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M5 = (A11 + A12) * B22
			sum_quad(mat_a->child_quad[0], mat_a->child_quad[1], temp6);
			strassen_quad(temp6, mat_b->child_quad[3], m5, n);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M6 = (A21 - A11) * (B11 + B12)
			subtract_quad(mat_a->child_quad[2], mat_a->child_quad[0], temp7);
			sum_quad(mat_b->child_quad[0], mat_b->child_quad[1], temp8);
			strassen_quad(temp7, temp8, m6, n);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M7 = (A12 - A22) * (B21 + B22)
			subtract_quad(mat_a->child_quad[1], mat_a->child_quad[3], temp9);
			sum_quad(mat_b->child_quad[2], mat_b->child_quad[3], temp10);
			strassen_quad(temp9, temp10, m7, n);
		}
	}

#ifdef OPENMP_SECTIONS
#pragma omp parallel sections
#endif /* OPENMP_SECTIONS */
	{
#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// C11 = M1 + M4 - M5 + M7
			sum_quad(m1, m4, temp11);
			subtract_quad(m7, m5, temp12);
			sum_quad(temp11, temp12, mat_c->child_quad[0]);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{

			// C12 = M3 + M5
			sum_quad(m3, m5, mat_c->child_quad[1]);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// C21 = M2 + M4
			sum_quad(m2, m4, mat_c->child_quad[2]);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// C22 = M1 - M2 + M3 + M6
			subtract_quad(m1, m2, temp13);
			sum_quad(temp13, m3, mat_c->child_quad[3]);
			sum_quad(mat_c->child_quad[3], m6, mat_c->child_quad[3]);
		}
	}

	/* Free temporary matrices */
	temp1 = free_quad_matrix(temp1);
	temp2 = free_quad_matrix(temp2);
	temp3 = free_quad_matrix(temp3);
	temp4 = free_quad_matrix(temp4);
	temp5 = free_quad_matrix(temp5);
	temp6 = free_quad_matrix(temp6);
	temp7 = free_quad_matrix(temp7);
	temp8 = free_quad_matrix(temp8);
	temp9 = free_quad_matrix(temp9);
	temp10 = free_quad_matrix(temp10);
	temp11 = free_quad_matrix(temp11);
	temp12 = free_quad_matrix(temp12);
	temp13 = free_quad_matrix(temp13);	

	/* Free M-matrices. */
	m1 = free_quad_matrix(m1);
	m2 = free_quad_matrix(m2);
	m3 = free_quad_matrix(m3);
	m4 = free_quad_matrix(m4);
	m5 = free_quad_matrix(m5);
	m6 = free_quad_matrix(m6);
	m7 = free_quad_matrix(m7);
}
/*
 * Matrix multiplication using quad layout of matrix.
 */
int strassen_with_quad_layout(unsigned int matrix_length)
{	
	MatQuad *mat_a_quad, *mat_b_quad, *mat_c_quad;
	float *mat_a_seq = NULL, *mat_b_seq = NULL, *mat_c_seq = NULL;
	clock_t t1, t2, t3;
	int return_value = 0;

#if (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE))
	/* Allocate matrix A. */
	mat_a_seq = allocate_single_dim_matrix(matrix_length);
	if (mat_a_seq == NULL) {
		printf("Error allocating memory for matrix A\n");
		return_value = -1;
		goto mat_a_seq_alloc_fail;
	}

	/* Allocate matrix B. */
	mat_b_seq = allocate_single_dim_matrix(matrix_length);
	if (mat_b_seq == NULL) {
		printf("Error allocating memory for matrix B\n");
		return_value = -1;
		goto mat_b_seq_alloc_fail;
	}

	/* Allocate matrix C. */
	mat_c_seq = allocate_single_dim_matrix(matrix_length);
	if (mat_c_seq == NULL) {
		printf("Error allocating memory for matrix C\n");
		return_value = -1;
		goto mat_c_seq_alloc_fail;
	}
#endif

	/* Allocate matrix A. */
	mat_a_quad = allocate_quad_matrix(matrix_length);
	if (mat_a_quad == NULL) {
		printf("Error allocating memory for matrix A\n");
		return_value = -1;
		goto mat_a_quad_alloc_fail;
	}

	/* Allocate matrix B. */
	mat_b_quad = allocate_quad_matrix(matrix_length);
	if (mat_b_quad == NULL) {
		printf("Error allocating memory for matrix B\n");
		return_value = -1;
		goto mat_b_quad_alloc_fail;
	}

	/* Allocate matrix C. */
	mat_c_quad = allocate_quad_matrix(matrix_length);
	if (mat_c_quad == NULL) {
		printf("Error allocating memory for matrix C\n");
		return_value = -1;
		goto mat_c_quad_alloc_fail;
	}

#if (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE))
	/* Initialize the matrices */
	initialize_single_dim_matrix(mat_a_seq, matrix_length, 0);
	initialize_single_dim_matrix(mat_b_seq, matrix_length, 0);	
#endif /* (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE)) */	

	/* Initialize A matrix in row-major format. */
	initialize_quad_matrix(mat_a_quad, matrix_length, mat_a_seq);
	
	/* Initialize the B matrix in col-major format. */
	initialize_quad_matrix(mat_b_quad, matrix_length, mat_b_seq);

#ifdef PRINT_OUTPUT
	/* Print the Quad pointers for debugging. 
	 */
	/*printf("******** MAT A ***********\n");
	print_quad(mat_a_quad);
	printf("**************************\n");
	printf("******** MAT C ***********\n");
	print_quad(mat_c_quad);
	printf("**************************\n");*/

	printf("\n*********************\n");
	printf("Matrix A : \n");
	print_matrix(mat_a_seq, matrix_length);

	printf("\n*********************\n");
	printf("Matrix A Quad : \n");
	print_quad_layout_matrix(mat_a_quad, matrix_length);

	printf("\n*********************\n");
	printf("Matrix B : \n");
	print_matrix(mat_b_seq, matrix_length);

	printf("\n*********************\n");
	printf("Matrix B Quad : \n");
	print_quad_layout_matrix(mat_b_quad, matrix_length);

#endif /* PRINT_OUTPUT */

	t1 = clock();

	/* Perform the sequential multiplication. */
#if (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE))
	sequential_mat_mul_single_dim (mat_a_seq, mat_b_seq, mat_c_seq, matrix_length);
#endif /* VERIFY_OUTPUT */

	t2 = clock();

	for (int i = 0; i<NUM_ITERATIONS; i++)
	{
		strassen_quad(mat_a_quad, mat_b_quad, mat_c_quad, matrix_length);
		sum_quad(mat_c_quad, mat_a_quad, mat_c_quad);
	}

	t3 = clock();

	printf("Sequential : %f (seconds) Strassen : %f (seconds)\n",
		(double)(t2 - t1) / CLOCKS_PER_SEC,
		(double)(t3 - t2) / (CLOCKS_PER_SEC * NUM_ITERATIONS));

#if defined(PRINT_OUTPUT) && (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE))
	printf("\n*********************\n");
	printf("Seq Matrix C : \n");
	print_matrix(mat_c_seq, matrix_length);
	printf("\n*********************\n");
#endif /* defined(PRINT_OUTPUT) && (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE)) */

#ifdef PRINT_OUTPUT
	printf("Strassen Quad Matrix C : \n");
	print_quad_layout_matrix(mat_c_quad, matrix_length);
	printf("\n*********************\n");
#endif /* PRINT_OUTPUT */

	/* Verify the output. If incorrect, write the
	 * output matrices for analysis.
	 */
#ifdef VERIFY_OUTPUT
	if (verify_quad_output(mat_c_seq, mat_c_quad, matrix_length))
		write_quad_output_files(mat_c_seq, mat_c_quad, 
								mat_a_seq, mat_a_quad,
								mat_b_seq, mat_b_quad, matrix_length);	
#endif

	/* Cleanup the allocated memory. */
	mat_c_quad = free_quad_matrix(mat_c_quad);
	
mat_c_quad_alloc_fail:
	mat_b_quad = free_quad_matrix(mat_b_quad);

mat_b_quad_alloc_fail:
	mat_a_quad = free_quad_matrix(mat_a_quad);

mat_a_quad_alloc_fail:
#if (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE))
	mat_c_seq = free_single_dim_matrix(mat_c_seq);

mat_c_seq_alloc_fail:
	mat_b_seq = free_single_dim_matrix(mat_b_seq);

mat_b_seq_alloc_fail:
	mat_a_seq = free_single_dim_matrix(mat_a_seq);

mat_a_seq_alloc_fail:
#endif /* #if (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE)) */	
	return return_value;
}

/*------------------------------------------------------------------------------*/