
#ifndef _MATRIX_MUL_H
#define _MATRIX_MUL_H

#include <stdio.h>


/* Row size for each element of quad matrix layout.
 */
#define MATQUAD_ROW_SIZE 16

/*
 * Quad matrix layout structure
 *
 * elements : pointer to matrix element block of size (MATQUAD_ROW_SIZE * MATQUAD_ROW_SIZE)
 *			  set to NULL when the element has children (or is a non-leaf node).
 * child_quad : child elements of quad matrix
 *				set to NULL when it is a leaf node.
 *
 * The layout should be as follows in the memory:
 *
 * ---------------------
 * | 0  | 1  | 4  | 5  |
 * ---------------------
 * | 2  | 3  | 6  | 7  |
 * ---------------------
 * | 8  | 9  | 12 | 13 |
 * ---------------------
 * | 10 | 11 | 14 | 15 |
 * ---------------------
 *
 * each element will be pointing to a block in the matrix.
 * next Quad super-block will be pointing to [0,1,2,3] as children.
 * further would point to [4,5,6,7] as children and then [8,9,10,11], [12,13,14,15] so on.
 *
 * higher level of super block or root node will point to all these four nodes mentioned
 * above as its children.
 *
 * The layout should look like below:
 *	Root ->
 *		child[0]->
 *				elements -> NULL
 *				child[0]-> 
 *						elements->mat_block[0]
 *						child[0,1,2,3] -> NULL
 *				child[1]->
 *						elements->mat_block[1]
 *						child[0,1,2,3] -> NULL
 *				child[2]->
 *						elements->mat_block[2]
 *						child[0,1,2,3] -> NULL
 *				child[3]->
 *						elements->mat_block[3]
 *						child[0,1,2,3] -> NULL
 *
 *		child[1]->
 *				elements -> NULL
 *				child[0]->
 *						elements->mat_block[4]
 *						child[0,1,2,3] -> NULL
 *				child[1]->
 *						elements->mat_block[5]
 *						child[0,1,2,3] -> NULL
 *				child[2]->
 *						elements->mat_block[6]
 *						child[0,1,2,3] -> NULL
 *				child[3]->
 *						elements->mat_block[7]
 *						child[0,1,2,3] -> NULL
 *		child[2]->
 *				elements -> NULL
 *				child[0]->
 *						elements->mat_block[8]
 *						child[0,1,2,3] -> NULL
 *				child[1]->
 *						elements->mat_block[9]
 *						child[0,1,2,3] -> NULL
 *				child[2]->
 *						elements->mat_block[10]
 *						child[0,1,2,3] -> NULL
 *				child[3]->
 *						elements->mat_block[11]
 *						child[0,1,2,3] -> NULL
 *
 *		child[3]->
 *				elements -> NULL
 *				child[0]->
 *						elements->mat_block[12]
 *						child[0,1,2,3] -> NULL
 *				child[1]->
 *						elements->mat_block[13]
 *						child[0,1,2,3] -> NULL
 *				child[2]->
 *						elements->mat_block[14]
 *						child[0,1,2,3] -> NULL
 *				child[3]->
 *						elements->mat_block[15]
 *						child[0,1,2,3] -> NULL
 *
 */
typedef struct matquad
{
	float *elements;
	struct matquad *child_quad[4];
} MatQuad;

/* Print all the matrices */
// #define PRINT_OUTPUT

/* 
 * Enable OpenMP Directives.
 *
 * After the analysis, the bottleneck of the code is 
 * sequential_mult_with_stride() function with more than 71% of time
 * spent in this code. It was necessary to optimize this function to 
 * get the speed-up.
 *
 * The OpenMP directives gives a chance to exploit the 
 * available data-parallelism in the algorithm. One thing to keep in
 * mind is the cache-thrashing as matrix size goes beyond the cache
 * size, due to parallel execution of threads. One way to possibly mitigate
 * the effects, I think,  would be to allocate a copy of array to each thread.
 * This has a chance of reducing the thrashing. Since this processor has only
 * two (or 4 logical) cores, it will be worthwhile to restrict the number of parallel
 * threads to only 4 to avoid the thrashing. This needs further investigation.
 */
// #define OPENMP
// #define OPENMP_SECTIONS
// #define OPENMP_FOR_LOOPS

/* Verify strassen algorithm output against sequential algorithm. */
#ifndef VERIFY_OUTPUT
// #define VERIFY_OUTPUT
#endif /* VERIFY_OUTPUT */

/*
 * Execute sequential ijk multiplication algorithm
 * for comparison. This will slow down the total program
 * execution.
 */
#ifndef EXECUTE_SEQUENTIAL_CODE
// #define EXECUTE_SEQUENTIAL_CODE
#endif /* EXECUTE_SEQUENTIAL_CODE */

/*
 * I am implementing the matrix multiplication for square matrices with
 * power of 2. Non-power of two can be padded with zeros, or need further 
 * code implementation / optimization.
 */
#ifndef MATRIX_LENGTH
#define MATRIX_LENGTH 2048
#endif /* MATRIX_LENGTH */

/* 
 * Number of iterations of matrix multiplication to 
 * measure the execution time of the strassen algorithm 
 * (improves measured time accuracy).
 */
#ifdef VERIFY_OUTPUT
#define NUM_ITERATIONS 1
#else
#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 5
#endif /* NUM_ITERATIONS */
#endif /* VERIFY_OUTPUT */

/* Leave a possibility to supply LEAF_SIZE using makefile.
 *
 * Higher the leaf-size, we have lower granularity of parallelism
 * and more n^3 multiplication to perform.
 * Lower leaf-size than cache-line, less utilization of cache locality.
 *
 * Since the cache-line size if of 64-bytes, it will be beneficial
 * to use entire cache-line in all add, subtract and multiply functions.
 * Thus leave leaf size to 16 for float data type where each float is 4-byte
 * total 16 * 4 = 64 bytes. Following results prove the theory
 *
 * For matrix size of 2048, following results were obtained :
 *		LEAF_SIZE	Execution time(seconds)
 *		   8			48.30
 *		   16			39.75
 *		   32			45.99
 */
#ifndef LEAF_SIZE
#define LEAF_SIZE 16
#endif /* LEAF_SIZE */

/*
 * SIMD instructions for vector operations.
 * Core i5 supports AVX instructions
 * Enables non-portable code.
 */
#ifndef INTEL_SIMD_INSTRUCTIONS
#define INTEL_SIMD_INSTRUCTIONS
#endif /* INTEL_SIMD_INSTRUCTIONS */

/*
 * Give an un-rolling hint to the Intel compiler.
 * This gives a slight improvement to improve the branching code.
 * However with a modern processor, the improvements are not significant
 * due to enhanced branch prediction hardware.
 * Enables non-portable code.
 */
#ifndef INTEL_COMPILER_PRAGMA_UNROLL
//#define INTEL_COMPILER_PRAGMA_UNROLL
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */

/*
 * Store matrix B in column major format to improve 
 * data cache locality.
 *
 * Quad-tree memory layout should give further speed-up.
 * Source : http://crpit.com/confpapers/CRPITV26ElGindy.pdf
 */
#ifndef MATRIX_B_COLUMN_MAJOR_FORMAT
// #define MATRIX_B_COLUMN_MAJOR_FORMAT
#endif /* MATRIX_B_COLUMN_MAJOR_FORMAT */

extern void sequential_mat_mul_single_dim(float *mat_a, float *mat_b, float *mat_c, unsigned int length);
extern void initialize_single_dim_matrix(float *input_matrix, unsigned int length, unsigned int col_major);
extern float *allocate_single_dim_matrix(int length);
extern float *free_single_dim_matrix(float *matrix);

extern void print_matrix(float *input_matrix, unsigned int length);
extern void write_output_files(float *seq_output, float *strassen_output, float *a, float *b, unsigned int length);
extern int verify_output(float *seq_output, float *strassen_output, unsigned int length);
extern void write_quad_output_files(float *mat_c_seq, MatQuad *mat_c_quad,
	float *mat_a_seq, MatQuad *mat_a_quad, float *mat_b_seq,
	MatQuad *mat_b_quad, unsigned int length);
extern unsigned int get_quad_index(unsigned int normal_index, unsigned int matrix_length);
extern float *get_root_element(MatQuad *input_matrix);
extern int verify_quad_output(float*seq_matrix, MatQuad *quad_matrix, unsigned int matrix_length);
extern void print_matrix_two_dim(float **input_matrix, unsigned int length);
extern void print_quad_layout_matrix(MatQuad *quad_matrix, unsigned int matrix_length);
extern int test_sequential_single_dim_ptr(unsigned int matrix_length);
extern int test_two_dim_sequential(unsigned int matrix_length);

#endif /* _MATRIX_MUL_H */