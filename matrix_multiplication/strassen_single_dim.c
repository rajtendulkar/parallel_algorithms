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

void sum_single_dim(float *mat_a, float *mat_b, float *mat_c,
	unsigned int matrix_length,
	unsigned int stride_a, unsigned int stride_b, unsigned int stride_c)
{

#ifdef INTEL_SIMD_INSTRUCTIONS
	unsigned int j;
	for (unsigned int i = 0; i < matrix_length; i++) {
		for (j = 0; j <= matrix_length - 8; j += 8) {
			*((__m256*)(mat_c + i * stride_c + j)) = _mm256_add_ps(
				*((__m256*)(mat_a + i * stride_a + j)),
				*((__m256*)(mat_b + i * stride_b + j)));
		}

		/* Any residual elements.
		* It is unlikely that we will execute this for-loop.
		* I will leave it here for sake of correctness.
		*/
		for (; j < matrix_length; j++)
			mat_c[i * stride_c + j] = mat_a[i * stride_a + j] + mat_b[i * stride_b + j];
	}

#else

#ifdef OPENMP_FOR_LOOPS
#pragma omp parallel for
#endif /* OPENMP_SECTIONS */

#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
	for (unsigned int i = 0; i < matrix_length; i++)

#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
		for (unsigned int j = 0; j < matrix_length; j++)

			mat_c[i * stride_c + j] = mat_a[i * stride_a + j] + mat_b[i * stride_b + j];
#endif
}

void subtract_single_dim(float *mat_a, float *mat_b, float *mat_c,
	unsigned int matrix_length,
	unsigned int stride_a, unsigned int stride_b, unsigned int stride_c)
{

#ifdef INTEL_SIMD_INSTRUCTIONS
	unsigned int j;
	for (unsigned int i = 0; i < matrix_length; i++) {
		for (j = 0; j <= matrix_length - 8; j += 8) {
			*((__m256*)(mat_c + i * stride_c + j)) = _mm256_sub_ps(
				*((__m256*)(mat_a + i * stride_a + j)),
				*((__m256*)(mat_b + i * stride_b + j)));
		}

		/* Any residual elements.
		* It is unlikely that we will execute this for-loop.
		* I will leave it here for sake of correctness.
		*/
		for (; j < matrix_length; j++)
			mat_c[i * stride_c + j] = mat_a[i * stride_a + j] - mat_b[i * stride_b + j];
	}

#else

#ifdef OPENMP_FOR_LOOPS
#pragma omp parallel for
#endif /* OPENMP_SECTIONS */

#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
	for (unsigned int i = 0; i < matrix_length; i++)

#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
		for (unsigned int j = 0; j < matrix_length; j++)
			mat_c[i * stride_c + j] = mat_a[i * stride_a + j] - mat_b[i * stride_b + j];

#endif /* INTEL_SIMD_INSTRUCTIONS */
}

#ifdef INTEL_SIMD_INSTRUCTIONS
typedef union {
	float flt[8];
	__m256 long_256_bit;
} long_element;
#endif /* INTEL_SIMD_INSTRUCTIONS */

static void sequential_mult_with_stride(float *mat_a, float *mat_b, float *mat_c,
	unsigned int matrix_length,
	unsigned int stride_a, unsigned int stride_b, unsigned int stride_c)
{
#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
	for (unsigned int i = 0; i < matrix_length; i++)

#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
		for (unsigned int j = 0; j < matrix_length; j++) {

			/* Caching the sum of the matrix to a local value. */
			float sum = 0.0;

#ifdef MATRIX_B_COLUMN_MAJOR_FORMAT
#ifdef INTEL_SIMD_INSTRUCTIONS

#if (LEAF_SIZE != 16)
#error "This code is optimized only for leaf-size 16. "
#endif /* (LEAF_SIZE != 16) */
			/*
			 * I am not sure if in-place vector operations will end-up
			 * in any pipeline hazards. Need to read arch. manual.
			 */
			long_element elem1, elem2, elem3;
			
			elem1.long_256_bit = _mm256_mul_ps(
				*((__m256*)(mat_a + i * stride_a)),
				*((__m256*)(mat_b + j * stride_b)));

			elem2.long_256_bit = _mm256_mul_ps(
				*((__m256*)(mat_a + i * stride_a + 8)),
				*((__m256*)(mat_b + j * stride_b + 8)));

			elem3.long_256_bit = elem2.long_256_bit = _mm256_add_ps(
				elem1.long_256_bit,
				elem2.long_256_bit);

			/* Ignore elem2 results. 
				* Add horizontally to save additional 4 + 2 sums.
				*/
			elem1.long_256_bit = _mm256_hadd_ps(elem3.long_256_bit, elem2.long_256_bit);
			elem3.long_256_bit = _mm256_hadd_ps(elem1.long_256_bit, elem2.long_256_bit);
				
			sum += elem3.flt[0];				
			sum += elem3.flt[4];			
#else

#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
			for (unsigned int k = 0; k < matrix_length; k++)
				/* Multiplication considering B in the col-major format */
				sum += mat_a[i * stride_a + k] * mat_b[j * stride_b + k];
#endif /* INTEL_SIMD_INSTRUCTIONS */
#else

#ifdef INTEL_COMPILER_PRAGMA_UNROLL
#pragma unroll (LEAF_SIZE)
#endif /* INTEL_COMPILER_PRAGMA_UNROLL */
			for (unsigned int k = 0; k < matrix_length; k++)
				/* Multiplication considering B in the row-major format */
				sum += mat_a[i * stride_a + k] * mat_b[k * stride_b + j];

#endif /* MATRIX_B_COLUMN_MAJOR_FORMAT */
			mat_c[i * stride_c + j] = sum;
		}
}

static void strassen(float *mat_a, float *mat_b, float *mat_c, unsigned int matrix_length,
	unsigned int stride_a, unsigned int stride_b, unsigned int stride_c)
{

	if (matrix_length <= LEAF_SIZE) {

		/**
		* If the matrix size is less than certain leaf size, then it is better
		* to perform a direct matrix multiplication rather than using strassen algorithm
		* to further sub-divide the matrix.
		*
		* Source : https://martin-thoma.com/strassen-algorithm-in-python-java-cpp/
		*
		*/
		sequential_mult_with_stride(mat_a, mat_b, mat_c,
			matrix_length, stride_a, stride_b, stride_c);
		return;
	}

	/**
	* source : https://en.wikipedia.org/wiki/Strassen_algorithm
	*
	* mat_a is split as :
	*
	*	|-----------|
	*	| A11 | A12 |
	*	|-----------|
	*   | A21 | A22 |
	*	|-----------|
	*
	*  and so are mat_b, mat_c.
	*
	*/

	/* divide the matrix into 2 x 2 sub-matrices. */

	/* That is why it is easier to have a matrix with power two,
	* so we don't have to deal with odd values of n. If we want to
	* deal with it, then one way is to pad the matrices with zero
	* and the upper left corner of the result will have a correct
	* value.
	*/
	unsigned int n = matrix_length / 2;

	/* The following expressions can be simplified.
	 * I keep them for better readability. The compiler
	 * optimizes them anyways.
	 */
	float *a11 = mat_a;
	float *a12 = mat_a + n;
	float *a21 = mat_a + n * stride_a;
	float *a22 = mat_a + n * stride_a + n;

	float *b11 = mat_b;
#ifdef MATRIX_B_COLUMN_MAJOR_FORMAT
	float *b21 = mat_b + n;
	float *b12 = mat_b + n * stride_b;
#else
	float *b12 = mat_b + n;
	float *b21 = mat_b + n * stride_b;
#endif /* MATRIX_B_COLUMN_MAJOR_FORMAT */
	float *b22 = mat_b + n * stride_b + n;

	float *c11 = mat_c;
	float *c12 = mat_c + n;
	float *c21 = mat_c + n * stride_c;
	float *c22 = mat_c + n * stride_c + n;

	float *m1, *m2, *m3, *m4, *m5, *m6, *m7;

	// We need 19 matrices of size n * n
	float *p = (float*)malloc(19 * n * n * sizeof(float));
	if (p == NULL) {
		printf("Insufficient memory for temporary matrix\n");
		return;
	}

#ifdef OPENMP_SECTIONS
#pragma omp parallel sections
#endif /* OPENMP_SECTIONS */
	{

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M1 = (A11 + A22) * (B11 + B22)
			m1 = &p[2 * n*n];
			sum_single_dim(a11, a22, &p[0], n, stride_a, stride_a, n);
			sum_single_dim(b11, b22, &p[n*n], n, stride_b, stride_b, n);
			strassen(&p[0], &p[n*n], m1, n, n, n, n);
		}

	// M1 is stored at p[2*n*n]
	// utilization p array is upto 3*n*n - 1

#ifdef OPENMP_SECTIONS
#pragma omp section
	#endif /* OPENMP_SECTIONS */
		{
			// M2 = (A21 + A22) * B11
			m2 = &p[4 * n*n];
			sum_single_dim(a21, a22, &p[3 * n*n], n, stride_a, stride_a, n);
			strassen(&p[3 * n*n], b11, m2, n, n, stride_b, n);
		}

	// M2 is stored at p[4*n*n]
	// utilization p array is upto 5*n*n - 1

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M3 = A11 * (B12 - B22)
			m3 = &p[6 * n*n];
			subtract_single_dim(b12, b22, &p[5 * n*n], n, stride_b, stride_b, n);
			strassen(a11, &p[5 * n*n], m3, n, stride_a, n, n);
		}

	// M3 is stored at p[6*n*n]
	// utilization p array is upto 6*n*n - 1

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M4 = A22 * (B21 - B11)
			m4 = &p[8 * n*n];
			subtract_single_dim(b21, b11, &p[7 * n*n], n, stride_b, stride_b, n);
			strassen(a22, &p[7 * n*n], m4, n, stride_a, n, n);
		}

	// M4 is stored at p[8*n*n]
	// utilization p array is upto 8*n*n - 1

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M5 = (A11 + A12) * B22
			m5 = &p[10 * n*n];
			sum_single_dim(a11, a12, &p[9 * n*n], n, stride_a, stride_a, n);
			strassen(&p[9 * n*n], b22, m5, n, n, stride_b, n);
		}

	// M5 is stored at p[10*n*n]
	// utilization p array is upto 10*n*n - 1

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M6 = (A21 - A11) * (B11 + B12)
			m6 = &p[13 * n*n];
			subtract_single_dim(a21, a11, &p[11 * n*n], n, stride_a, stride_a, n);
			sum_single_dim(b11, b12, &p[12 * n*n], n, stride_b, stride_b, n);
			strassen(&p[11 * n*n], &p[12 * n*n], m6, n, n, n, n);
		}

	// M6 is stored at p[13*n*n]
	// utilization p array is upto 13*n*n - 1

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// M7 = (A12 - A22) * (B21 + B22)
			m7 = &p[16 * n*n];
			subtract_single_dim(a12, a22, &p[14 * n*n], n, stride_a, stride_a, n);
			sum_single_dim(b21, b22, &p[15 * n*n], n, stride_b, stride_b, n);
			strassen(&p[14 * n*n], &p[15 * n*n], m7, n, n, n, n);
		}
	}

	/* There will be an implicit OpenMP barrier here to collect the results. */

#ifdef OPENMP_SECTIONS
#pragma omp parallel sections
#endif /* OPENMP_SECTIONS */
	{
#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// C11 = M1 + M4 - M5 + M7

			// Note: If we do in-place operations for C11, we can save n*n memory.
			//		 It is better for data locality.
			// 		 There is a un-harvested potential parallelism here, but I
			//		 think it will not outperform memory optimization.
			sum_single_dim (m1, m4, &p[17 * n*n], n, n, n, n);
			subtract_single_dim (m7, m5, c11, n, n, n, stride_c);
			sum_single_dim (&p[17 * n*n], c11, c11, n, n, stride_c, stride_c);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// C12 = M3 + M5
			sum_single_dim (m3, m5, c12, n, n, n, stride_c);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// C21 = M2 + M4
			sum_single_dim (m2, m4, c21, n, n, n, stride_c);
		}

#ifdef OPENMP_SECTIONS
#pragma omp section
#endif /* OPENMP_SECTIONS */
		{
			// C22 = M1 - M2 + M3 + M6
			// In-place operations to save memory. If we want to 
			// harvest parallelism, we will need additional (n * n) matrix.
			subtract_single_dim (m1, m2, &p[18 * n*n], n, n, n, n);
			sum_single_dim (m3, m6, c22, n, n, n, stride_c);
			sum_single_dim (&p[18 * n*n], c22, c22, n, n, stride_c, stride_c);
		}
	}

	/* There will be an implicit OpenMP barrier here to collect the results. */

	free(p);
}

int strassen_with_single_dim_ptr(unsigned int matrix_length)
{
	/*
	 * We decide to use floating point data type for number representation.
	 * The advantage is that for sum and multiplication we don't have to 
	 * upgrade to higher data type. If we use uint8, then the result of sum 
	 * and product must be stored in uint16. Thus if our input arrays are in 
	 * integer variables, then we need to upgrade to higher datatype first which
	 * is an additional overhead. It can be argued that floating point operations
	 * or pipelines are more expensive than arithmetic. However in algorithms like
	 * matrix multiplication, generally memory is a bottleneck rather than the CPU.
	 * It is possible to use double instead of float, but will need further code
	 * enhancement where we have introduced non-portable code.
	 */
	float *mat_a, *mat_b, *mat_c, *st_mat_c;
	clock_t t1, t2, t3;
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

	st_mat_c = allocate_single_dim_matrix(matrix_length);
	if (st_mat_c == NULL) {
		printf("Error allocating memory for matrix C\n");
		return_value = -1;
		goto st_mat_c_alloc_fail;
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

#if (defined(VERIFY_OUTPUT) || defined(EXECUTE_SEQUENTIAL_CODE))
	sequential_mat_mul_single_dim(mat_a, mat_b, mat_c, matrix_length);
#endif /* VERIFY_OUTPUT */

	t2 = clock();

	for (int i = 0; i<NUM_ITERATIONS; i++)
	{
		strassen(mat_a, mat_b, st_mat_c, matrix_length,
			matrix_length, matrix_length, matrix_length);
		sum_single_dim (st_mat_c, mat_a, st_mat_c, matrix_length,
			matrix_length, matrix_length, matrix_length);
	}

	t3 = clock();

	printf("Sequential : %f (seconds) Strassen : %f (seconds)\n",
		(double)(t2 - t1) / CLOCKS_PER_SEC,
		(double)(t3 - t2) / (CLOCKS_PER_SEC * NUM_ITERATIONS));

#ifdef PRINT_OUTPUT
	printf("\n*********************\n");
	printf("Seq Matrix C : \n");
	print_matrix(mat_c, matrix_length);
	printf("\n*********************\n");
#endif /* PRINT_OUTPUT */

#ifdef PRINT_OUTPUT
	printf("Strassen Matrix C : \n");
	print_matrix(st_mat_c, matrix_length);
	printf("\n*********************\n");
#endif /* PRINT_OUTPUT */

	/* Verify the output. If incorrect, write the 
	 * output matrices for analysis.
	 */
#ifdef VERIFY_OUTPUT
	if (verify_output(mat_c, st_mat_c, matrix_length))
		write_output_files(mat_c, st_mat_c, mat_a, mat_b, matrix_length);
#endif

	st_mat_c = free_single_dim_matrix(st_mat_c);

st_mat_c_alloc_fail:
	mat_c = free_single_dim_matrix(mat_c);

mat_c_alloc_fail:
	mat_b = free_single_dim_matrix(mat_b);

mat_b_alloc_fail:
	mat_a = free_single_dim_matrix(mat_a);

mat_a_alloc_fail:
	return return_value;
}

/*------------------------------------------------------------------------------*/