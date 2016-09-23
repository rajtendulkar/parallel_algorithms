#include "matrix_mul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int test_strassen_wiki(unsigned int matrix_length);
extern int strassen_with_single_dim_ptr(unsigned int matrix_length);
extern int strassen_with_quad_layout(unsigned int matrix_length);
extern int print_processor_info();

/*
 * Tested on Intel® Core™ i5-4300U Processor 
 *		# of Cores	2
 *		# of Threads	4
 *		Max # of Memory Channels	2
 *		Max Memory Bandwidth	25.6 GB/s
 *		Instruction Set	64-bit
 *		
 *		Level 1 Cache	: 128 KB
 *				2 x 32 KB 8-way set associative instruction caches
 *				2 x 32 KB 8-way set associative data caches
 *
 *		Level 2 Cache	: 512 KB
 *				2 x 256 KB 8-way set associative caches
 *
 *		Level 3 Cache	: 3072 KB
 *				3 MB 12-way set associative shared cache
 *		
 *		All caches with cacheline size 64-bytes
 *
 *	Used Intel C++ Compiler v17.0 on Windows 7 platform.
 *
 */
int main(int argc, char *argv[])
{
	// print_processor_info();

	printf("Testing matrix length : %u\n", MATRIX_LENGTH);
	/* Wiki based implementation. */
	// test_strassen_wiki(MATRIX_LENGTH);

	/* Single pointer instead of double pointer matrix. */
	// strassen_with_single_dim_ptr(MATRIX_LENGTH);

	strassen_with_quad_layout(MATRIX_LENGTH);
	
	printf("Finished program execution.\n");	
	return 0;
}