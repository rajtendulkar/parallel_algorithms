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

	/* Wiki based implementation. */	
	for (unsigned int i = 16; i <= 8192; i <<= 1) {
		printf("Testing matrix length : %u\n", i);
		// TODO : Some code duplication can be avoided.
		// But this is convenient right now.

		// test_strassen_wiki(i);
		// strassen_with_single_dim_ptr(i);
		strassen_with_quad_layout(i);
		// test_sequential_single_dim_ptr(i);
		// test_two_dim_sequential(i);
	}
	
	printf("Finished program execution.\n");	
	return 0;
}