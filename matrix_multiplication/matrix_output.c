#include "matrix_mul.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef PRINT_OUTPUT
void print_matrix(float *input_matrix, unsigned int length)
{
	if (input_matrix == NULL || length == 0) {
		printf("%s : Invalid arguments\n", __func__);
		return;
	}

	for (unsigned int i = 0; i<length; i++) {
		for (unsigned int j = 0; j<length; j++)
			printf("%.1f\t", input_matrix[i*length + j]);
		printf("\n");
	}
}

void print_matrix_two_dim(float **input_matrix, unsigned int length)
{
	if (input_matrix == NULL || length == 0) {
		printf("%s : Invalid arguments\n", __func__);
		return;
	}

	for (unsigned int i = 0; i<length; i++) {
		for (unsigned int j = 0; j<length; j++)
			printf("%.1f\t", input_matrix[i][j]);
		printf("\n");
	}
}

void print_quad_layout_matrix(MatQuad *quad_matrix, unsigned int matrix_length)
{
	float *quad_elements = get_root_element(quad_matrix);
	for (unsigned int i = 0; i < matrix_length; i++) {
		for (unsigned int j = 0; j < matrix_length; j++)
			printf("%.1f\t", quad_elements[get_quad_index(i*matrix_length + j, matrix_length)]);
		printf("\n");
	}
}

static void print_quad(MatQuad *quad_matrix)
{
	if (quad_matrix != NULL) {
		printf("Quad head : %p elements %p\n", quad_matrix, quad_matrix->elements);
		printf("Quad childs : %p %p %p %p\n",
			quad_matrix->child_quad[0], quad_matrix->child_quad[1],
			quad_matrix->child_quad[2], quad_matrix->child_quad[3]);
		print_quad(quad_matrix->child_quad[0]);
		print_quad(quad_matrix->child_quad[1]);
		print_quad(quad_matrix->child_quad[2]);
		print_quad(quad_matrix->child_quad[3]);
	}
}
#endif /* PRINT_OUTPUT */

#ifdef VERIFY_OUTPUT
void write_quad_output_files(float *mat_c_seq, MatQuad *mat_c_quad,
	float *mat_a_seq, MatQuad *mat_a_quad, float *mat_b_seq,
	MatQuad *mat_b_quad, unsigned int length)
{
	FILE*seq_fp = fopen("c_seq.txt", "w+");
	FILE*str_fp = fopen("c_quad.txt", "w+");
	FILE*a_fp = fopen("a_seq.txt", "w+");
	FILE*b_fp = fopen("b_seq.txt", "w+");
	FILE*a_quad_fp = fopen("a_quad.txt", "w+");
	FILE*b_quad_fp = fopen("b_quad.txt", "w+");

	if (seq_fp == NULL ||
		str_fp == NULL ||
		a_fp == NULL ||
		b_fp == NULL ||
		a_quad_fp == NULL ||
		b_quad_fp == NULL) {
		if (!seq_fp) fclose(seq_fp);
		if (!str_fp) fclose(str_fp);
		if (!a_fp) fclose(a_fp);
		if (!b_fp) fclose(b_fp);
		printf("unable to open seq file for writing\n");
		return;
	}

	float *c_quad_elements = get_root_element(mat_c_quad);
	float *a_quad_elements = get_root_element(mat_a_quad);
	float *b_quad_elements = get_root_element(mat_b_quad);

	for (unsigned int i = 0; i < length * length; i++) {
		if (i % length == 0) {
			fprintf(seq_fp, "\n");
			fprintf(str_fp, "\n");
			fprintf(a_fp, "\n");
			fprintf(b_fp, "\n");
			fprintf(a_quad_fp, "\n");
			fprintf(b_quad_fp, "\n");
		}

		fprintf(seq_fp, "%.1f ", mat_c_seq[i]);
		fprintf(str_fp, "%.1f ", c_quad_elements[get_quad_index(i, length)]);
		fprintf(a_fp, "%.1f ", mat_a_seq[i]);
		fprintf(b_fp, "%.1f ", mat_b_seq[i]);
		fprintf(a_quad_fp, "%.1f ", a_quad_elements[get_quad_index(i, length)]);
		fprintf(b_quad_fp, "%.1f ", b_quad_elements[get_quad_index(i, length)]);
	}

	fclose(seq_fp);
	fclose(str_fp);
	fclose(a_fp);
	fclose(b_fp);
	fclose(a_quad_fp);
	fclose(b_quad_fp);
}

void write_output_files(float *seq_output, float *strassen_output, float *a, float *b, unsigned int length)
{
	FILE*seq_fp = fopen("seq_c.txt", "w+");
	FILE*str_fp = fopen("str_c.txt", "w+");
	FILE*a_fp = fopen("a.txt", "w+");
	FILE*b_fp = fopen("b.txt", "w+");
	if (seq_fp == NULL ||
		str_fp == NULL ||
		a_fp == NULL ||
		b_fp == NULL) {
		if (!seq_fp) fclose(seq_fp);
		if (!str_fp) fclose(str_fp);
		if (!a_fp) fclose(a_fp);
		if (!b_fp) fclose(b_fp);
		printf("unable to open seq file for writing\n");
		return;
	}

	for (unsigned int i = 0; i < length * length; i++) {
		if (i % length == 0) {
			fprintf(seq_fp, "\n");
			fprintf(str_fp, "\n");
			fprintf(a_fp, "\n");
			fprintf(b_fp, "\n");
		}

		fprintf(seq_fp, "%.1f ", seq_output[i]);
		fprintf(str_fp, "%.1f ", strassen_output[i]);
		fprintf(a_fp, "%.1f ", a[i]);
		fprintf(b_fp, "%.1f ", b[i]);
	}

	fclose(seq_fp);
	fclose(str_fp);
	fclose(a_fp);
	fclose(b_fp);
}
#endif /* VERIFY_OUTPUT */