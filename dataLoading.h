#pragma once
#pragma warning(disable:4996)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
	double* training_input_data;
	double* training_output_data;
	double* testing_input_data;
	double* testing_output_data;
	int input_row_size;
	int output_row_size;
	int training_row_num;
	int testing_row_num;
} DataSet;

void freeDataSet(DataSet* data)
{
	free(data->training_input_data);
	free(data->training_output_data);
	free(data->testing_input_data);
	free(data->testing_output_data);
	free(data);
}

/*

*/
void dataSetNormalize(DataSet* data)
{
	int tot_row_size = data->input_row_size + data->output_row_size;
	int tot_row_num = data->training_row_num + data->testing_row_num;
	double* col_min_nums = (double*)malloc(sizeof(double) * tot_row_size);
	double* col_max_nums = (double*)malloc(sizeof(double) * tot_row_size);
	for (int i = 0; i < data->input_row_size; i++) {
		col_min_nums[i] = data->training_input_data[i];
		col_max_nums[i] = data->training_input_data[i];
	}
	for (int i = 0; i < data->output_row_size; i++) {
		col_min_nums[data->input_row_size + i] = data->training_output_data[i];
		col_max_nums[data->input_row_size + i] = data->training_output_data[i];
	}

	for (int i = 0; i < tot_row_num; i++) {
		for (int j = 0; j < data->input_row_size; j++) {
			double num = (i < data->training_row_num) ? 
				data->training_input_data[i * data->input_row_size + j] : 
				data->testing_input_data[(i - data->training_row_num) * data->input_row_size + j];
			col_min_nums[j] = (num < col_min_nums[j]) ? num : col_min_nums[j];
			col_max_nums[j] = (num > col_max_nums[j]) ? num : col_max_nums[j];
		}
		for (int j = 0; j < data->output_row_size; j++) {
			double num = (i < data->training_row_num) ?
				data->training_output_data[i * data->output_row_size + j] :
				data->testing_output_data[(i - data->training_row_num) * data->output_row_size + j];
			int ind = j + data->input_row_size;
			col_min_nums[ind] = (num < col_min_nums[ind]) ? num : col_min_nums[ind];
			col_max_nums[ind] = (num > col_max_nums[ind]) ? num : col_max_nums[ind];
		}
	}
	for (int i = 0; i < tot_row_size; i++) {
		col_max_nums[i] -= col_min_nums[i];
	}

	for (int i = 0; i < tot_row_num; i++) {
		for (int j = 0; j < data->input_row_size; j++) {
			if (i < data->training_row_num) {
				int offset = i * data->input_row_size + j;
				data->training_input_data[offset] = (data->training_input_data[offset] - col_min_nums[j]) / col_max_nums[j];
			}
			else {
				int offset = (i - data->training_row_num) * data->input_row_size + j;
				data->testing_input_data[offset] = (data->testing_input_data[offset] - col_min_nums[j]) / col_max_nums[j];
			}
		}
		for (int j = 0; j < data->output_row_size; j++) {
			int col_off = j + data->input_row_size;
			if (i < data->training_row_num) {
				int offset = i * data->output_row_size + j;
				data->training_output_data[offset] = (data->training_output_data[offset] - col_min_nums[col_off]) / col_max_nums[col_off];
			}
			else {
				int offset = (i - data->training_row_num) * data->output_row_size + j;
				data->testing_output_data[offset] = (data->testing_output_data[offset] - col_min_nums[col_off]) / col_max_nums[col_off];
			}
		}
	}
}

/*
reads a CSV file and returns a DataSet pointer.
output_indexes should be an array of the indexes of each row that are outputs.
all other indexes will be assumed to be inputs.
*/
DataSet* loadCSV(const char file_name[], int* output_indexes, int output_num, int* remove_indexes, int remove_num, double percent_for_test)
{
	FILE* file;
	errno_t err = fopen_s(&file, file_name, "r");
	if (err != 0) {
		perror("Failed to open the file");
		return NULL;
	}

	DataSet* new_ds = (DataSet*)malloc(sizeof(DataSet));

	#define MAXCHAR 1024
	char row[MAXCHAR];
	char* token;


	int row_size = 0;
	int row_num = 0;

	while (!feof(file)) {
		fgets(row, MAXCHAR, file);
		if (row_num == 0) {
			token = strtok(row, ",");
			while (token != NULL) {
				token = strtok(NULL, ",");
				row_size++;
			}

		}

		row_num++;
	}
	row_num -= 2;
	
	new_ds->input_row_size = row_size - output_num - remove_num;
	new_ds->output_row_size = output_num;
	new_ds->testing_row_num = (int)ceil(row_num * percent_for_test);
	new_ds->training_row_num = row_num - new_ds->testing_row_num;
	new_ds->training_input_data = (double*)malloc(sizeof(double) * new_ds->input_row_size * new_ds->training_row_num);
	new_ds->training_output_data = (double*)malloc(sizeof(double) * new_ds->output_row_size * new_ds->training_row_num);
	new_ds->testing_input_data = (double*)malloc(sizeof(double) * new_ds->input_row_size * new_ds->testing_row_num);
	new_ds->testing_output_data = (double*)malloc(sizeof(double) * new_ds->output_row_size * new_ds->testing_row_num);

	bool* bool_output_indexes = (bool*)calloc(sizeof(bool), row_size);
	bool* bool_removed_indexes = (bool*)calloc(sizeof(bool), row_size);
	for (int i = 0; i < output_num; i++) {	
		bool_output_indexes[output_indexes[i]] = true;
	}
	for (int i = 0; i < remove_num; i++) {
		bool_removed_indexes[remove_indexes[i]] = true;
	}
	
	fseek(file, 0, SEEK_SET);
	fgets(row, MAXCHAR, file);
	int train_input_index = 0;
	int train_output_index = 0;
	int test_input_index = 0;
	int test_output_index = 0;

	for (int row_index = 0; row_index < row_num; row_index++) {
		fgets(row, MAXCHAR, file);
		token = strtok(row, ",");
		for (int read_index = 0; read_index < row_size; read_index++) {
			if (bool_removed_indexes[read_index]) {
				token = strtok(NULL, ",");
				continue;
			}
			bool writing_training = row_index < new_ds->training_row_num;
			bool writing_input = !bool_output_indexes[read_index];

			double number = strtod(token, NULL);

			if (writing_training && writing_input) {
				new_ds->training_input_data[train_input_index] = number;
				train_input_index++;
			}
			else if (writing_training && !writing_input) {
				new_ds->training_output_data[train_output_index] = number;
				train_output_index++;
			}
			else if (!writing_training && writing_input) {
				new_ds->testing_input_data[test_input_index] = number;
				test_input_index++;
			}
			else {
				new_ds->testing_output_data[test_output_index] = number;
				test_output_index++;
			}

			token = strtok(NULL, ",");
		}
	}
	free(bool_output_indexes);
	free(bool_removed_indexes);

	fclose(file);

	return new_ds;
}