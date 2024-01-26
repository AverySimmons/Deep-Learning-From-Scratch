#pragma once

#include "matrixOperations.h"
#include "dataLoading.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

/*
a single fully connected neuron layer.
weights are stored iterating over prev_layer per cur_layer.
*/
typedef struct
{
	double* weights;
	double* biases;
} NeuronLayer;

/*
a structure in charge of holding values needed for backpropogation, can be freed after training is completed.
subset_size is the number of input ROWS in a single subset.
*/
typedef struct
{
	int subset_size;
	double** values;
	double** costs;
	double** dBiases;
	double** dWeights;
} NeuralNetworkTrainer;

/*
a neural network comprised of a series of fully connected neuron layers.
*/
typedef struct
{
	int input_num;
	NeuronLayer** layers;
	int layer_num;
	int* layer_sizes;
	NeuralNetworkTrainer* trainer;
}NeuralNetwork;

/*
returns a zeroed NeuronLayer.
*/
NeuronLayer* createNeuronLayer(int prev_layer_size, int layer_size) 
{
	NeuronLayer* new_layer = (NeuronLayer*)malloc(sizeof(NeuronLayer));
	new_layer->biases = (double*)calloc(layer_size, sizeof(double));
	new_layer->weights = (double*)calloc(prev_layer_size * layer_size, sizeof(double));
	return new_layer;
}

/*
frees a NeuronLayer.
*/
void freeNeuronLayer(NeuronLayer* layer) 
{
	free(layer->weights);
	free(layer->biases);
}

/*
sets the weights of a layer based on the He Distribution.
*/
void neuronLayerHeDistribution(NeuronLayer* layer, int layer_size, int prev_layer_size)
{
	double stddev = sqrt(2.0f / prev_layer_size);

	for (int i = 0; i < layer_size * prev_layer_size; i++) {
		double random_val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
		layer->weights[i] = random_val * stddev;
	}
}

/*
sets a NeuronLayer's biases to a constant.
*/
void neuronLayerSetBias(NeuronLayer* layer, double bias, int layer_size)
{
	for (int i = 0; i < layer_size; i++) {
		layer->biases[i] = bias;
	}
}

/*
manually sets a neuron layer by copying arrays.
*/
void neuronLayerSetValues(NeuronLayer* layer, double* weights, int weight_num, double* biases, int bias_num)
{
	for (int i = 0; i < weight_num; i++) {
		layer->weights[i] = weights[i];
	}
	for (int i = 0; i < bias_num; i++) {
		layer->biases[i] = biases[i];
	}
}

/*
attaches an allocated NeuralNetworkTrainer to a NeuralNetwork.
*/
void attachNeuralNetworkTrainer(NeuralNetwork* nn, int subset_size)
{
	NeuralNetworkTrainer* new_trainer = (NeuralNetworkTrainer*)malloc(sizeof(NeuralNetworkTrainer));
	new_trainer->subset_size = subset_size;
	new_trainer->values = (double**)malloc(sizeof(double*) * nn->layer_num);
	new_trainer->costs = (double**)malloc(sizeof(double*) * nn->layer_num);
	new_trainer->dBiases = (double**)malloc(sizeof(double*) * nn->layer_num);
	new_trainer->dWeights = (double**)malloc(sizeof(double*) * nn->layer_num);
	for (int i = 0; i < nn->layer_num; i++) {
		int prev_layer_size = (i == 0) ? nn->input_num : nn->layer_sizes[i - 1];
		new_trainer->values[i] = (double*)malloc(sizeof(double) * nn->layer_sizes[i] * subset_size);
		new_trainer->costs[i] = (double*)malloc(sizeof(double) * nn->layer_sizes[i] * subset_size);
		new_trainer->dBiases[i] = (double*)malloc(sizeof(double) * nn->layer_sizes[i]);
		new_trainer->dWeights[i] = (double*)malloc(sizeof(double) * nn->layer_sizes[i] * prev_layer_size);
	}
	nn->trainer = new_trainer;
}

/*
removes the NeuralNetworkTrainer from a NeuralNetwork.
*/
void removeNeuralNetworkTrainer(NeuralNetwork* nn)
{
	for (int i = 0; i < nn->layer_num; i++) {
		free(nn->trainer->values[i]);
		free(nn->trainer->costs[i]);
		free(nn->trainer->dBiases[i]);
		free(nn->trainer->dWeights[i]);
	}
	free(nn->trainer->values);
	free(nn->trainer->costs);
	free(nn->trainer->dBiases);
	free(nn->trainer->dWeights);
	free(nn->trainer);
	nn->trainer = NULL;
}

/*
returns an allocated and randomized NeuralNetwork.
*/
NeuralNetwork* createNeuralNetwork(int input_num, int layer_num, int* layer_sizes)
{
	NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	nn->input_num = input_num;
	nn->layer_num = layer_num;
	nn->layer_sizes = (int*)malloc(sizeof(int) * layer_num);
	nn->layers = (NeuronLayer**)malloc(sizeof(NeuronLayer*) * layer_num);
	nn->trainer = NULL;
	for (int i = 0; i < layer_num; i++) {
		nn->layer_sizes[i] = layer_sizes[i];
	}

	for (int i = 0; i < layer_num; i++) {
		int prev_layer_size = (i == 0) ? input_num : layer_sizes[i - 1];
		NeuronLayer* new_layer = createNeuronLayer(prev_layer_size, layer_sizes[i]);
		neuronLayerHeDistribution(new_layer, layer_sizes[i], prev_layer_size);
		neuronLayerSetBias(new_layer, 0.1f, layer_sizes[i]);
		nn->layers[i] = new_layer;
	}

	return nn;
}

/*
frees a NeuralNetwork.
*/
void freeNeuralNetwork(NeuralNetwork* nn)
{
	free(nn->layer_sizes);
	for (int i = 0; i < nn->layer_num; i++) {
		freeNeuronLayer(nn->layers[i]);
	}
	if (nn->trainer != NULL) {
		removeNeuralNetworkTrainer(nn);
	}
	free(nn);
}

/*
intakes an input matrix and a layer and writes the resulting output to a pointer.
*/
void neuronLayerMatrixForwardCalcNext(NeuronLayer* layer, int layer_size, int prev_layer_size, double* inputs, 
										int subset_size, double* output)
{
	for (int i = 0; i < subset_size; i++) {
		for (int j = 0; j < layer_size; j++) {
			output[i * layer_size + j] = layer->biases[j];
		}
	}

	int matrix1_size = prev_layer_size * subset_size;
	int matrix2_size = layer_size * prev_layer_size;
	int out_matrix_size = layer_size * subset_size;

	matrixMultiply(
		true,				// row_first_matrix1
		true,				// row_first_matrix2
		false,				// transpose_matrix1
		true,				// transpose_matrix2
		inputs,				// matrix1
		matrix1_size,		// matrix1_size
		prev_layer_size,	// matrix1_per
		layer->weights,		// matrix2
		matrix2_size,		// matrix2_size
		prev_layer_size,	// matrix2_per
		output,				// out_matrix
		out_matrix_size,	// out_matrix_size
		true,				// out_matrix_row_first
		1,					// alpha
		1					// beta
	);
	matrixReLU(output, output, out_matrix_size);
}

/*
performs a forward pass through a NeuralNetwork given an input matrix array.
sets values for each NeuronLayer inside the NeuralNetworkTrainer for backpropogation.
every row of inputs is considered one expirement.
*/
void neuralNetworkMatrixForwardPass(NeuralNetwork* nn, double* inputs)
{
	if (nn->trainer == NULL) {
		perror("Attempted forward pass without neural network trainer");
		return;
	}

	for (int i = 0; i < nn->layer_num; i++) {
		int prev_layer_size;
		double* prev_layer_values;
		if (i == 0) {
			prev_layer_size = nn->input_num;
			prev_layer_values = inputs;
		} else {
			prev_layer_size = nn->layer_sizes[i - 1];
			prev_layer_values = nn->trainer->values[i - 1];
		}

		neuronLayerMatrixForwardCalcNext(
			nn->layers[i],
			nn->layer_sizes[i],
			prev_layer_size,
			prev_layer_values,
			nn->trainer->subset_size,
			nn->trainer->values[i]
		);
	}
}

/*
intakes a layer and its values, costs, and the previus layer's values, as well as pointers to memory that can be used to store temp values.
ajusts the layer's weights and biases by differenciating the cost function for each paramiter.
*/
void neuronLayerBackPropogateCalcNext(NeuronLayer* layer, double* cur_values, double* prev_values, double* costs, double* dBiases, 
										double* dWeights, int subnet_size, int layer_size, int prev_layer_size, bool calc_prev_costs, 
										double* prev_costs, double training_rate)
{
	int cur_values_size = subnet_size * layer_size;
	int prev_values_size = subnet_size * prev_layer_size;
	int weights_size = layer_size * prev_layer_size;

	// iterate over values, if zero change corisponding cost to zero.
	for (int i = 0; i < cur_values_size; i++) {
		if (cur_values[i] == 0) {
			costs[i] = 0;
		}
	}

	// update biases, iterate over costs add them up in dBias then divide by subnet num and subtract from biases.
	for (int i = 0; i < cur_values_size; i++) {
		dBiases[i % layer_size] = (i < layer_size) ? costs[i] : dBiases[i % layer_size] + costs[i];
	}
	for (int i = 0; i < layer_size; i++) {
		dBiases[i] /= subnet_size;
		layer->biases[i] -= dBiases[i] * training_rate;
	}

	// calc prev costs, multiply costs by weights.
	if (calc_prev_costs) {
		matrixMultiply(
			true,				// row_first_matrix1
			true,				// row_first_matrix2
			false,				// transpose_matrix1
			false,				// transpose_matrix2
			costs,				// matrix1
			cur_values_size,	// matrix1_size
			layer_size,			// matrix1_per
			layer->weights,		// matrix2
			weights_size,		// matrix2_size
			prev_layer_size,	// matrix2_per
			prev_costs,			// out_matrix
			prev_values_size,	// out_matrix_size
			true,				// out_matrix_row_first
			1,					// alpha
			0					// beta
		);
	}

	// update weights, iterate over subnet num multipling the vector from values and costs 
	// and depositiing it into dWeights, then divide by subnet num and subtract.
	for (int i = 0; i < subnet_size; i++) {
		double beta = (i == 0) ? 0 : 1.0f;
		double* cost_pointer = costs + i * layer_size;
		double* prev_value_pointer = prev_values + i * prev_layer_size;
		matrixMultiply(
			false,				// row_first_matrix1
			true,				// row_first_matrix2
			false,				// transpose_matrix1
			false,				// transpose_matrix2
			cost_pointer,		// matrix1
			layer_size,			// matrix1_size
			layer_size,			// matrix1_per
			prev_value_pointer,	// matrix2
			prev_layer_size,	// matrix2_size
			prev_layer_size,	// matrix2_per
			dWeights,			// out_matrix
			weights_size,		// out_matrix_size
			true,				// out_matrix_row_first
			1,					// alpha
			beta				// beta
		);
	}
	for (int i = 0; i < weights_size; i++) {
		dWeights[i] /= subnet_size;
		layer->weights[i] -= dWeights[i] * training_rate;
	}
}

/*
backpropagates a NeuralNetwork using a NeuralNetworkTrainer.
the values of the NeuralNetwork need to be set before hand so make sure to perform a forward pass first.
takes in a matrix of expected outputs, the intial inputs used 
*/
double neuralNetworkBackPropogate(NeuralNetwork* nn, double* expected, double* inputs, double training_rate)
{
	if (nn->layer_num < 1) {
		perror("Attempted back propogate with empty neural network");
		return -1;
	}
	double tot_cost = 0;
	double* starting_costs = nn->trainer->costs[nn->layer_num - 1];
	int starting_costs_size = nn->layer_sizes[nn->layer_num - 1] * nn->trainer->subset_size;
	for (int i = 0; i < starting_costs_size; i++) {
		double dif = nn->trainer->values[nn->layer_num - 1][i] - expected[i];
		starting_costs[i] = 2 * dif;
		tot_cost += dif * dif;
	}

	for (int i = nn->layer_num - 1; i >= 0; i--) {
		double* prev_values = (i == 0) ? inputs : nn->trainer->values[i - 1];
		int prev_layer_size = (i == 0) ? nn->input_num : nn->layer_sizes[i - 1];
		double* prev_costs = (i == 0) ? NULL : nn->trainer->costs[i - 1];
		neuronLayerBackPropogateCalcNext(
			nn->layers[i],
			nn->trainer->values[i],
			prev_values,
			nn->trainer->costs[i],
			nn->trainer->dBiases[i],
			nn->trainer->dWeights[i],
			nn->trainer->subset_size,
			nn->layer_sizes[i],
			prev_layer_size,
			(i != 0),
			prev_costs,
			training_rate
		);
	}
	return tot_cost;
}

/*
takes in a NeuralNetwork, inputs, total size of input data, size of each input row, expected values, total expected values size,
expected values row size, number of subsets the data is split into (should be divisable), number of epoches, and the training rate.
trains the neural network on the data set by performing forward passes and back propagation.
*/
void neuralNetworkTrain(NeuralNetwork* nn, double* inputs, int input_size, int input_row_size,
					double* expected, int expected_size, int expected_row_size, int subset_num,
					int epoch_num, double training_rate)
{
	if (nn->input_num != input_row_size || nn->layer_sizes[nn->layer_num - 1] != expected_row_size) {
		perror("In and out of data does not match neural network");
		return;
	}

	if (nn->trainer != NULL) {
		removeNeuralNetworkTrainer(nn);
	}

	int subset_size = input_size / input_row_size / subset_num;

	attachNeuralNetworkTrainer(nn, subset_size);

	int input_subset_tot_size = input_size / subset_num;
	int expected_subset_tot_size = expected_size / subset_num;
	for (int epoch_ind = 0; epoch_ind < epoch_num; epoch_ind++) {
		for (int subset_ind = 0; subset_ind < subset_num; subset_ind++) {
			double* input_pointer = inputs + subset_ind * input_subset_tot_size;
			double* expected_pointer = expected + subset_ind * expected_subset_tot_size;
			neuralNetworkMatrixForwardPass(nn, input_pointer);
			double cost = neuralNetworkBackPropogate(nn, expected_pointer, input_pointer, training_rate);
			//printf("cost: %lf\n", cost);
		}
	}
}

/*

*/
void neuralNetworkDataSetTrain(NeuralNetwork* nn, DataSet* data, int epoch_num, float training_rate)
{
	neuralNetworkTrain(
		nn,
		data->training_input_data,
		data->input_row_size * data->training_row_num,
		data->input_row_size,
		data->training_output_data,
		data->output_row_size * data->training_row_num,
		data->output_row_size,
		1,
		epoch_num,
		training_rate
	);

	removeNeuralNetworkTrainer(nn);
	attachNeuralNetworkTrainer(nn, data->testing_row_num);
	neuralNetworkMatrixForwardPass(nn, data->testing_input_data);
	double cost = neuralNetworkBackPropogate(nn, data->testing_output_data, data->testing_input_data, 0.05f);
	printf("cost: %lf\n", cost);
	// change trainer
	// forward pass testing
	// compare output to expected
}