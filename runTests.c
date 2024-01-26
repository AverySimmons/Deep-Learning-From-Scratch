#pragma once

#define EPSILON 1e-3

#include "matrixOperations.h"
#include "neuralNetwork.h"
#include "dataLoading.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

/*
takes a bool value on whether or not result == expected
outputs debug text to terminal and increases test_count and success_count
variables for further debugging
*/
void testFunction(bool success, int* test_count, int* success_count, char function_name_and_info[]) 
{
    (*test_count)++;
    if (success) {
        (*success_count)++;
        printf("Test %i PASSED - %s\n", *test_count, function_name_and_info);
    }
    else {
        printf("Test %i FAILED - %s\n", *test_count, function_name_and_info);
    }
}

/*
compares two double arrays, returns true if equal, otherwise false.
*/
bool cmpArray(double* array1, double* array2, int size)
{
    for (int i = 0; i < size; i++) {
        if (fabs(array1[i] - array2[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

void testMatrixAdd(int* test_count, int* success_count) 
{
    int size1 = 6;
    double result1[6];
    double first_input1[] = { 1, 3, 2, 4, 5, 6 };
    double second_input1[] = { 2, 4, 2, 2, 4, 4 };
    double expected1[] = { 3, 7, 4, 6, 9, 10 };

    matrixAdd(first_input1, second_input1, result1, size1);
    testFunction(cmpArray(result1, expected1, size1), test_count, success_count, "matrixAdd 1");

    int size2 = 3;
    double result2[3];
    double first_input2[] = { 1.0, 2.0, 3.0 };
    double second_input2[] = { -1.0, -2.0, -3.0 };
    double expected2[] = { 0.0, 0.0, 0.0 };

    matrixAdd(first_input2, second_input2, result2, size2);
    testFunction(cmpArray(result2, expected2, size2), test_count, success_count, "matrixAdd 2");

}

void testMatrixSubtract(int* test_count, int* success_count)
{
    int size1 = 6;
    double result1[6];
    double first_input1[] = { 1, 3, 2, 4, 5, 6 };
    double second_input1[] = { 2, 4, 2, 2, 4, 4 };
    double expected1[] = { -1, -1, 0, 2, 1, 2 };

    matrixSubtract(first_input1, second_input1, result1, size1);
    testFunction(cmpArray(result1, expected1, size1), test_count, success_count, "matrixSubtract 1");

    int size2 = 4;
    double result2[4];
    double first_input2[] = { 10.5, 5.0, 3.5, 8.0 };
    double second_input2[] = { 2.5, 2.0, 1.5, 4.0 };
    double expected2[] = { 8.0, 3.0, 2.0, 4.0 };

    matrixSubtract(first_input2, second_input2, result2, size2);
    testFunction(cmpArray(result2, expected2, size2), test_count, success_count, "matrixSubtract 2");
}

void testMatrixFlatMultiply(int* test_count, int* success_count)
{
    int size1 = 6;
    double result1[6];
    double input1[] = { 1, 3, 2, 4, 5, 6 };
    double scalar1 = 2.0;
    double expected1[] = { 2, 6, 4, 8, 10, 12 };

    matrixFlatMultiply(input1, scalar1, result1, size1);
    testFunction(cmpArray(result1, expected1, size1), test_count, success_count, "matrixFlatMultiply 1");

    int size2 = 4;
    double result2[4];
    double input2[] = { -0.5, 1.5, -2.0, 0.0 };
    double scalar2 = 3.0;
    double expected2[] = { -1.5, 4.5, -6.0, 0.0 };

    matrixFlatMultiply(input2, scalar2, result2, size2);
    testFunction(cmpArray(result2, expected2, size2), test_count, success_count, "matrixFlatMultiply 2");
}

void testMatrixInPlaceMultiply(int* test_count, int* success_count)
{
    int size1 = 6;
    double result1[6];
    double first_input1[] = { 1, 3, 2, 4, 5, 6 };
    double second_input1[] = { 2, 4, 2, 2, 4, 4 };
    double expected1[] = { 2, 12, 4, 8, 20, 24 };

    matrixInPlaceMultiply(first_input1, second_input1, result1, size1);
    testFunction(cmpArray(result1, expected1, size1), test_count, success_count, "matrixInPlaceMultiply 1");

    int size2 = 4;
    double result2[4];
    double first_input2[] = { 1, 0, 0, 0 };
    double second_input2[] = { 1, 1, 1, 1 };
    double expected2[] = { 1, 0, 0, 0 };

    matrixInPlaceMultiply(first_input2, second_input2, result2, size2);
    testFunction(cmpArray(result2, expected2, size2), test_count, success_count, "matrixInPlaceMultiply 2");
}

void testMatrixReLU(int* test_count, int* success_count)
{
    int size1 = 6;
    double result1[6];
    double input1[] = { 1, -3, 2, -4, 5, 0 };
    double expected1[] = { 1, 0, 2, 0, 5, 0 };

    matrixReLU(input1, result1, size1);
    testFunction(cmpArray(result1, expected1, size1), test_count, success_count, "matrixReLU 1");

    int size2 = 3;
    double result2[3];
    double input2[] = { -2.0, -1.0, -0.5 };
    double expected2[] = { 0.0, 0.0, 0.0 };

    matrixReLU(input2, result2, size2);
    testFunction(cmpArray(result2, expected2, size2), test_count, success_count, "matrixReLU 2");
}

void testMatrixReLUPrime(int* test_count, int* success_count)
{
    int size1 = 6;
    double result1[6];
    double input1[] = { 1, -3, 2, -4, 5, 0 };
    double expected1[] = { 1, 0, 1, 0, 1, 0 };

    matrixReLUPrime(input1, result1, size1);
    testFunction(cmpArray(result1, expected1, size1), test_count, success_count, "matrixReLUPrime 1");

    int size2 = 3;
    double result2[3];
    double input2[] = { -2.0, -1.0, -0.5 };
    double expected2[] = { 0.0, 0.0, 0.0 };

    matrixReLUPrime(input2, result2, size2);
    testFunction(cmpArray(result2, expected2, size2), test_count, success_count, "matrixReLUPrime 2");
}

void testMatrixMultiply(int* test_count, int* success_count)
{
    bool row_first_matrix1_1 = false;
    bool row_first_matrix2_1 = false;
    bool transpose_matrix1_1 = false;
    bool transpose_matrix2_1 = false;
    double matrix1_1[] = { 2, 4, 0, 1, 3, 1, 9, 1 };
    int matrix1_size_1 = 8;
    int matrix1_elements_per_1 = 4;
    double matrix2_1[] = { 2, 0, 2, 1, 5, 6};
    int matrix2_size_1 = 6;
    int matrix2_elements_per_1 = 2;
    double out_matrix_1[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int out_matrix_size_1 = 12;
    bool row_first_out_matrix_1 = false;
    double alpha_1 = 1;
    double beta_1 = 0;
    double expected_1[] = { 4, 8, 0, 2, 7, 9, 9, 3, 28, 26, 54, 11 };

    matrixMultiply(row_first_matrix1_1, row_first_matrix2_1, transpose_matrix1_1, transpose_matrix2_1, matrix1_1,
        matrix1_size_1, matrix1_elements_per_1, matrix2_1, matrix2_size_1, matrix2_elements_per_1, out_matrix_1, out_matrix_size_1,
        row_first_out_matrix_1, alpha_1, beta_1);
    testFunction(cmpArray(out_matrix_1, expected_1, out_matrix_size_1), test_count, success_count, "testMatrixMultiply");
}

void testNeuronLayerSetValues(int* test_count, int* success_count)
{
    int prev_layer_size1 = 3;
    int layer_size1 = 2;
    int weight_num1 = 6;
    NeuronLayer* layer1 = createNeuronLayer(prev_layer_size1, layer_size1);
    double weights1[] = { 0.2f, 0.3f, 0.5f, 0.2f, 0.1f, 0.3f };
    double biases1[] = { 1.2f, -2.3f };
    neuronLayerSetValues(layer1, weights1, weight_num1, biases1, layer_size1);

    bool weight_true1 = cmpArray(layer1->weights, weights1, weight_num1);
    bool bias_true1 = cmpArray(layer1->biases, biases1, layer_size1);
    testFunction(weight_true1 && bias_true1, test_count, success_count, "testNeuronLayerSetValues");
    freeNeuronLayer(layer1);
}

void testNeuronLayerMatrixForwardCalcNext(int* test_count, int* success_count)
{
    int prev_layer_size1 = 3;
    int layer_size1 = 2;
    int weight_num1 = 6;
    NeuronLayer* layer1 = createNeuronLayer(prev_layer_size1, layer_size1);
    double weights1[] = { 0.2f, 0.3f, -0.5f, -0.2f, 0.1f, 0.3f };
    double biases1[] = { 1.2f, -2.3f };
    neuronLayerSetValues(layer1, weights1, weight_num1, biases1, layer_size1);
    int subset_size = 3;
    double inputs1[] = { 0.5f, 0, -1, 500, 2, 0.001f, 0, 0, 0 };
    double output1[6];
    double expected1[] = { 1.8f, 0, 101.7995f, 0, 1.2f, 0 };
    neuronLayerMatrixForwardCalcNext(layer1, layer_size1, prev_layer_size1, inputs1, subset_size, output1);

    testFunction(cmpArray(output1, expected1, 6), test_count, success_count, "testNeuronLayerMatrixForwardCalcNext");
    freeNeuronLayer(layer1);
}

void testNeuralNetworkMatrixForwardPass(int* test_count, int* success_count)
{
    int input_num1 = 2;
    int layer_num1 = 3;
    int layer_sizes1[] = { 2, 3, 2 };
    double biases1_1[] = { 1, -0.5f };
    double biases2_1[] = { 0, 1.5f, 0.2f };
    double biases3_1[] = { 1.2f, 0.5f };
    double weights1_1[] = { 0.1f, 0.2f, 0.5f, 0.3f };
    double weights2_1[] = { 0.3f, -0.1f, 0.51f, 0.7f, 0.2f, 0 };
    double weights3_1[] = { 1, 0.2f, 0.7f, 0.3f, 0.1f, -0.01f };
    double* bias_arr1[] = { biases1_1, biases2_1, biases3_1 };
    double* weight_arr1[] = {weights1_1, weights2_1, weights3_1};
    
    int subset_size1 = 2;
    NeuralNetwork* nn1 = createNeuralNetwork(input_num1, layer_num1, layer_sizes1);
    for (int i = 0; i < layer_num1; i++) {
        int weight_num = (i == 0) ? 4 : 6;
        neuronLayerSetValues(nn1->layers[i], weight_arr1[i], weight_num, bias_arr1[i], layer_sizes1[i]);
    }

    attachNeuralNetworkTrainer(nn1, subset_size1);
    double inputs1[] = { 0.1f, 1, -1, -0.4f };
    double expected1[] = { 2.29582f, 0.81619f, 2.08444f, 0.76198f };
    neuralNetworkMatrixForwardPass(nn1, inputs1);
    double* output1 = nn1->trainer->values[2];
    testFunction(cmpArray(output1, expected1, 4), test_count, success_count, "testNeuralNetworkMatrixForwardPass");

    freeNeuralNetwork(nn1);
}

void testNeuronLayerBackPropogateCalcNext(int* test_count, int* success_count)
{
    int layer_size1 = 2;
    int prev_layer_size1 = 2;
    int weight_num1 = 4;
    int subnet_size1 = 2;
    double weights1[] = { 0.2f, 0.1f, -1.2f, 0.3f };
    double biases1[] = { 1, 0.5f };
    double values1[] = { 1.09f, 0, 1.01f, 1.12f };
    double prev_values1[] = { 0.5f, -0.1f, -0.05f, 0.2f };
    double costs1[] = { 0.99f, -0.5f, 0.51f, 0.92f };
    double training_rate1 = 0.05f;
    double dWeights1[4];
    double dBiases1[2];
    double out_costs1[4];
    NeuronLayer* layer1 = createNeuronLayer(prev_layer_size1, layer_size1);
    neuronLayerSetValues(layer1, weights1, weight_num1, biases1, layer_size1);
    neuronLayerBackPropogateCalcNext(layer1, values1, prev_values1, costs1, dBiases1,
        dWeights1, subnet_size1, layer_size1, prev_layer_size1, true, out_costs1, training_rate1);

    double expected_out_costs1[] = { 0.198f, 0.099f, -1.002f, 0.327f };
    double expected_weights1[] = { 0.188263f, 0.099925f, -1.19885f, 0.2954f };
    double expected_biases1[] = { 0.9625f, 0.477f };
    bool cost_success1 = cmpArray(expected_out_costs1, out_costs1, subnet_size1 * layer_size1);
    bool weight_success1 = cmpArray(expected_weights1, layer1->weights, weight_num1);
    bool bias_success1 = cmpArray(expected_biases1, layer1->biases, layer_size1);

    testFunction(cost_success1 && weight_success1 && bias_success1, test_count, success_count, "testNeuronLayerBackPropogateCalcNext");
    freeNeuronLayer(layer1);
}

void testNeuralNetworkTrain(int* test_count, int* success_count)
{
    int input_num1 = 3;
    double inputs1[] = { 
        0, 0, 0,
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
        1, 1, 0, 
        1, 0, 1,
        0, 1, 1,
        1, 1, 1
    };
    int layer_num1 = 3;
    int layer_sizes1[] = { 20, 20, 1 };
    double expected1[] = { 
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    };
    NeuralNetwork* nn1 = createNeuralNetwork(input_num1, layer_num1, layer_sizes1);
    neuralNetworkTrain(nn1, inputs1, 24, 3, expected1, 8, 1, 1, 4000, 0.05f);

    neuralNetworkMatrixForwardPass(nn1, inputs1);
    for (int i = 0; i < nn1->layer_sizes[nn1->layer_num - 1] * nn1->trainer->subset_size; i++) {
        printf("%f ", nn1->trainer->values[nn1->layer_num - 1][i]);
    }
    printf("\n");

    freeNeuralNetwork(nn1);
}

void testMatrixOperations(int* test_count, int* success_count)
{
    testMatrixAdd(test_count, success_count);
    testMatrixSubtract(test_count, success_count);
    testMatrixFlatMultiply(test_count, success_count);
    testMatrixInPlaceMultiply(test_count, success_count);
    testMatrixReLU(test_count, success_count);
    testMatrixReLUPrime(test_count, success_count);
    testMatrixMultiply(test_count, success_count);
}

void testNeuralNetwork(int* test_count, int* success_count)
{
    testNeuronLayerSetValues(test_count, success_count);
    testNeuronLayerMatrixForwardCalcNext(test_count, success_count);
    testNeuralNetworkMatrixForwardPass(test_count, success_count);
    testNeuronLayerBackPropogateCalcNext(test_count, success_count);
    testNeuralNetworkTrain(test_count, success_count);
}

int main() 
{
    srand(1);

    int test_count = 0;
    int success_count = 0;

    // test functions
    testMatrixOperations(&test_count, &success_count);
    testNeuralNetwork(&test_count, &success_count);


    printf("%i/%i tests passed\n", success_count, test_count);

    int input_num = 33 - 3;
    int output_num = 1;
    int remove_num = 2;
    double test_percent = 0.05;
    int output_indexes[] = { 2 };
    int removed_indexs[] = { 0, 31 };
    DataSet* data = loadCSV("DATA (1).csv", output_indexes, output_num, removed_indexs, remove_num, test_percent);
    dataSetNormalize(data);
    int layer_num = 3;
    int layer_sizes[] = { 50, 50, output_num };
    NeuralNetwork* nn = createNeuralNetwork(input_num, layer_num, layer_sizes);
    int epoch_num = 30000;
    float training_rate = 0.05f;
    neuralNetworkDataSetTrain(nn, data, epoch_num, training_rate);

    freeDataSet(data);
    freeNeuralNetwork(nn);
    
    return 0;
}