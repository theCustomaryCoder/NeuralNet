#include <stdlib.h>
#include <math.h>
#ifndef NEURALNET_H
#define NEURALNET_H

#define MAX_NODES_PER_LAYER 10

double sigmoid(double x){
    return 1 / (1 + exp(-x)); 
}

double dSigmoid(double x){
    return x * (1 - x);
}

typedef struct {
    int numberOfNodes;
    double nodes[MAX_NODES_PER_LAYER];
    double layerWeights[MAX_NODES_PER_LAYER][MAX_NODES_PER_LAYER];
    double layerBias[MAX_NODES_PER_LAYER];
    double deltaLayer[MAX_NODES_PER_LAYER];
    Layer* prevLayer;
} Layer;

Layer generateLayer(int numberOfNodes, Layer* prevLayer){
    Layer layer;
    layer.numberOfNodes = numberOfNodes;
    layer.prevLayer = prevLayer;

    for(int i = 0; i < numberOfNodes; i++){
        layer.layerBias[i] = ((double)rand()) / ((double)RAND_MAX);
        for(int j = 0; j < prevLayer->numberOfNodes; j++){
            layer.layerWeights[j][i] = ((double)rand()) / ((double)RAND_MAX);
        }
    }

    return layer;
}

Layer generateLayer(int numberOfNodes){
    Layer layer;
    layer.numberOfNodes = numberOfNodes;

    for(int i = 0; i < numberOfNodes; i++){
        layer.layerBias[i] = ((double)rand()) / ((double)RAND_MAX);
    }

    return layer;
}

void computeLayerActivation(Layer* layer, Layer *prevLayer){
    for(int i = 0; i < layer->numberOfNodes; i++){
        double activation = layer->layerBias[i];

        for(int j = 0; j < prevLayer->numberOfNodes; j++){
            activation += (prevLayer->nodes[j] * layer->layerWeights[j][i]);
        }

        layer->nodes[i] = sigmoid(activation);
    }
}

void applyChanges(Layer* layer, Layer *prevLayer, double learningRate){
    for(int i = 0; i < layer->numberOfNodes; i++){
        layer->layerBias[i] += layer->deltaLayer[i] * learningRate;
        for(int j = 0; j < prevLayer->numberOfNodes; j++){
            layer->layerWeights[j][i] += prevLayer->nodes[j] * layer->deltaLayer[i] * learningRate;
        }
    }
}

void calculateDelta(Layer* layer, Layer *nextLayer){
    for(int i = 0; i < layer->numberOfNodes; i++){
        double error = 0.0f;
        for(int j = 0; j < nextLayer->numberOfNodes; j++){
            error+= nextLayer->deltaLayer[j] * nextLayer->layerWeights[i][j];
        }
        layer->deltaLayer[i] = error * dSigmoid(layer->nodes[i]);
    }
}

void calculateDelta(Layer* layer, double *trainingData){
    for(int i = 0; i < layer->numberOfNodes; i++){
        double error = (trainingData[i] - layer->nodes[i]);
        layer->deltaLayer[i] = error * dSigmoid(layer->nodes[i]);
    }
}
#endif