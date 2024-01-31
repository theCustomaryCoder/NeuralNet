#include <stdlib.h>
#ifndef NEURALNET_H
#define NEURALNET_H

#define MAX_NODES_PER_LAYER 10

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

#endif