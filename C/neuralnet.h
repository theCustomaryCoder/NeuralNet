#ifndef NEURALNET_H
#define NEURALNET_H

#define MAX_NODES_PER_LAYER 10

typedef struct {
    int numberOfNodes;
    double nodes[MAX_NODES_PER_LAYER];
    double layerWeights[MAX_NODES_PER_LAYER][MAX_NODES_PER_LAYER];
    double layerBias[MAX_NODES_PER_LAYER];
    double deltaLayer[MAX_NODES_PER_LAYER];
} Layer;

#endif