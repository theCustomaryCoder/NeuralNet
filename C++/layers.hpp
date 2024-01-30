#include <stdlib.h>
#include <math.h>
#ifndef LAYERS_HPP
#define LAYERS_HPP
#define MAX_NODES_PER_LAYER 10

double sigmoid(double x){
    return 1 / (1 + exp(-x)); 
}

double dSigmoid(double x){
    return x * (1 - x);
}

class Layer {
    private:
        int numberOfNodes;
        double nodes[MAX_NODES_PER_LAYER];
        double layerWeights[MAX_NODES_PER_LAYER][MAX_NODES_PER_LAYER];
        double layerBias[MAX_NODES_PER_LAYER];
        double deltaLayer[MAX_NODES_PER_LAYER];
    
    public:

        double initWeights(){
            return ((double)rand()) / ((double)RAND_MAX);
        }

        double getDeltaLayer(int node){
            return this->deltaLayer[node];
        }

        int getNumberOfNodes(){
            return this->numberOfNodes;
        }

        double getNodeValue(int nodeIndex){
            return this->nodes[nodeIndex];
        }

        double getLayerBias(int nodeIndex){
            return this->layerBias[nodeIndex];
        }

        double getLayerWeights(int prevLayerNodeIndex, int nodeIndex){
            return this->layerWeights[prevLayerNodeIndex][nodeIndex];
        }

        void setNodeValue(int nodeIndex, double value){
            this->nodes[nodeIndex] = value;
        }

        void setNumberOfNodes(int numberOfNodes){
            this->numberOfNodes = numberOfNodes;
        }

        void setLayerBias(int nodeIndex, double value){
            this->layerBias[nodeIndex] = value;
        }
        void setLayerWeights(int prevLayerNodeIndex, int nodeIndex, double value){
            this->layerWeights[prevLayerNodeIndex][nodeIndex] = value;
        }

        void setDeltaLayer(int nodeIndex, double value){
            this->deltaLayer[nodeIndex] = value;
        }

        void computeLayerActivation(Layer *prevLayer){
            for(int i = 0; i < this->getNumberOfNodes(); i++){
                double activation = this->getLayerBias(i);

                for(int j = 0; j < prevLayer->getNumberOfNodes(); j++){
                    activation += (prevLayer->getNodeValue(j) * this->getLayerWeights(j, i));
                }

                this->setNodeValue(i, sigmoid(activation));
            }
        }

        void applyChanges(Layer *prevLayer, double learningRate){
            for(int i = 0; i < this->getNumberOfNodes(); i++){
                this->setLayerBias(i, this->getLayerBias(i) + this->getDeltaLayer(i)* learningRate);
                for(int j = 0; j < prevLayer->getNumberOfNodes(); j++){
                    this->setLayerWeights(j, i, this->getLayerWeights(j, i) + prevLayer->getNodeValue(j) * this->getDeltaLayer(i) * learningRate);
                }
            }
        }
};

class InputLayer : public Layer {
    public:
        InputLayer(int numberOfNodes){
            this->setNumberOfNodes(numberOfNodes);
        }

        InputLayer() = default;
};

class HiddenLayer : public Layer {
    public:
        HiddenLayer(int numberOfNodes, Layer *prevLayer){
            this->setNumberOfNodes(numberOfNodes);
            for(int i = 0; i < this->getNumberOfNodes(); i++){
                this->setLayerBias(i, this->initWeights());
                for(int j = 0; j< prevLayer->getNumberOfNodes(); j++){
                    this->setLayerWeights(j, i, this->initWeights());
                }
            }
        }

        HiddenLayer(int nodes, double *layerBias, double layerWeights[MAX_NODES_PER_LAYER][MAX_NODES_PER_LAYER]){
            this->setNumberOfNodes(nodes);
            for(int i = 0; i < nodes; i++){ //Possible error nodes<-> MAX_NODES_PER_LAYER
                this->setLayerBias(i, layerBias[i]);
                for(int j = 0; j < MAX_NODES_PER_LAYER; j++){
                    this->setLayerWeights(j, i, layerWeights[j][i]);
                }
            }
        }

        HiddenLayer() = default;

        void calculateDelta(Layer *nextLayer){
            for(int i = 0; i < this->getNumberOfNodes(); i++){
                double error = 0.0f;
                for(int j = 0; j < nextLayer->getNumberOfNodes(); j++){
                    error+= nextLayer->getDeltaLayer(j) * nextLayer->getLayerWeights(i, j);
                }
                this->setDeltaLayer(i, error * dSigmoid(this->getNodeValue(i)));
            }
        }
};

class OutputLayer : public Layer {
    public:
        OutputLayer(int numberOfNodes, Layer *prevLayer){
            this->setNumberOfNodes(numberOfNodes);

            for(int i = 0; i <numberOfNodes; i++){
                this->setLayerBias(i, this->initWeights());
                for(int j = 0; j< prevLayer->getNumberOfNodes(); j++){
                    this->setLayerWeights(j, i, this->initWeights());
                }
            }
        }

        OutputLayer(int numberOfNodes, double *layerBias, double layerWeights[MAX_NODES_PER_LAYER][MAX_NODES_PER_LAYER]){
            this->setNumberOfNodes(numberOfNodes);
            for(int i = 0; i < numberOfNodes; i++){ //Possible error nodes<-> MAX_NODES_PER_LAYER
                this->setLayerBias(i, layerBias[i]);
                for(int j = 0; j < MAX_NODES_PER_LAYER; j++){
                    this->setLayerWeights(j, i, layerWeights[j][i]);
                }
            }
        }

        OutputLayer() = default;

        void calculateDelta(double *trainingData){
            for(int i = 0; i <this->getNumberOfNodes(); i++){
                double error = (trainingData[i] - this->getNodeValue(i));
                this->setDeltaLayer(i, error * dSigmoid(this->getNodeValue(i)));
            }
        }
};
#endif