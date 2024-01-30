#include "layers.hpp"
#include <stdio.h>
#include <fstream>

#ifndef NEURALNET_HPP
#define NEURALNET_HPP
#define MAX_NUMBER_OF_HIDDEN_LAYERS 30
#define MAX_SIZE_OF_TRAINING_SET 40
#define MEDIUM_HIT_PROBABILITY 0.5f
#define HIGH_HIT_PROBABILITY 0.7f

double forecast(int n, double p){
    double my = n * p;
    double sigma = sqrt(my * (1 - p));
    return my + 1.96f * sigma;
}

typedef struct {
    InputLayer inputLayer;
    OutputLayer outputLayer;
    int numberOfHiddenLayers;
    HiddenLayer hiddenLayers[MAX_NUMBER_OF_HIDDEN_LAYERS];
} ImportData;

ImportData importNetFromFile(char *path){
    ImportData importData;
    std::ifstream fin(path);
    int nodesOfPrevLayer;

    //Loading input layer
    int inputNodes;
    fin >> inputNodes;
    importData.inputLayer = InputLayer(inputNodes);
    nodesOfPrevLayer = inputNodes;

    //Loading hidden layers
    int numberOfHiddenLayers;
    int layerNodes;
    double layerBias[MAX_NODES_PER_LAYER];
    double layerWeights[MAX_NODES_PER_LAYER][MAX_NODES_PER_LAYER];
    fin >> numberOfHiddenLayers;
    for(int i = 0; i < numberOfHiddenLayers; i++){
        fin >> layerNodes;
        for(int j = 0; j < layerNodes; j++){
            fin >> layerBias[j];
        }
        for(int j = 0; j < layerNodes; j++){
            for(int k = 0; k < nodesOfPrevLayer; k++){
                fin >> layerWeights[k][j];
            }
        }
        importData.hiddenLayers[i] = HiddenLayer(layerNodes, layerBias, layerWeights);
        nodesOfPrevLayer = layerNodes;
    }
    importData.numberOfHiddenLayers = numberOfHiddenLayers;

    //Loading output layer
    int outputNodes;
    double outputLayerBias[MAX_NODES_PER_LAYER];
    double outputLayerWeights[MAX_NODES_PER_LAYER][MAX_NODES_PER_LAYER];
    fin >> outputNodes;
    for(int i = 0; i < outputNodes; i++){
        fin >> outputLayerBias[i];
    }
    for(int i = 0; i < outputNodes; i++){
        for(int j = 0; j < nodesOfPrevLayer; j++){
            fin >> outputLayerWeights[j][i];
        }
    }
    importData.outputLayer = OutputLayer(outputNodes, outputLayerBias, outputLayerWeights);

    fin.close();
    return importData;
}

class TrainingData {
    private:
        double trainingInput[MAX_SIZE_OF_TRAINING_SET][MAX_NODES_PER_LAYER];
        double trainingOutput[MAX_SIZE_OF_TRAINING_SET][MAX_NODES_PER_LAYER];
        int playlist[MAX_SIZE_OF_TRAINING_SET];
        int siezeOfTrainingSet;
    public:
        TrainingData(int setSize, double (*inputGenerator) (int, int), double (*outputGenerator) (int, int)){
            for(int i = 0; i < setSize; i++){
                for(int j = 0; j < MAX_NODES_PER_LAYER; j++){
                    this->trainingInput[i][j] = inputGenerator(i, j);
                    this->trainingOutput[i][j] = outputGenerator(i, j);
                } 
            }
            this->siezeOfTrainingSet = setSize;
            for(int i = 0; i < siezeOfTrainingSet; i++){
                this->playlist[i] = i;
            }
        }

        TrainingData(int setSize, double trainingInput[MAX_SIZE_OF_TRAINING_SET][MAX_NODES_PER_LAYER], double trainingOutput[MAX_SIZE_OF_TRAINING_SET][MAX_NODES_PER_LAYER]){
            for(int i = 0; i < setSize; i++){
                for(int j = 0; j < MAX_NODES_PER_LAYER; j++){
                    this->trainingInput[i][j] = trainingInput[i][j];
                    this->trainingOutput[i][j] = trainingOutput[i][j];
                } 
            }
            this->siezeOfTrainingSet = setSize;
            for(int i = 0; i < siezeOfTrainingSet; i++){
                this->playlist[i] = i;
            }
        }

        void shufflePlaylist(){
            if(this->siezeOfTrainingSet > 1){
                size_t i;
                 for(i = 0; i < this->siezeOfTrainingSet - 1; i++){
                    size_t j = i + rand() / (RAND_MAX / (this->siezeOfTrainingSet - i) + 1);
                    int t = this->playlist[j];
                    this->playlist[j] = this->playlist[i];
                    this->playlist[i] = t;
                }
            }
        }

        double getTrainingInput(int i, int nodeNumber){
            return this->trainingInput[i][nodeNumber];
        }

        double getTrainingOutput(int i, int nodeNumber){
            return this->trainingOutput[i][nodeNumber];
        }

        int getSizeOfTrainingSet(){
            return this->siezeOfTrainingSet;
        }

        int getSampleIndexFromPlaylist(int i){
            return this->playlist[i];
        }
};

class NeuralNet {
    private:
        InputLayer *inputLayer;
        HiddenLayer *hiddenLayers;
        OutputLayer *outputLayer;
        int numberOfHiddenLayers;
    public:
        NeuralNet(InputLayer *inputLayer, HiddenLayer *hiddenLayers, int numberOfHiddenLayers, OutputLayer *outputLayer){
            this->inputLayer = inputLayer;
            this->hiddenLayers = hiddenLayers;
            this->outputLayer = outputLayer;
            this->numberOfHiddenLayers = numberOfHiddenLayers;
        }

        NeuralNet() = default;

        void train(TrainingData trainingData, int epochs, double learningRate, bool debugInfoOutput){
            TrainingData dataset = trainingData;
            int setSize = dataset.getSizeOfTrainingSet();
            int correct = 0;
            for(int i = 1; i <= epochs; i++){
                if(debugInfoOutput){
                    printf("Epoch: %d\n", i);
                }
                dataset.shufflePlaylist();
                for(int j = 0; j < setSize; j++){
                    int sampleIndex = dataset.getSampleIndexFromPlaylist(j);

                    //Setup for inputLayer
                    for(int nodeNumber = 0; nodeNumber < (this->inputLayer->getNumberOfNodes()); nodeNumber++){
                        this->inputLayer->setNodeValue(nodeNumber, dataset.getTrainingInput(sampleIndex, nodeNumber));
                    }
                    if(debugInfoOutput || i == epochs){
                        printf("Input[");
                        for(int ct = 0; ct < this->inputLayer->getNumberOfNodes(); ct++){
                            printf("%g ", this->inputLayer->getNodeValue(ct));
                        }
                        printf("]\n");
                    }

                    //Pass data through hidden layers
                    if(this->numberOfHiddenLayers >= 1){
                        this->hiddenLayers[0].computeLayerActivation((this->inputLayer));
                    }
                    if(this->numberOfHiddenLayers > 1){
                        for(int layerNumber = 1; layerNumber < this->numberOfHiddenLayers; layerNumber++){
                            this->hiddenLayers[layerNumber].computeLayerActivation(&(this->hiddenLayers[layerNumber - 1]));
                        }
                    }

                    //Pass data through outputLayer
                    if(this->numberOfHiddenLayers == 0){
                        this->outputLayer->computeLayerActivation(this->inputLayer);
                    }
                    else{
                        this->outputLayer->computeLayerActivation(&(this->hiddenLayers[this->numberOfHiddenLayers - 1]));
                    }

                    if(debugInfoOutput || i == epochs){
                        for(int nodeNumber = 0; nodeNumber < this->outputLayer->getNumberOfNodes(); nodeNumber++){
                            if(round(this->outputLayer->getNodeValue(nodeNumber)) == dataset.getTrainingOutput(sampleIndex, nodeNumber)){
                                printf("\033[0;32m");
                                correct++;
                            }
                            else{
                                printf("\033[0;31m");
                            }
                            printf("Expected output: %g     Predicted output: %g\n", dataset.getTrainingOutput(sampleIndex, nodeNumber), this->outputLayer->getNodeValue(nodeNumber));
                            printf("\033[0m");
                        }
                        printf("\n");
                    }

                    //Backpropagation
                    //Compute change in output Weights
                    double sample[MAX_NODES_PER_LAYER];
                    for(int sampleNodeIndex = 0; sampleNodeIndex < dataset.getSizeOfTrainingSet(); sampleNodeIndex++){
                        sample[sampleNodeIndex] = dataset.getTrainingOutput(sampleIndex, sampleNodeIndex);
                    }
                    this->outputLayer->calculateDelta(sample);

                    //Compute change in hidden weights
                    if(this->numberOfHiddenLayers >= 1){
                        this->hiddenLayers[this->numberOfHiddenLayers - 1].calculateDelta(this->outputLayer);
                    }
                    if(this->numberOfHiddenLayers > 1){
                        for(int layerNumber = 2; layerNumber <= this->numberOfHiddenLayers; layerNumber++){
                            //HiddenLayerIF* layer = &(this->hiddenLayers[this->numberOfHiddenLayers - layerNumber + 1]);
                            this->hiddenLayers[this->numberOfHiddenLayers - layerNumber].calculateDelta(&(this->hiddenLayers[this->numberOfHiddenLayers - layerNumber + 1]));
                        }
                    }

                    //Apply changes to output weights
                    if(this->numberOfHiddenLayers == 0){
                        this->outputLayer->applyChanges(this->inputLayer, learningRate);
                    }
                    else{
                        this->outputLayer->applyChanges(&(this->hiddenLayers[this->numberOfHiddenLayers -1]), learningRate);
                    }

                    //Apply changes to hidden weights
                    if(this->numberOfHiddenLayers >= 1){
                        this->hiddenLayers[0].applyChanges(this->inputLayer, learningRate);
                    }
                    if(this->numberOfHiddenLayers > 1){
                        for(int layerNumber = 1; layerNumber < this->numberOfHiddenLayers; layerNumber++){
                            this->hiddenLayers[layerNumber].applyChanges(&(this->hiddenLayers[layerNumber - 1]), learningRate);
                        }
                    }
                }
                if(debugInfoOutput || i == epochs){
                    printf("CORRECT:");
                    if(correct > forecast(dataset.getSizeOfTrainingSet(), HIGH_HIT_PROBABILITY)){
                        printf("\033[0;32m");
                    }
                    else if(correct > forecast(dataset.getSizeOfTrainingSet(), MEDIUM_HIT_PROBABILITY)){
                        printf("\033[0;33m");
                    }
                    else{
                        printf("\033[0;31m");
                    }
                    printf("    %d/%d\n", correct, dataset.getSizeOfTrainingSet());
                    printf("\033[0m");
                    printf("######################################################################\n");
                }
                correct = 0;
            }
        }

        double *eval(double* input){
            //Setup for inputLayer
            for(int nodeNumber = 0; nodeNumber < (this->inputLayer->getNumberOfNodes()); nodeNumber++){
                this->inputLayer->setNodeValue(nodeNumber, input[nodeNumber]);
            }

            //Pass data through hidden layers
            if(this->numberOfHiddenLayers >= 1){
                this->hiddenLayers[0].computeLayerActivation((this->inputLayer));
            }
            if(this->numberOfHiddenLayers > 1){
                for(int layerNumber = 1; layerNumber < this->numberOfHiddenLayers; layerNumber++){
                    this->hiddenLayers[layerNumber].computeLayerActivation(&(this->hiddenLayers[layerNumber - 1]));
                }
            }

            //Pass data through outputLayer
            if(this->numberOfHiddenLayers == 0){
                this->outputLayer->computeLayerActivation(this->inputLayer);
            }
            else{
                this->outputLayer->computeLayerActivation(&(this->hiddenLayers[this->numberOfHiddenLayers - 1]));
            }
            static double out[MAX_NODES_PER_LAYER];
            for(int outLoop = 0; outLoop < this->outputLayer->getNumberOfNodes(); outLoop++){
                out[outLoop] = this->outputLayer->getNodeValue(outLoop);
            }
            return out;
        }

        void loadImport(ImportData importData){
            this->inputLayer = &importData.inputLayer;
            this->hiddenLayers = importData.hiddenLayers;
            this->outputLayer = &importData.outputLayer;
            this->numberOfHiddenLayers = importData.numberOfHiddenLayers;
        }

        void saveToFile(char *path){
            std::ofstream fout;
            fout.open(path);

            //Export input layer
            fout << this->inputLayer->getNumberOfNodes() << std::endl;

             //Export hidden layers
            fout << this->numberOfHiddenLayers << std::endl;
            for(int i = 0; i < this->numberOfHiddenLayers; i++){
                fout << this->hiddenLayers[i].getNumberOfNodes() << std::endl;
                for(int j = 0; j < this->hiddenLayers[i].getNumberOfNodes(); j++){
                    fout << this->hiddenLayers[i].getLayerBias(j) << " ";
                }
                fout << std::endl;
                for(int j = 0; j < this->hiddenLayers[i].getNumberOfNodes(); j++){
                    if(i == 0){
                        for(int k = 0; k < this->inputLayer->getNumberOfNodes(); k++){
                            fout << this->hiddenLayers[i].getLayerWeights(k, j) << " ";
                        }
                    }
                    else {
                        for(int k = 0; k <this->hiddenLayers[i - 1].getNumberOfNodes(); k++){
                            fout << this->hiddenLayers[i].getLayerWeights(k, j) << " ";
                        }
                    }
                    fout << std::endl;
                }
            }

            //Export output layer
            fout << this->outputLayer->getNumberOfNodes() << std::endl;
            for(int i = 0; i < this->outputLayer->getNumberOfNodes(); i++){
                fout << this->outputLayer->getLayerBias(i) << " ";
            }
            fout << std::endl;
            for(int i = 0; i < this->outputLayer->getNumberOfNodes(); i++){
                if(this->numberOfHiddenLayers == 0){
                    for(int j = 0; j < this->inputLayer->getNumberOfNodes(); j++){
                        fout << this->outputLayer->getLayerWeights(j, i) << " ";
                    }
                    fout << std::endl;
                }
                else {
                    for(int j = 0; j < this->hiddenLayers[this->numberOfHiddenLayers - 1].getNumberOfNodes(); j++){
                        fout << this->outputLayer->getLayerWeights(j, i) << " ";
                    }
                    fout << std::endl;
                }
            }

           fout.close(); 
        }
};
#endif