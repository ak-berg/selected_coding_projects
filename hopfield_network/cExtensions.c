/*
C Extension package for the implementation of a Hopfield network in Python.
This package includes functions to set the interaction- as well as the distance-matrix. 
*/

#include <stdlib.h>

// Set the interaction matrix of the neuronal network.
void setInteractionMatrix(int patternNumber, int patternSize, int patternSize2, double patterns[patternNumber][patternSize2], double W[patternSize2][patternSize2]){
    for (int i = 0; i < patternSize2; i++)
    {
        for (int j = 0; j < patternSize2; j++)
        {
            if (i != j)
            {
                if (i < j)
                {
                    for (int k = 0; k < patternNumber; k++)
                    {
                        W[i][j] = W[i][j] + patterns[k][i] * patterns[k][j] / patternSize;
                    }
                }
                else
                {
                    W[i][j] = W[j][i];
                }
                
            }
        } 
    }
}

// Help-function for 'setDistanceMatrix' (see below).
double countMismatches(int patternSize2, double a[patternSize2], double b[patternSize2]){
    int mismatches = 0;

    for (int i = 0; i < patternSize2; i++)
    {
        if (a[i] != b[i])
        {
            mismatches++;
        }    
    }
    return mismatches;
}

// Set the distance matrix of the neuronal network.
void setDistanceMatrix(int iterations, int patternNumber, int patternSize, int patternSize2, double patterns[patternNumber][patternSize2], double W[patternSize2][patternSize2], double newVec[patternSize2], double distMat[iterations][patternNumber]){

    for (int iteration = 0; iteration < iterations; iteration++)
    {
        for (int i = 0; i < patternSize2; i++)
        {
            int neuron = rand() % (patternSize2 + 1);
            double delta = 0;

            for (int j = 0; j < patternSize2; j++)
            {
                delta = delta + W[neuron][j] * newVec[j];
            }
            newVec[neuron] = (delta > 0) - (delta < 0);
        }
        for (int k = 0; k < patternNumber; k++)
        {
            distMat[iteration][k] = countMismatches(patternSize2, patterns[k], newVec);
        } 
    }
}