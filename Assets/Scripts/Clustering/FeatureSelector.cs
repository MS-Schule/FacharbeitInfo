using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Mono.CompilerServices.SymbolWriter;
using UnityEngine;
using UnityEngine.UIElements.Experimental;

public static class FeatureSelector
{
    public static float[,] Convolve(float[,] input, float[,] pattern) {
        Vector2Int size = new(input.GetLength(0)-(pattern.GetLength(0)-1), input.GetLength(1)-(pattern.GetLength(1)-1));
        float[,] output = new float[size.x, size.y];

        for(int i = 0; i < size.x; i++) {
            for(int j = 0; j < size.y; j++) {
                output[i,j] = 0;
                for (int m = 0; m < pattern.GetLength(0); m++) {
                    for(int n = 0; n < pattern.GetLength(1); n++) {
                        output[i, j] += pattern[m, n] * input[i + m, j + n];
                    }
                }
                output[i,j] = Mathf.Abs(output[i,j] * .5f);
            }
        }
        return output;//Normalize(output, 0, 1);
    }
}

/*

-----Fields of convolutional layer

public readonly Vector2Int Size;
public readonly Vector2Int InputSize;
private readonly IActivation ActivFunction;

public float[,] value;
public float[,] valueGradient;
public float[,] activation;
public float[,] kernel;
public float[,] kernelGradient;
public float[,] kernelVelocity;

public FeatureSelector(float[,] pattern, Vector2Int inputSize, ActivationType activationType)
{
    InputSize = inputSize;
    Size = new Vector2Int(InputSize.x - (pattern.GetLength(0) - 1), InputSize.y - (pattern.GetLength(1) - 1));
    kernel = pattern;
    kernelGradient = new float[kernel.GetLength(0), kernel.GetLength(1)];
    kernelVelocity = new float[kernel.GetLength(0), kernel.GetLength(1)];
    Debug.Log("Size: " + Size);
    value = new float[Size.x, Size.y];
    valueGradient = new float[Size.x, Size.y];
    activation = new float[Size.x, Size.y];

    ActivFunction = Activation.GetActivationFromType(activationType);
}
*/

/*

---- Methods for convolutional layer

public void BackwardPass(float[,] input, float[,] nextGradient, float[,] nextKernel) {
    for (int m = 0; m < Size.x; m++) {
        for (int n = 0; n < Size.y; n++) {
            float cost = 0;
            float actDeriv = ActivFunction.Derivative(value[m,n]);
            for (int i = 0; i < kernel.GetLength(0); i++) {
                for (int j = 0; j < kernel.GetLength(1); j++) {
                    if(m-i < 0 || n-j < 0 || m-i >= nextGradient.GetLength(0) || n-j >= nextGradient.GetLength(1)) continue;
                    float previousCost = nextGradient[m - i, n - j];
                    float kernelEntry = nextKernel[i, j];
                    cost += previousCost * kernelEntry;
                }
            }
            valueGradient[m,n] = cost * actDeriv;
        }
    }

    for (int i = 0; i < kernel.GetLength(0); i++) {
        for (int j = 0; j < kernel.GetLength(1); j++) {
            for (int m = 0; m < Size.x; m++) {
                for (int n = 0; n < Size.y; n++) {
                    kernelGradient[i, j] += valueGradient[m,n] * input[m + i,n + j];
                }
            }
        }
    }
}

public void BackwardPass(float[,] rawInput, float[] nextGradient, float[][] weights) {
    for (int m = 0; m < kernel.GetLength(0); m++) {
        for (int n = 0; n < kernel.GetLength(1); n++) {
            kernelGradient[m, n] = 0;
            for (int i = 0; i < Size.x; i++) {
                for (int j = 0; j < Size.y; j++) {
                    for (int k = 0; k < nextGradient.Length; k++) {
                        kernelGradient[m, n] += rawInput[m + i, n + j] * nextGradient[k] * weights[k][m * Size.y + n];
                    }
                }
            }
        }
    }

    for (int m = 0; m < Size.x; m++) {
        for (int n = 0; n < Size.y; n++) {
            valueGradient[m, n] = 0; 
            for (int k = 0; k < nextGradient.Length; k++) {
                for (int i = 0; i < kernel.GetLength(0); i++) {
                    for (int j = 0; j < kernel.GetLength(1); j++) {
                        valueGradient[m, n] += nextGradient[k] * weights[k][i * Size.y + j] * ActivFunction.Derivative(rawInput[m + i, n + j]);
                    }
                }
            }
        }
    }
}

public void ApplyGradients(float rate, float regularization, float momentum)
{
    float decay = 1 - regularization * rate;

    for (int n = 0; n < kernel.GetLength(0); n++){
        for (int m = 0; m < kernel.GetLength(1); m++)
        {
            kernel[n,m] *= decay;
            float velW = kernelVelocity[n,m] * momentum - kernelGradient[n,m] * rate;
            kernel[n,m] += velW;
            kernelVelocity[n,m] = velW;
            kernelGradient[n,m] = 0;
        }
    }
}*/
