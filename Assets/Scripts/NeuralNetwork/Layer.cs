using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    public float[] Bias;
    public float[] BiasGradient { get; private set; }
    public float[] BiasVelocity { get; private set; }
    public float[][] Weights;
    public float[][] WeightsGradient { get; private set; }
    public float[][] WeightsVelocity { get; private set; }

    public int Size { get; private set; }
    public int InputSize { get; private set; }

    private readonly IActivation ActivFunction;
    private readonly ICost CostFunction;

    public Layer(int size, int inputSize, ActivationType ActivationType, CostType costType)
    {
        Size = size;
        InputSize = inputSize;

        ActivFunction = Activation.GetActivationFromType(ActivationType);
        CostFunction = Cost.GetCostFromType(costType);
        InitiateNodes();
    }

    public float[] ForwardPass(float[] input)
    {
        float[] layerResult = new float[Size];
        for(int index = 0; index < Size; index++) {
            layerResult[index] = Bias[index];
            for (int m = 0; m < InputSize; m++)
            {
                layerResult[index] += Weights[index][m] * input[m];
            }
            layerResult[index] = ActivFunction.Output(layerResult[index]);
        }
        return layerResult;
    }

    public float[] ForwardPass(float[] input, ref LayerLearnData learnData)
    {
        learnData.inputs = input;
        for(int index = 0; index < Size; index++) {
            learnData.values[index] = Bias[index];
            for (int m = 0; m < InputSize; m++)
            {
                learnData.values[index] += Weights[index][m] * input[m];
            }
            learnData.activations[index] = ActivFunction.Output(learnData.values[index]);
        }
        return learnData.activations;
    }

    public void BackwardPassHiddenLayer(ref LayerLearnData learnData, float[] nextLocalGradients, float[][] nextWeights)
    {
        for(int n = 0; n < Size; n++){
            float cost = 0;
            float actDeriv = ActivFunction.Derivative(learnData.values[n]);

            for (int m = 0; m < nextLocalGradients.Length; m++)
            {
                float nextCost = nextLocalGradients[m];
                float weight = nextWeights[m][n];
                cost += nextCost * weight;
            }

            learnData.localGradient[n] = cost * actDeriv;
            BiasGradient[n] += learnData.localGradient[n];

            for (int m = 0; m < InputSize; m++)
            {
                WeightsGradient[n][m] += learnData.localGradient[n] * learnData.inputs[m];
            }
        }
    }
    
    public void BackwardPassOutputLayer(ref LayerLearnData learnData, int desiredOutputs) {
        for(int n = 0; n < Size; n++) {
            float expectedValue = desiredOutputs == n ? 1 : -1;
            float costDeriv = CostFunction.Derivative(learnData.activations[n], expectedValue);
            float activDeriv = ActivFunction.Derivative(learnData.values[n]);
            learnData.localGradient[n] = costDeriv * activDeriv;

            BiasGradient[n] += learnData.localGradient[n];

            for (int m = 0; m < InputSize; m++)
            {
                WeightsGradient[n][m] += learnData.localGradient[n] * learnData.inputs[m];
            }
        }
    }

    public void ApplyGradients(float rate, float regularization, float momentum)
    {
        float weightDecay = 1 - regularization * rate;

        for (int n = 0; n < Size; n++)
        {
            float velB = BiasVelocity[n] * momentum - BiasGradient[n] * rate;
            Bias[n] += velB;
            BiasVelocity[n] = velB;
            BiasGradient[n] = 0;
            for (int m = 0; m < InputSize; m++)
            {
                Weights[n][m] *= weightDecay;
                float velW = WeightsVelocity[n][m] * momentum - WeightsGradient[n][m] * rate;
                Weights[n][m] += velW;
                WeightsVelocity[n][m] = velW;
                WeightsGradient[n][m] = 0;
            }
        }
    }

    private void InitiateNodes()
    {
        Bias = new float[Size];
        BiasGradient = new float[Size];
        BiasVelocity = new float[Size];

        Weights = new float[Size][];
        WeightsGradient = new float[Size][];
        WeightsVelocity = new float[Size][];
        for (int i = 0; i < Size; i++)
        {
            Bias[i] = (UnityEngine.Random.value * .2f) - .1f;
            Weights[i] = new float[InputSize];
            WeightsGradient[i] = new float[InputSize];
            WeightsVelocity[i] = new float[InputSize];
            for (int j = 0; j < InputSize; j++)
            {
                Weights[i][j] = (UnityEngine.Random.value * .6f) - .3f;
                WeightsGradient[i][j] = 0;
                WeightsVelocity[i][j] = 0;
            }
        }
    }
}
