using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;

public static class LeakyReLU
{
    private const float A = 0.1f;
    public static float Output(float value)
    {
        return value >= 0 ? value : A * value;
    }

    public static float Derivative(float value)
    {
        return value >= 0 ? 1 : A;
    }
}

public static class Tanh
{
    public static float Output(float value)
    {
        return math.tanh(value);
    }

    public static float Derivative(float value)
    {
        return 1 - math.pow(math.tanh(value), 2);
    }
}

public static class LossFunc
{
    public static float Output(float a, float z)
    {
        return math.pow(z - a, 2) / 2;
    }
    public static float Derivative(float a, float z)
    {
        return a - z;
    }
}

public enum LayerType
{
    inner,
    semiConnected,
    input,
    output
};

public class Layer
{
    public int chunkSize;
    public float[] desiredOutputs;
    public List<float> value;
    public List<float> activation;
    public List<float> valueGradient;   // temp per sample
    public List<float> bias;
    public List<float> biasGradient;    // remains for one mini-batch
    public List<float> biasVelocity;    // adds momentum to next mini-batch
    public List<float[]> weights;
    public List<float[]> weightsGradient; // remains for one mini-batch
    public List<float[]> weightsVelocity; // adds momentum to next mini-batch

    public int Size { get; protected set; }
    public LayerType LayerType { get; protected set; }
    public Layer PreviousLayer { get; protected set; }
    public ComputeShader LayerShader { get; protected set; }

    public virtual void IterateNeurons()
    {
        System.Threading.Tasks.Parallel.For(0, Size,
            index => {
                value[index] = bias[index];
                for (int m = 0; m < PreviousLayer.Size; m++)
                {
                    value[index] += weights[index][m] * PreviousLayer.activation[m];
                }
                activation[index] = Tanh.Output(value[index]);
            }
        );
    }

    public virtual void CalculateGradient(Layer nextLayer = null)
    {
        for (int n = 0; n < Size; n++)
        {
            float cost = 0;
            int M = nextLayer.Size;
            float actDeriv = Tanh.Derivative(value[n]);

            for (int m = 0; m < M; m++)
            {
                float previousCost = nextLayer.valueGradient[m];
                float weight = nextLayer.weights[m][n];
                cost += previousCost * weight;
            }

            valueGradient[n] = cost * actDeriv;
            biasGradient[n] += valueGradient[n];
            for (int m = 0; m < PreviousLayer.Size; m++)
            {
                weightsGradient[n][m] += valueGradient[n] * PreviousLayer.activation[m];
            }
        }
    }

    public virtual void ApplyGradients(float rate, float regularization, float momentum)
    {
        float weightDecay = 1 - regularization * rate;

        for (int n = 0; n < Size; n++)
        {
            float velB = biasVelocity[n] * momentum - biasGradient[n] * rate;
            bias[n] += velB;
            biasGradient[n] = 0;
            for (int m = 0; m < PreviousLayer.Size; m++)
            {
                weights[n][m] *= weightDecay;
                float velW = weightsVelocity[n][m] * momentum - weightsGradient[n][m] * rate;
                weights[n][m] += velW;
                weightsGradient[n][m] = 0;
            }
        }
    }

    protected virtual void InitiateNodes()
    {
        for (int i = 0; i < Size; i++)
        {
            value.Add(0);
            valueGradient.Add(0);
            activation.Add(0);
            bias.Add((UnityEngine.Random.value * .2f) - .1f);
            biasGradient.Add(0);
            biasVelocity.Add(0);
            weights.Add(new float[PreviousLayer.Size]);
            weightsGradient.Add(new float[PreviousLayer.Size]);
            weightsVelocity.Add(new float[PreviousLayer.Size]);
            for (int j = 0; j < PreviousLayer.Size; j++)
            {
                weights[i][j] = (UnityEngine.Random.value * 2f) - 1f;
                weightsGradient[i][j] = 0;
                weightsVelocity[i][j] = 0;
            }
        }
    }
}

public class InputLayer : Layer
{
    public InputLayer(int size)
    {
        Size = size;
        chunkSize = size / 16;
        activation = new();
        LayerType = LayerType.input;
        InitiateNodes();
    }

    public override void CalculateGradient(Layer nextLayer)
    {
        // do nothing
    }

    public override void IterateNeurons()
    {
        // do nothing
    }

    public override void ApplyGradients(float rate, float regularization, float momentum)
    {
        // do nothing
    }

    protected override void InitiateNodes()
    {
        for (int i = 0; i < Size; i++)
        {
            activation.Add(0);
        }
    }
}

public class SemiHiddenLayer : Layer
{

    public SemiHiddenLayer(int size, Layer prevLayer)
    {
        Size = size;
        value = new();
        valueGradient = new();
        activation = new();
        bias = new();
        biasGradient = new();
        biasVelocity = new();
        LayerType = LayerType.semiConnected;
        PreviousLayer = prevLayer;
        weights = new List<float[]>();
        weightsGradient = new List<float[]>();
        weightsVelocity = new List<float[]>();
        InitiateNodes();
    }

    public override void ApplyGradients(float rate, float regularization, float momentum)
    {
        float weightDecay = 1 - regularization * rate;

        for (int n = 0; n < Size; n++)
        {
            float velB = biasVelocity[n] * momentum - biasGradient[n] * rate;
            bias[n] += velB;
            biasGradient[n] = 0;
            for (int m = 0; m < PreviousLayer.chunkSize; m++)
            {
                weights[n][m] *= weightDecay;
                float velW = weightsVelocity[n][m] * momentum - weightsGradient[n][m] * rate;
                weights[n][m] += velW;
                weightsGradient[n][m] = 0;
            }
        }
    }

    protected override void InitiateNodes()
    {
        for (int i = 0; i < Size; i++)
        {
            value.Add(0);
            valueGradient.Add(0);
            activation.Add(0);
            bias.Add((UnityEngine.Random.value * .2f) - .1f);
            biasGradient.Add(0);
            biasVelocity.Add(0);
            weights.Add(new float[PreviousLayer.chunkSize]);
            weightsGradient.Add(new float[PreviousLayer.chunkSize]);
            weightsVelocity.Add(new float[PreviousLayer.Size]);
            for (int j = 0; j < PreviousLayer.chunkSize; j++)
            {
                weights[i][j] = (UnityEngine.Random.value * 2f) - 1f;
                weightsGradient[i][j] = 0;
                weightsVelocity[i][j] = 0;
            }
        }
    }

    public override void IterateNeurons()
    {
        for (int n = 0; n < Size; n++)
        {
            value[n] = bias[n];
            for (int m = 0; m < PreviousLayer.chunkSize; m++)
            {
                value[n] += weights[n][m] * PreviousLayer.activation[n * PreviousLayer.chunkSize + m];
            }
            activation[n] = Tanh.Output(value[n]);
        }
    }

    public override void CalculateGradient(Layer nextLayer = null)
    {
        for (int n = 0; n < Size; n++)
        {
            float cost = 0;
            int M = nextLayer.Size;
            float actDeriv = Tanh.Derivative(value[n]);

            for (int m = 0; m < M; m++)
            {
                float previousCost = nextLayer.valueGradient[m];
                float weight = nextLayer.weights[m][n];
                cost += previousCost * weight;
            }

            valueGradient[n] = cost * actDeriv;
            biasGradient[n] += valueGradient[n];
            for (int m = 0; m < PreviousLayer.chunkSize; m++)
            {
                weightsGradient[n][m] += valueGradient[n] * PreviousLayer.activation[n * PreviousLayer.chunkSize + m];
            }
        }
    }
}

public class HiddenLayer : Layer
{

    public HiddenLayer(int size, Layer prevLayer, ComputeShader shader)
    {
        Size = size;
        value = new();
        activation = new();
        bias = new();
        biasGradient = new();
        biasVelocity = new();
        valueGradient = new();
        LayerType = LayerType.inner;
        LayerShader = shader;
        PreviousLayer = prevLayer;
        weights = new List<float[]>();
        weightsGradient = new List<float[]>();
        weightsVelocity = new List<float[]>();
        InitiateNodes();
    }
}

public class OutputLayer : Layer
{

    public OutputLayer(int size, Layer prevLayer)
    {
        Size = size;
        desiredOutputs = new float[size];
        value = new();
        activation = new();
        bias = new();
        biasGradient = new();
        biasVelocity = new();
        valueGradient = new();
        LayerType = LayerType.output;
        PreviousLayer = prevLayer;
        weights = new List<float[]>();
        weightsGradient = new List<float[]>();
        weightsVelocity = new List<float[]>();
        InitiateNodes();
    }

    public override void CalculateGradient(Layer nextLayer = null)
    {
        for (int n = 0; n < Size; n++)
        {
            valueGradient[n] = LossFunc.Derivative(activation[n], desiredOutputs[n]) * Tanh.Derivative(value[n]);
            biasGradient[n] += valueGradient[n];
            for (int m = 0; m < PreviousLayer.Size; m++)
            {
                weightsGradient[n][m] += valueGradient[n] * PreviousLayer.activation[m];
            }
        }
    }
}
