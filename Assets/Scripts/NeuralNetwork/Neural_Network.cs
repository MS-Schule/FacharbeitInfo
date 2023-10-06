using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neural_Network
{
    public Layer[] layers;

    public int layerNum;
    private readonly ComputeShader shader;


    public Neural_Network(ComputeShader shader)
    {
        layerNum = 2;
        layers = new Layer[layerNum];
        this.shader = shader;
        InitiateLayer(new int[] {2, 2});
    }

    public Neural_Network(int[] layerSizes, ComputeShader shader)
    {
        layerNum = layerSizes.Length;
        layers = new Layer[layerNum];
        this.shader = shader;
        InitiateLayer(layerSizes);
    }

    public void ExecutePropagator()
    {
        var watch = new System.Diagnostics.Stopwatch();
        watch.Start();
        Compute();
        Debug.Log("Time needed for forwardpropagation: " + watch.ElapsedTicks);
        watch.Restart();
        Backpropagate();
        Debug.Log("Time needed for backpropagation: " + watch.ElapsedTicks);
    }

    public void Backpropagate()
    {
        GetOutputLayer().CalculateGradient();
        for (int i = layerNum - 2; i > 0; i--)
        {
            layers[i].CalculateGradient(layers[i + 1]);
        }
    }

    public void Compute()
    {
        foreach (var layer in layers)
            layer.IterateNeurons();
    }

    private void InitiateLayer(int[] layerSizes)
    {
        layers[0] = new InputLayer(layerSizes[0]);
        for (int i = 1; i < layerNum - 1; i++)
        {
            layers[i] = new HiddenLayer(layerSizes[i], layers[i - 1], shader);
        }
        layers[^1] = new OutputLayer(layerSizes[^1], layers[^2]);
    }

    public Layer GetOutputLayer()
    {
        return layers[^1];
    }
}
