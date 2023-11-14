using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlTypes;
using System.Linq;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.EventSystems;

public class NeuralNetwork
{
    public Layer[] Layers { get; set;}

    public int LayerNum { get; private set; }

    public Vector2Int DataResolution { get; private set; }
    public ActivationType activType;
    public CostType costType;
    public NeuralNetwork(int[] Layersizes, Vector2Int dataRes, ActivationType activFunction, CostType costFunction)
    {
        LayerNum = Layersizes.Length;
        Layers = new Layer[LayerNum];
        DataResolution = dataRes;
        activType = activFunction;
        costType = costFunction;
        InitiateLayer(Layersizes);
    }

    public void Reset(NetworkMiscs parameters) {
        LayerNum = parameters.layerNum;
        Layers = new Layer[LayerNum];
        DataResolution = parameters.inputSize;
        activType = parameters.activType;
        costType = parameters.costType;
        InitiateLayer(parameters.layerSizes);
    }

    public void Apply(float rate) {
        foreach(var layer in Layers) {
            layer.ApplyGradients(rate, HyperParameters.regularization, HyperParameters.momentum);
        }
    }

    public void Backpropagate(List<Dataset> batchTrainingData)
    {

        System.Threading.Tasks.Parallel.For(0, batchTrainingData.Count(), i => {
            UpdateGradients(batchTrainingData[i]);
        });
    }

    private void UpdateGradients(Dataset input) {
        NetworkLearnData learnData = new(Layers);

        float[] data = MatToVec(input.Data);

        for(int i = 0; i < LayerNum; i++) {
            data = Layers[i].ForwardPass(data, ref learnData.layerData[i]);
        }

        Layers[^1].BackwardPassOutputLayer(ref learnData.layerData[^1], input.Label);

        for(int i = LayerNum - 2; i >= 0; i--) {
            Layers[i].BackwardPassHiddenLayer(ref learnData.layerData[i], learnData.layerData[i+1].localGradient, Layers[i+1].Weights);
        }
    }

    public float[] Compute(float[,] input)
    {
        float[] result = MatToVec(input);

        foreach(var layer in Layers) {
            result = layer.ForwardPass(result);
        }
        return result;
    }

    private void InitiateLayer(int[] LayerSizes)
    {
        int inputSize = DataResolution.x * DataResolution.y;
        for(int i = 0; i < LayerNum; i++) {
            Layers[i] = new Layer(LayerSizes[i], inputSize, activType, costType);
            inputSize = LayerSizes[i];
        }
    }

    private static float[,] VecToMat(float[] vec, Vector2Int matDimension) {
        float[,] result = new float[matDimension.x, matDimension.y];
        
        for(int i = 0; i < matDimension.x; i++) {
            for(int j = 0; j < matDimension.y; j++) {
                result[i,j] = vec[i * matDimension.y + j];
            }
        }
        return result;
    }

    private static float[] MatToVec(float[,] mat) {
        float[] result = new float[mat.Length];
        
        for(int i = 0; i < mat.GetLength(0); i++) {
            for(int j = 0; j < mat.GetLength(1); j++) {
                result[i * mat.GetLength(1) + j] = mat[i, j];
            }
        }
        return result;
    }

    public IEnumerable<int> GetLayerSizes() {
        foreach(var l in Layers)
            yield return l.Size;
    }
}

public class NetworkLearnData
{
	public LayerLearnData[] layerData;

	public NetworkLearnData(Layer[] layers)
	{
		layerData = new LayerLearnData[layers.Length];
		for (int i = 0; i < layers.Length; i++)
		{
			layerData[i] = new LayerLearnData(layers[i]);
		}
	}
}

public class LayerLearnData
{
	public float[] activations;
    public float[] values;
    public float[] inputs;
    public float[] localGradient;

	public LayerLearnData(Layer layer)
	{
		activations = new float[layer.Size];
        values = new float[layer.Size];
        localGradient = new float[layer.Size];
        inputs = new float[layer.InputSize];
	}
}
