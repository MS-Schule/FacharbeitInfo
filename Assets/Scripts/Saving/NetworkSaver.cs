using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

public struct NetworkMiscs {
    public int layerNum;
    public int[] layerSizes;
    public Vector2Int inputSize;
    public ActivationType activType;
    public CostType costType;
}

public class NetworkSaver : ISaver
{
    public readonly string FolderPath = "Assets/Saves/NetworkSaves/";
    public NeuralNetwork network;

    private readonly string[] fileNames = new string[3] {
        "layer{0}_weights",
        "layer{0}_biases",
        "netw_misc"
    };


    public NetworkSaver(NeuralNetwork nN) {
        network = nN;
    }

    public void Load(int index) {
        string directory = string.Format(FolderPath + "networkSave_{0}/", index);

        string fileName = string.Format(directory + fileNames[2]);
        NetworkMiscs networkParameters = JsonConvert.DeserializeObject<NetworkMiscs>(File.ReadAllText(fileName));

        network.Reset(networkParameters);
        
        for(int i = 0; i < network.LayerNum; i++) {
            fileName = string.Format(directory + fileNames[0], i);
            network.Layers[i].Weights = JsonConvert.DeserializeObject<float[][]>(File.ReadAllText(fileName));

            fileName = string.Format(directory + fileNames[1], i);
            network.Layers[i].Bias = JsonConvert.DeserializeObject<float[]>(File.ReadAllText(fileName));
        }
    }

    public void Save() {
        int index = 0;
        string directory = string.Format(FolderPath + "networkSave_{0}/", index);
        while(Directory.Exists(directory)) {
            index++;
            directory = string.Format(FolderPath + "networkSave_{0}/", index);
            Debug.Log("Saved to index: " + index);
        }
        Directory.CreateDirectory(directory);

        string fileName = string.Format(directory + fileNames[2]);
        NetworkMiscs nMiscs = new() {
            layerNum = network.LayerNum,
            layerSizes = network.GetLayerSizes().ToArray(),
            inputSize = network.DataResolution,
            activType = network.activType,
            costType = network.costType
        };
        File.WriteAllText(fileName, JsonConvert.SerializeObject(nMiscs, Formatting.Indented));

        for(int i = 0; i < network.LayerNum; i++) {
            fileName = string.Format(directory + fileNames[0], i);
            File.WriteAllText(fileName, JsonConvert.SerializeObject(network.Layers[i].Weights, Formatting.Indented));

            fileName = string.Format(directory + fileNames[1], i);
            File.WriteAllText(fileName, JsonConvert.SerializeObject(network.Layers[i].Bias, Formatting.Indented));
        }
    }
}
