using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

public class ClusterSaver : ISaver
{
    public readonly string FolderPath = "Assets/Saves/ClusterSaves/";
    public Clusterer clusterer;


    public ClusterSaver(Clusterer cl) {
        clusterer = cl;
    }

    public void Load(int index) {
        string fileName = string.Format(FolderPath + "cl_centrSave_{0}", index);
        List<float[,]> centroids = new();
        foreach(var c in JsonConvert.DeserializeObject<List<float[,]>>(File.ReadAllText(fileName))) {
            centroids.Add(c);
        }
        clusterer.SetCentroids(centroids);
    }

    public void Save() {
        int index = 0;
        string fileName = string.Format(FolderPath + "cl_centrSave_{0}", index);
        while(File.Exists(fileName)) {
            index++;
            fileName = string.Format(FolderPath + "cl_centrSave_{0}", index);
            Debug.Log("Saved to index: " + index);
        }
        List<float[,]> centroids = new();
        foreach(var c in clusterer.GetCentroids()) {
            centroids.Add(c);
        }
        File.WriteAllText(fileName, JsonConvert.SerializeObject(centroids, Formatting.Indented));
    }
}
