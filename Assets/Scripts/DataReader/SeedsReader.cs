using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public static class SeedsReader
{
    private const string TrainData = "Assets/Data/Training/AlphaTraining/seeds_dataset.txt";
    public static IEnumerable<Dataset> ReadTrainingData()
    {
        foreach (var item in Read(TrainData))
        {
            yield return item;
        }
    }

    private static IEnumerable<Dataset> Read(string dataPath)
    {
        string[] rawDataSet;
        rawDataSet = File.ReadAllLines(dataPath);

        for (int i = 0; i < rawDataSet.Length; i++)
        {
            string line = rawDataSet[i];
            string[] lineContents = line.Split("	", System.StringSplitOptions.RemoveEmptyEntries);
            float[,] values = new float[lineContents.Length, 1];
            for (int j = 0; j < lineContents.Length - 1; j++)
            {
                values[j, 0] = float.Parse(lineContents[j], System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
            }
            byte label = (byte)(int.Parse(lineContents[7], System.Globalization.CultureInfo.InvariantCulture.NumberFormat) - 1);

            yield return new Dataset()
            {
                Data = values,
                Label = label
            };
        }
    }
}
