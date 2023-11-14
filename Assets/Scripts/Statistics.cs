using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class Statistics
{
    public static void Normalize(ref List<float> data)
    {
        float min = data.Min();
        float max = data.Max();
        data = data.Select((i, index) => ((i - min) / (max - min))).ToList();
    }

    public static float[,] Normalize(float[,] data)
    {
        List<float> values = new();
        for (int i = 0; i < data.GetLength(0); i++)
        {
            for (int j = 0; j < data.GetLength(1); j++)
            {
                values.Add(data[i, j]);
            }
        }

        Normalize(ref values);

        for (int i = 0; i < data.GetLength(0); i++)
        {
            for (int j = 0; j < data.GetLength(1); j++)
            {
                data[i, j] = values[i * data.GetLength(1) + j];
            }
        }
        return data;
    }

    public static void Standardize(ref List<float> data)
    {
        float epsylon = .01f;
        float mean = data.Average();
        float deviation = data.Select(x => Mathf.Pow(x - mean, 2)).Sum();

        deviation /= data.Count - 1;
        deviation = Mathf.Sqrt(deviation + epsylon);
        if (deviation == 0) data = data.Select(x => x = 0).ToList();
        else for (int j = 0; j < data.Count; j++)
            {
                data[j] = (data[j] - mean) / deviation;
            }
    }

    public static float[,] Standardize(float[,] data)
    {
        List<float> values = new();
        for (int i = 0; i < data.GetLength(0); i++)
        {
            for (int j = 0; j < data.GetLength(1); j++)
            {
                values.Add(data[i, j]);
            }
        }

        Standardize(ref values);

        for (int i = 0; i < data.GetLength(0); i++)
        {
            for (int j = 0; j < data.GetLength(1); j++)
            {
                data[i, j] = values[i * data.GetLength(1) + j];
            }
        }
        return data;
    }
}
