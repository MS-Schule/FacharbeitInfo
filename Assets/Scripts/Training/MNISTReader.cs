using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

//https://stackoverflow.com/questions/49407772/reading-mnist-database
public static class MnistReader
{
    private const string TrainImages = "Assets/Data/Training/MNISTTraining/train-images.idx3-ubyte";
    private const string TrainLabels = "Assets/Data/Training/MNISTTraining/train-labels.idx1-ubyte";
    //private const string TestImages = "mnist/t10k-images.idx3-ubyte";
    //private const string TestLabels = "mnist/t10k-labels.idx1-ubyte";

    public static IEnumerable<Dataset> ReadTrainingData()
    {
        foreach (var item in Read(TrainImages, TrainLabels))
        {
            yield return item;
        }
    }

    //public static IEnumerable<Image> ReadTestData()
    //{
    //    foreach (var item in Read(TestImages, TestLabels))
    //    {
    //        yield return item;
    //    }
    //}

    private static IEnumerable<Dataset> Read(string imagesPath, string labelsPath)
    {
        BinaryReader labels = new(new FileStream(labelsPath, FileMode.Open));
        BinaryReader images = new(new FileStream(imagesPath, FileMode.Open));

        int magicNumber = images.ReadBigInt32();
        int numberOfImages = images.ReadBigInt32();
        int width = images.ReadBigInt32();
        int height = images.ReadBigInt32();

        int magicLabel = labels.ReadBigInt32();
        int numberOfLabels = labels.ReadBigInt32();

        for (int i = 0; i < numberOfImages; i++)
        {
            var bytes = images.ReadBytes(width * height);
            var arr = new float[height, width];

            for (int j = 0; j < arr.GetLength(0); j++)
            {
                for (int k = 0; k < arr.GetLength(1); k++)
                {
                    arr[j, k] = bytes[j * height + k];
                }
            }

            yield return new Dataset()
            {
                Data = arr,
                Label = labels.ReadByte()
            };
        }
    }
}

public class Dataset
{
    public byte Label { get; set; }
    public float[,] Data { get; set; }
}

public static class Extensions
{
    public static int ReadBigInt32(this BinaryReader br)
    {
        var bytes = br.ReadBytes(sizeof(int));
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    public static void ForEach<T>(this T[,] source, Action<int, int> action)
    {
        for (int w = 0; w < source.GetLength(0); w++)
        {
            for (int h = 0; h < source.GetLength(1); h++)
            {
                action(w, h);
            }
        }
    }
}
