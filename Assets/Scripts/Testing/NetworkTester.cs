using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


public class NetworkTester : ITester
{
    protected NeuralNetwork network;
    protected Vector2Int DataDimension;
    protected Dataset[] TestSet;

    public NetworkTester() {
        DataDimension = new(28, 28);
        TestSet = MnistReader.ReadTestData().ToArray();
    }

    public void Init(Main control) {
        network = control.neuralNetwork;
    }

    public (Dataset img, float[] outputs) Classify(int index) {
        float[,] inputs = Statistics.Standardize(TestSet[index].Data);

        float[] output = network.Compute(inputs);

        return (TestSet[index], output);
    }

    public float CalculateAccuracy() {
        System.Diagnostics.Stopwatch watch = new();
        watch.Start();
        float result = 0;
        int testCount = 1000;
        
        for(int i = 0; i < testCount; i++) {
            int index = Random.Range(0, TestSet.Count());
            float[] output = network.Compute(TestSet[index].Data);
            result += TestSet[index].Label == output.ToList().IndexOf(output.Max()) ? 1 : 0;
        }

        Debug.Log("Test Acc: " + result / testCount + "  Calculation Time: " + watch.ElapsedMilliseconds + "ms");
        return result / testCount;
    }

    public float FullSetAccuracy() {
        System.Diagnostics.Stopwatch watch = new();
        watch.Start();
        float result = 0;
        
        for(int i = 0; i < TestSet.Count(); i++) {
            float[] output = network.Compute(TestSet[i].Data);
            result += TestSet[i].Label == output.ToList().IndexOf(output.Max()) ? 1 : 0;
        }

        Debug.Log("Test Acc: " + result / TestSet.Count() + "  Calculation Time: " + watch.ElapsedMilliseconds + "ms");
        return result / TestSet.Count();
    }

    public int ImageCount() {
        return TestSet.Count();
    }
}
