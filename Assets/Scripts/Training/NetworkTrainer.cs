using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

public class NetworkSeedTrainer : BaseNetworkTrainer
{
    public NetworkSeedTrainer(Main control)
    {
        DataDimension = new(7, 1);
        DataSet = SeedsReader.ReadTrainingData().ToArray();
        control.ui.fullBatchSize.text = DataSet.Length.ToString();
        StandardizeData();
        WriteConvertedData(DataSet);
    }

    private void WriteConvertedData(Dataset[] dataSet)
    {
        string fileName = string.Format("seeds_datasetCONVERTED.txt");

        File.Create("Assets/Data/Training/AlphaTraining/" + fileName).Close();
        using var writer = new StreamWriter("Assets/Data/Training/AlphaTraining/" + fileName);
        for (int img = 0; img < dataSet.Length; img++)
        {
            Dataset image = dataSet[img];

            List<float> values = new();
            for (int i = 0; i < DataDimension.x; i++)
            {
                values.Add(dataSet[img].Data[i, 0]);
            }

            StringBuilder lineBuilder = new();
            for (int j = 0; j < DataDimension.x; j++)
            {
                StringBuilder valueBuilder = new();
                valueBuilder.Append(string.Format("{0:N2}", values[j]));
                if (valueBuilder[0] != '-') valueBuilder.Insert(0, ' ');
                valueBuilder.Append("    ");
                lineBuilder.Append(valueBuilder.ToString());
            }
            writer.WriteLine(lineBuilder.ToString());
        }
    }

    private void StandardizeData()
    {
        for (int i = 0; i < DataDimension.x; i++)
        {
            for (int j = 0; j < DataDimension.y; j++)
            {
                List<float> values = new();
                for (int k = 0; k < DataSet.Length; k++)
                {
                    values.Add(DataSet[k].Data[i, j]);
                }

                Statistics.Standardize(ref values);

                for (int k = 0; k < DataSet.Length; k++)
                {
                    DataSet[k].Data[i, j] = values[k];
                }
            }
        }
    }
}

public class NetworkMNISTTrainer : BaseNetworkTrainer
{
    public NetworkMNISTTrainer(Main control)
    {
        DataDimension = new(28, 28);
        DataSet = MnistReader.ReadTrainingData().ToArray();
        control.ui.fullBatchSize.text = DataSet.Length.ToString();
    }
}

public class BaseNetworkTrainer : ITrainer
{
    protected NeuralNetwork network;
    protected Vector2Int DataDimension;
    protected Dataset[] DataSet;
    protected NetworkTester tester;

    public void Init(Main control) {
        network = control.neuralNetwork;
    }

    public IEnumerator ExecuteEpochs(TrainerInfo info, Action<bool, TrainerInfo, int, int> FinishFrame, Action DocumentErrorRate)
    {
        var watch = new System.Diagnostics.Stopwatch();

        long tickBudget = (long)(System.Diagnostics.Stopwatch.Frequency * 12 / 1000f);

        watch.Restart();
        for (int ep = 0; ep < info.epochAmount; ep++)
        {
            for (int bat = 0; bat < info.batchAmount; bat++)
            {
                ComputeMiniBatch(bat, info.miniBatchSize);
                if (watch.ElapsedTicks > tickBudget)
                {
                    FinishFrame(true, info, ep, bat);
                    yield return null;
                    watch.Restart();
                }
                network.Apply(HyperParameters.learnRate / info.miniBatchSize);
                if (watch.ElapsedTicks > tickBudget)
                {
                    FinishFrame(true, info, ep, bat);
                    yield return null;
                    watch.Restart();
                }
            }
            DocumentErrorRate();
        }
    }

    private void ComputeMiniBatch(int batchIndex, int batchSize)
    {
        int startIndex = batchIndex * batchSize;
        int stopIndex = (batchIndex + 1) * batchSize;
        List<Dataset> batchImages = new();
        for (int i = startIndex; i < stopIndex; i++)
        {
            batchImages.Add(DataSet[i]);
            batchImages.Last().Data = Statistics.Standardize(batchImages.Last().Data);
        }
        
        network.Backpropagate(batchImages);
    }

    public (Dataset img, float[] outputs) Classify(int index)
    {
        float[,] inputs = Statistics.Standardize(DataSet[index].Data);

        float[] output = network.Compute(inputs);

        return (DataSet[index], output);
    }

    public float CalculateAccuracy() {
        System.Diagnostics.Stopwatch watch = new();
        watch.Start();
        float result = 0;
        int testCount = 1000;
        
        for(int i = 0; i < testCount; i++) {
            int index = UnityEngine.Random.Range(0, DataSet.Count());
            float[] output = network.Compute(DataSet[index].Data);
            result += DataSet[index].Label == output.ToList().IndexOf(output.Max()) ? 1 : 0;
        }

        Debug.Log("Train Acc: " + result / testCount + "  Calculation Time: " + watch.ElapsedMilliseconds);
        return result / testCount;
    }
}