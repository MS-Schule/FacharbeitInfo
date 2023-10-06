using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

public class AlphaTrainer : BaseTrainer
{
    public AlphaTrainer(Main control)
    {
        Network = control.neuralNetwork;
        DataDimension = new(7, 1);
        DataSet = SeedsReader.ReadTrainingData().ToArray();
        control.fullBatchSize.text = DataSet.Length.ToString();
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
}

public class MNISTTrainer : BaseTrainer
{
    public MNISTTrainer(Main control)
    {
        Network = control.neuralNetwork;
        DataDimension = new(28, 28);
        DataSet = MnistReader.ReadTrainingData().ToArray();
        control.fullBatchSize.text = DataSet.Length.ToString();
    }
}

public class BaseTrainer
{
    protected Neural_Network Network;
    protected Vector2Int DataDimension;
    protected Dataset[] DataSet;

    public IEnumerator ExecuteEpochs(int epochAmount, int fullBatchSize, int miniBatchSize, Action<bool, int, int, int, int, int> FinishFrame)
    {
        var watch = new System.Diagnostics.Stopwatch();

        long tickBudget = (long)(System.Diagnostics.Stopwatch.Frequency * 12 / 1000f);

        int batchesInEpoch = fullBatchSize / miniBatchSize;
        watch.Restart();
        for (int ep = 0; ep < epochAmount; ep++)
        {
            for (int bat = 0; bat < batchesInEpoch; bat++)
            {
                ComputeMiniBatch(bat, miniBatchSize);
                if (watch.ElapsedTicks > tickBudget)
                {
                    FinishFrame(true, epochAmount, fullBatchSize, miniBatchSize, ep, bat);
                    yield return null;
                    watch.Restart();
                }
                Debug.Log("Time before applying: " + watch.ElapsedTicks);
                foreach (var layer in Network.layers)
                    layer.ApplyGradients(HyperParameters.learnRate / miniBatchSize, HyperParameters.regularization, HyperParameters.momentum);
                Debug.Log("Time after applying: " + watch.ElapsedTicks);
            }
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
        }

        for (int sample = 0; sample < batchSize; sample++)
        {
            List<float> values = new();
            for (int j = 0; j < DataDimension.x; j++)
            {
                for (int k = 0; k < DataDimension.y; k++)
                {
                    values.Add(batchImages[sample].Data[j, k]);
                }
            }
            //Standardize(ref values);
            Network.layers[0].activation = values;

            Network.layers[Network.layerNum - 1].desiredOutputs = Network.layers[Network.layerNum - 1].desiredOutputs.ToList().Select(x => x = -.9f).ToArray();
            Network.layers[Network.layerNum - 1].desiredOutputs[batchImages[sample].Label] = .9f;

            Network.ExecutePropagator();
        }
    }

    private void ComputeMiniBatchNormalized(int batchIndex, int batchSize)
    {
        int startIndex = batchIndex * batchSize;
        int stopIndex = (batchIndex + 1) * batchSize;
        List<Dataset> batchImages = new();
        for (int i = startIndex; i < stopIndex; i++)
        {
            batchImages.Add(DataSet[i]);
        }

        List<float>[] valueBuffer = new List<float>[batchSize];

        // init
        for (int sample = 0; sample < batchSize; sample++)
        {
            List<float> values = new();
            for (int j = 0; j < DataDimension.x; j++)
            {
                for (int k = 0; k < DataDimension.y; k++)
                {
                    values.Add(batchImages[sample].Data[j, k]);
                }
            }
            Standardize(ref values);
            Network.layers[0].activation = values;
            Network.layers[1].IterateNeurons();
            valueBuffer[sample] = ((float[])Network.layers[1].activation.ToArray().Clone()).ToList();
        }

        //possible code for batch normalization
        /*for(int i = 0; i < Network.layers[1].Size; i++) {
            List<float> neuronValuesPerSample = GetDataColumn(valueBuffer, i).ToList();
            Standardize(ref neuronValuesPerSample);
            SetDataColumn(ref valueBuffer, neuronValuesPerSample, i);
        }*/

        // forward propagate
        for (int l = 2; l < Network.layerNum; l++)
        {
            for (int sample = 0; sample < batchSize; sample++)
            {
                for (int n = 0; n < Network.layers[l - 1].Size; n++)
                {
                    Network.layers[l - 1].activation[n] = valueBuffer[sample][n];
                }
                Network.layers[l].IterateNeurons();
                valueBuffer[sample] = ((float[])Network.layers[l].activation.ToArray().Clone()).ToList();
            }
            /*for(int i = 0; i < Network.layers[l].Size; i++) {
                List<float> neuronValuesPerSample = GetDataColumn(valueBuffer, i).ToList();
                Standardize(ref neuronValuesPerSample);
                SetDataColumn(ref valueBuffer, neuronValuesPerSample, i);
            }*/
        }

        // evaluate last layer
        for (int sample = 0; sample < batchSize; sample++)
        {
            for (int n = 0; n < Network.layers[Network.layerNum - 1].Size; n++)
            {
                Network.layers[Network.layerNum - 1].activation[n] = valueBuffer[sample][n];
            }
            Network.layers[Network.layerNum - 1].desiredOutputs = Network.layers[Network.layerNum - 1].desiredOutputs.ToList().Select(x => x = -.9f).ToArray();
            Network.layers[Network.layerNum - 1].desiredOutputs[batchImages[sample].Label] = .9f;
            Network.Backpropagate();
        }
    }

    public Dataset PassSample(int index)
    {
        ComputeMiniBatch(index, 1);
        return DataSet[index];
    }

    private void Normalize(ref List<float> data)
    {
        float min = data.Min();
        float max = data.Max();
        data = data.Select((i, index) => ((i - min) * 2 / (max - min)) - 1).ToList();
    }

    protected void Standardize(ref List<float> data)
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

    public void WriteSample(int index)
    {
        int logFileNo = 1;
        string fileName = String.Format("log{0}_{1}.txt", DataSet[index].Label, logFileNo);

        /*while (File.Exists(fileName))
        {
            logFileNo++;
            fileName = String.Format("log{0}_{1}.txt", DataSet[index].Label, logFileNo);
        }*/
        File.Create("Assets/Data/Training/MNISTTraining/" + fileName).Close();
        using var writer = new StreamWriter("Assets/Data/Training/MNISTTraining/" + fileName);
        Dataset image = DataSet[index];

        List<float> values = new();
        for (int i = 0; i < DataDimension.x; i++)
        {
            for (int j = 0; j < DataDimension.y; j++)
            {
                values.Add(DataSet[index].Data[i, j]);
            }
        }

        Standardize(ref values);

        for (int i = 0; i < DataDimension.x; i++)
        {
            StringBuilder lineBuilder = new();
            for (int j = 0; j < DataDimension.y; j++)
            {
                StringBuilder valueBuilder = new();
                valueBuilder.Append(string.Format("{0:N2}", values[i * DataDimension.y + j]));
                if (valueBuilder[0] != '-') valueBuilder.Insert(0, ' ');
                valueBuilder.Append("    ");
                lineBuilder.Append(valueBuilder.ToString());
            }
            writer.WriteLine(lineBuilder.ToString());
        }
    }

    protected void StandardizeData()
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

                Standardize(ref values);

                for (int k = 0; k < DataSet.Length; k++)
                {
                    DataSet[k].Data[i, j] = values[k];
                }
            }
        }
    }

    private float[] GetDataColumn(List<float>[] data, int columnNumber)
    {
        return Enumerable.Range(0, data.Length)
                .Select(x => data[x][columnNumber])
                .ToArray();
    }

    private float[] GetDataRow(List<float>[] data, int rowNumber)
    {
        return Enumerable.Range(0, data[rowNumber].Count)
                .Select(x => data[rowNumber][x])
                .ToArray();
    }

    private void SetDataColumn(ref List<float>[] data, List<float> inputColumn, int columnNumber)
    {
        for (var i = 0; i < data.Length; i++)
            data[i][columnNumber] = inputColumn[i];
    }

    private void SetDataRow(ref List<float>[] data, List<float> inputRow, int rowNumber)
    {
        data[rowNumber] = inputRow;
    }
}
