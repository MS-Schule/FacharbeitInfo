using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class ClusterTrainer : ITrainer
{
    protected Clusterer clusterer;
    protected Vector2Int DataDimension;
    protected Dataset[] DataSet;

    public ClusterTrainer(Main control)
    {
        DataDimension = new(28, 28);
        DataSet = MnistReader.ReadTrainingData().ToArray();
        control.ui.fullBatchSize.text = DataSet.Length.ToString();
    }

    public void Init(Main control) {
        clusterer = control.clusterer;
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
                int startIndex = bat * info.miniBatchSize;
                int stopIndex = (bat + 1) * info.miniBatchSize;
                List<Dataset> batchImages = new();
                for (int i = startIndex; i < stopIndex; i++)
                {
                    batchImages.Add(DataSet[i]);
                }

                for (int sample = 0; sample < info.miniBatchSize; sample++)
                {
                    float[,] inputs = Statistics.Normalize(batchImages[sample].Data);

                    clusterer.AssignToCluster(inputs);
                    if (watch.ElapsedTicks > tickBudget)
                    {
                        FinishFrame(true, info, ep, bat);
                        yield return null;
                        watch.Restart();
                    }
                }
                clusterer.UpdateClusterCentroids();
            }
            DocumentErrorRate();
        }
    }

    public void FeedMiniBatch(int batchIndex, int batchSize)
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
            float[,] inputs = Statistics.Normalize(batchImages[sample].Data);

            clusterer.AssignToCluster(inputs);
        }
    }

    public (Dataset img, float[] outputs) Classify(int index) {
        float[,] inputs = Statistics.Normalize(DataSet[index].Data);

        int cluster = clusterer.AssignToCluster(inputs);

        return (DataSet[index], new float[1] {cluster});
    }

    public IEnumerable<float[,]> GetRndSamples(int amount) {
        for(int i = 0; i < amount; i++) {
            int index = Mathf.RoundToInt(UnityEngine.Random.Range(0, DataSet.Count()));
            float[,] sample = new float[DataDimension.x, DataDimension.y];
            for(int x = 0; x < DataDimension.x; x++) {
                for(int y = 0; y < DataDimension.y; y++) {
                    sample[x,y] = DataSet[index].Data[x,y] / 255f;
                }
            }
            yield return sample;
        }
    }

    public float CalculateAccuracy() {  //Not implemented
        return 1;
    }
}

/*  ----Code for edge detection------
public static float[,] DetectEdges(float[,] input) {
    float[,] sobelFilterHorizontal = new float[3, 3] {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };

    float[,] sobelFilterVertical = new float[3, 3] {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    float[,] sobelFilterHorizontal = new float[2, 2] {
        {-1,  1},
        {-1,  1}
    };

    float[,] sobelFilterVertical = new float[2, 2] {
        { 1,  1},
        {-1, -1}
    };

    float[][,] processedData = new float[3][,];

    processedData[0] = FeatureSelector.Convolve(input, sobelFilterHorizontal);  //all vertical edges
    processedData[1] = FeatureSelector.Convolve(input, sobelFilterVertical);    //all horizontal edges

    processedData[2] = new float[processedData[0].GetLength(0),processedData[0].GetLength(1)];

    for(int i = 0; i < processedData[0].GetLength(0); i++) {
        for(int j = 0; j < processedData[0].GetLength(1); j++) {
            processedData[2][i,j] = processedData[0][i,j] + processedData[1][i,j];  //Combined edges
        }
    }

    return processedData[2];
}
*/
