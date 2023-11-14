using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class ClusterTester : ITester
{
    protected Clusterer clusterer;
    protected Vector2Int DataDimension;
    protected Dataset[] TestSet;

    public ClusterTester() {
        DataDimension = new(28, 28);
        TestSet = MnistReader.ReadTestData().ToArray();
    }

    public void Init(Main control) {
        clusterer = control.clusterer;
    }

    public (Dataset img, float[] outputs) Classify(int index) {
        float[,] inputs = Statistics.Normalize(TestSet[index].Data);

        int cluster = clusterer.AssignToCluster(inputs);

        return (TestSet[index], new float[1] {cluster});
    }

    public float CalculateAccuracy() {  //Not implemented
        return .5f;
    }

    public float FullSetAccuracy() {  //Not implemented
        return .5f;
    }

    public int ImageCount() {
        return TestSet.Count();
    }
}
