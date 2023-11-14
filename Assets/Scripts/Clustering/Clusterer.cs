using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEditor.UIElements;
using UnityEngine;

public class Cluster {
    public List<float[,]> Points { get; set; }
    public float[,] Centroid { get; set; }

    public Cluster(float[,] initialCentroid) {
        Centroid = initialCentroid;
    }

    public void UpdateCentroid() {
        if(Points.Count == 0) return;
        Vector2Int DataSize = new(Centroid.GetLength(0), Centroid.GetLength(1));
        float[,] averagePoint = new float[DataSize.x,DataSize.y];

        for(int i = 0; i < Points.Count; i++) {
            for(int x = 0; x < DataSize.x; x++) {
                for(int y = 0; y < DataSize.y; y++) {
                    averagePoint[x,y] += Points[i][x,y];
                }
            }
        }

        for(int x = 0; x < DataSize.x; x++) {
            for(int y = 0; y < DataSize.y; y++) {
                averagePoint[x,y] /= Points.Count;
            }
        }

        Centroid = averagePoint;
        Points = new();
    }

    public float DistanceL1(float[,] point) {
        float result = 0;
        for(int i = 0; i < point.GetLength(0); i++) {
            for(int j = 0; j < point.GetLength(1); j++) {
                result += point[i,j] - Centroid[i,j];   // L1 Norm
            }
        }
        return result;
    }

    public float DistanceSoS(float[,] point) {
        float result = 0;
        for(int i = 0; i < point.GetLength(0); i++) {
            for(int j = 0; j < point.GetLength(1); j++) {
                result += Mathf.Pow(point[i,j] - Centroid[i,j], 2);   // Sum-of-squares distance
            }
        }
        return result;
    }

    public float DistanceL2(float[,] point) {
        float result = 0;
        for(int i = 0; i < point.GetLength(0); i++) {
            for(int j = 0; j < point.GetLength(1); j++) {
                result += Mathf.Pow(point[i,j] - Centroid[i,j], 2);   // L2 Norm
            }
        }
        return Mathf.Sqrt(result);
    }
}

public class Clusterer
{
    public List<Cluster> clusters;
    public readonly int clusterAmount;
    public readonly Vector2Int DataSize;

    public Clusterer(int K, Vector2Int dataDimension, List<float[,]> samplesKPP) {
        clusterAmount = K;
        clusters = new();
        DataSize = dataDimension;
        InitializeClusters(samplesKPP);
    }

    public int AssignToCluster(float[,] input) {
        float min = float.MaxValue;
        int nearestCluster = 0;
        for(int i = 0; i < clusterAmount; i++) {
            float dist = clusters[i].DistanceSoS(input);
            if(dist < min) {
                nearestCluster = i;
                min = dist;
            }
        }
        clusters[nearestCluster].Points.Add(input);
        return nearestCluster;
    }

    public void UpdateClusterCentroids() {
        Parallel.ForEach(clusters, cluster => {
            cluster.UpdateCentroid();
        });
    }

    public IEnumerable<float[,]> GetCentroids() {
        foreach(var c in clusters) 
            yield return c.Centroid;
    }

    public void SetCentroids(List<float[,]> centroids) {
        for(int i = 0; i < centroids.Count; i++) {
            clusters[i].Centroid = centroids[i];
        }
    }

    private void InitializeClusters(List<float[,]> samplePoints) {  // k-means++
        for(int i = 0; i < clusterAmount; i++) {
            int maxIndex = 0;
            float maxVal = float.MinValue;
            for(int sample = 0; sample < samplePoints.Count; sample++) {
                float distSum = 0;
                for(int c = 0; c < i; c++) {
                    distSum += clusters[c].DistanceL2(samplePoints[sample]);
                }
                if(distSum > maxVal) {
                    maxVal = distSum;
                    maxIndex = sample;
                }
            }
            clusters.Add(new Cluster(samplePoints[maxIndex]));
            clusters.Last().Points = new();
        }
    }
}
