using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface ITester
{
    void Init(Main control);
    (Dataset img, float[] outputs) Classify(int index);
    float CalculateAccuracy();
    float FullSetAccuracy();
    int ImageCount();
}
