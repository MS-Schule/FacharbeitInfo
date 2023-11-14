using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.PlayerLoop;

public interface ITrainer
{
    void Init(Main control);
    IEnumerator ExecuteEpochs(TrainerInfo trainerInfo, Action<bool, TrainerInfo, int, int> FinishFrame, Action DocumentErrorRate);
    (Dataset img, float[] outputs) Classify(int index);
    float CalculateAccuracy();
}
