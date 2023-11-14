using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface ISecUI
{
    void Init(Main main);
    void UpdateScreen(bool isExecuting, TrainerInfo trainerInfo, int epoch = 0, int batch = 0);
    void ToggleActive();
}
