using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface ICost
{
    public float Output(float value, float expectedVal);
    public float Derivative(float value, float expectedVal);
    public CostType Type();
}
