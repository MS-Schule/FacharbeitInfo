using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IActivation
{
    public float Output(float value);
    public float Derivative(float value);
    public ActivationType Type();
}