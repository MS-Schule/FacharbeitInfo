using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public enum CostType {
    MeanSquareError,
    CrossEntropy
}

public readonly struct Cost
{

    public static ICost GetCostFromType(CostType type)
	{
		switch (type)
		{
			case CostType.MeanSquareError:
				return new MeanSquaredError();
			case CostType.CrossEntropy:
				return new CrossEntropy();
			default:
				UnityEngine.Debug.LogError("Undefined cost function");
				return new MeanSquaredError();
		}
	}

    public class MeanSquaredError : ICost
	{
		public float Output(float value, float expectedValue)
		{
			return 0.5f * math.pow(value - expectedValue, 2);
		}

		public float Derivative(float value, float expectedValue)
		{
			return value - expectedValue;
		}

		public CostType Type()
		{
			return CostType.MeanSquareError;
		}
	}

	public class CrossEntropy : ICost
	{
		// Note: expected outputs are expected to all be either 0 or 1
		public float Output(float value, float expectedValue)
		{
            float x = value;
            float y = expectedValue;
            float v = (y == 1) ? -Mathf.Log(x) : -Mathf.Log(1 - x);
			return float.IsNaN(v) ? 0 : v;;
		}

		public float Derivative(float value, float expectedValue)
		{
			float x = value;
            float y = expectedValue;
			if (x == -1 || x == 1)
			{
				return 0;
			}
			return (-x + y) / (x * (x - 1));
		}

		public CostType Type()
		{
			return CostType.CrossEntropy;
		}
	}
}
