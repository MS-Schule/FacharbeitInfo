using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public enum ActivationType {
    Sigmoid,
    TanH,
    ReLU,
    SiLU,
    LeakyReLU
}

public readonly struct Activation
{

    public static IActivation GetActivationFromType(ActivationType type) {
        switch(type) {
            case ActivationType.Sigmoid:
                return new Sigmoid();
            case ActivationType.TanH:
                return new TanH();
            case ActivationType.ReLU:
                return new ReLU();
            case ActivationType.SiLU:
                return new SiLU();
            case ActivationType.LeakyReLU:
                return new LeakyReLU();
            default:
                Debug.LogError("Undefined activation function");
                return new TanH();
        }
    }

    public readonly struct Sigmoid : IActivation {
        public float Output(float value) {
            return 1.0f / (1 + Mathf.Exp(-value));
        }

        public float Derivative(float value) {
            float a = Output(value);
            return a * (1 - a);
        }

        public ActivationType Type() {
            return ActivationType.Sigmoid;
        }
    }

    public readonly struct TanH : IActivation {
        public float Output(float value) {
            return math.tanh(value);
        }

        public float Derivative(float value) {
            float a = Output(value);
            return 1 - a * a;
        }

        public ActivationType Type() {
            return ActivationType.TanH;
        }
    }

    public readonly struct ReLU : IActivation {
        public float Output(float value) {
            return Mathf.Max(0, value);
        }

        public float Derivative(float value) {
            return value > 0 ? 1 : 0;
        }

        public ActivationType Type() {
            return ActivationType.ReLU;
        }
    }

    public readonly struct SiLU : IActivation {
        public float Output(float value) {
            return value / (1 + Mathf.Exp(-value));
        }

        public float Derivative(float value) {
            float sig = 1.0f / (1 + Mathf.Exp(-value));
            return value * sig * (1 - sig) + sig;
        }

        public ActivationType Type() {
            return ActivationType.SiLU;
        }
    }

    public readonly struct LeakyReLU : IActivation {
        private const float m = 0.2f;
        public float Output(float value) {
            return Mathf.Max(m * value, value);
        }

        public float Derivative(float value) {
            return value > 0 ? 1 : m;
        }

        public ActivationType Type() {
            return ActivationType.LeakyReLU;
        }
    }
}
