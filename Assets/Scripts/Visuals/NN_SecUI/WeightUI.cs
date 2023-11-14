using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

public class WeightUI : MonoBehaviour
{
    private Text nameLabel;
    private Text strengthLabel;
    private Text gradientLabel;
    private Text velocityLabel;
    private Scrollbar strengthSlider;
    private int layerIndex;
    private int outputIndex;
    private int inputIndex;
    private Button closeButton;
    public Main controller;
    private NeuralNetwork network;
    private WeightInspector currentInspector;
    void Start()
    {
        network = controller.neuralNetwork;
        gameObject.SetActive(false);
        layerIndex = 0;
        outputIndex = 0;
        inputIndex = 0;
        closeButton = gameObject.transform.GetChild(6).GetComponent<Button>();
        closeButton.onClick.AddListener(delegate { SetActive(false, null); });
        nameLabel = gameObject.transform.GetChild(1).GetComponent<Text>();
        strengthLabel = gameObject.transform.GetChild(2).GetComponent<Text>();
        gradientLabel = gameObject.transform.GetChild(3).GetComponent<Text>();
        velocityLabel = gameObject.transform.GetChild(4).GetComponent<Text>();
        strengthSlider = gameObject.transform.GetChild(5).GetComponent<Scrollbar>();
        strengthSlider.onValueChanged.AddListener(delegate { UpdateValue(); });
    }

    public void UpdateTable(int lIndex, int outIndex, int inIndex)
    {
        layerIndex = lIndex - 1;
        outputIndex = outIndex - 1;
        inputIndex = inIndex - 1;
        nameLabel.text = "Weight [" + lIndex + "," + outIndex + "] - [" + (lIndex - 1) + "," + inIndex + "]";
        UpdateTable();
    }

    public void UpdateTable()
    {
        float strength = network.Layers[layerIndex].Weights[outputIndex][inputIndex];
        float gradient = network.Layers[layerIndex].WeightsGradient[outputIndex][inputIndex];
        float velocity = network.Layers[layerIndex].WeightsVelocity[outputIndex][inputIndex];
        strengthLabel.text = "Strength: " + strength;
        gradientLabel.text = "Gradient: <color="
            + (-gradient > 0 ? "green>" : "red>")
            + -gradient
            + "</color>"; ;

        velocityLabel.text = "Velocity: <color="
            + (velocity > 0 ? "green>" : "red>")
            + velocity + "</color>";
    }

    private void UpdateValue()
    {
        network.Layers[layerIndex].Weights[outputIndex][inputIndex] = (strengthSlider.value * 2) - 1;
        strengthLabel.text = "Strength: " + network.Layers[layerIndex].Weights[outputIndex][inputIndex];
    }

    public void SetActive(bool active, WeightInspector WeightInspector)
    {
        if (gameObject.activeSelf && WeightInspector != currentInspector && currentInspector != null)
        {
            currentInspector.active = false;
        }
        currentInspector = WeightInspector;
        gameObject.SetActive(active);
    }
}
