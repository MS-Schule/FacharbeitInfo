using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

public class NeuronUI : MonoBehaviour
{
    private Text nameLabel;
    private Text valueLabel;
    private Text biasLabel;
    private Text typeLabel;
    private Scrollbar valueSlider;
    private Text valueCostLabel;
    private Text biasCostLabel;
    private Text expectedValLabel;
    private int layerIndex;
    private int neuronIndex;
    private Button closeButton;
    public Main controller;
    private Neural_Network network;
    private NeuronInspector currentInspector;
    void Start()
    {
        network = controller.neuralNetwork;
        gameObject.SetActive(false);
        layerIndex = 0;
        neuronIndex = 0;
        closeButton = gameObject.transform.GetChild(6).GetComponent<Button>();
        closeButton.onClick.AddListener(delegate { SetActive(false, null); });
        nameLabel = gameObject.transform.GetChild(1).GetComponent<Text>();
        valueLabel = gameObject.transform.GetChild(2).GetComponent<Text>();
        biasLabel = gameObject.transform.GetChild(3).GetComponent<Text>();
        typeLabel = gameObject.transform.GetChild(4).GetComponent<Text>();
        valueSlider = gameObject.transform.GetChild(5).GetComponent<Scrollbar>();
        valueCostLabel = gameObject.transform.GetChild(8).GetComponent<Text>();
        biasCostLabel = gameObject.transform.GetChild(9).GetComponent<Text>();
        expectedValLabel = gameObject.transform.GetChild(10).GetComponent<Text>();
        valueSlider.onValueChanged.AddListener(delegate { UpdateValue(); });
    }

    public void UpdateTable(int nIndex, int lIndex)
    {
        neuronIndex = nIndex - 1;
        layerIndex = lIndex - 1;
        nameLabel.text = "Neuron " + lIndex + " - " + nIndex;
        switch (network.layers[layerIndex].LayerType)
        {
            case LayerType.input:
                typeLabel.text = "Type: Input neuron";
                valueSlider.interactable = true;
                expectedValLabel.enabled = false;
                valueCostLabel.enabled = false;
                biasCostLabel.enabled = false;
                biasLabel.color = Color.grey;
                UpdateTable();
                return;
            case LayerType.inner:
            case LayerType.semiConnected:
                typeLabel.text = "Type: Hidden neuron";
                valueSlider.interactable = false;
                expectedValLabel.enabled = false;
                break;
            case LayerType.output:
                typeLabel.text = "Type: Output neuron";
                valueSlider.interactable = true;
                expectedValLabel.enabled = true;
                break;
        }
        biasLabel.color = new Color(0.1960784f, 0.1960784f, 0.1960784f);
        valueCostLabel.enabled = true;
        biasCostLabel.enabled = true;
        UpdateTable();
    }

    public void UpdateTable()
    {
        valueLabel.text = "Value: " + network.layers[layerIndex].activation[neuronIndex];
        switch (network.layers[layerIndex].LayerType)
        {
            case LayerType.input:
                biasLabel.text = "Bias: --";
                return;
            case LayerType.output:
                expectedValLabel.text = "Expected: " + network.GetOutputLayer().desiredOutputs[neuronIndex];
                break;
        }
        biasLabel.text = "Bias: " + network.layers[layerIndex].bias[neuronIndex];
        valueCostLabel.text = "Value: <color="
            + (-network.layers[layerIndex].valueGradient[neuronIndex] > 0 ? "green>" : "red>")
            + -network.layers[layerIndex].valueGradient[neuronIndex] + "</color>";
        biasCostLabel.text = "Bias: <color="
            + (-network.layers[layerIndex].biasGradient[neuronIndex] > 0 ? "green>" : "red>")
            + -network.layers[layerIndex].biasGradient[neuronIndex] + "</color>";
    }

    private void UpdateValue()
    {
        if (network.layers[layerIndex - 1].LayerType == LayerType.input)
        {
            network.layers[layerIndex - 1].activation[neuronIndex - 1] = (valueSlider.value * 2) - 1;
            valueLabel.text = "Value: " + network.layers[layerIndex - 1].activation[neuronIndex - 1];
        }
        else if (network.layers[layerIndex - 1].LayerType != LayerType.inner)
        {
            network.GetOutputLayer().desiredOutputs[neuronIndex - 1] = (valueSlider.value * 2) - 1;
            expectedValLabel.text = "Expected: " + network.GetOutputLayer().desiredOutputs[neuronIndex - 1];
        }
    }

    public void SetActive(bool active, NeuronInspector neuronInspector)
    {
        if (gameObject.activeSelf && neuronInspector != currentInspector && currentInspector != null)
        {
            currentInspector.active = false;
        }
        currentInspector = neuronInspector;
        gameObject.SetActive(active);
    }
}
