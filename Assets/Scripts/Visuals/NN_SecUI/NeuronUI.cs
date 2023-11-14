using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

public class NeuronUI : MonoBehaviour
{
    private Text nameLabel;
    private Text biasLabel;
    private Text typeLabel;
    private Text biasCostLabel;
    private int layerIndex;
    private int neuronIndex;
    private Button closeButton;
    public Main controller;
    private NeuralNetwork network;
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
        biasLabel = gameObject.transform.GetChild(2).GetComponent<Text>();
        typeLabel = gameObject.transform.GetChild(3).GetComponent<Text>();
        biasCostLabel = gameObject.transform.GetChild(4).GetComponent<Text>();
    }

    public void UpdateTable(int nIndex, int lIndex)
    {
        neuronIndex = nIndex - 1;
        layerIndex = lIndex - 1;
        nameLabel.text = "Neuron " + lIndex + " - " + nIndex;

        if(layerIndex == 0) {
            typeLabel.text = "Type: Input neuron";
            biasCostLabel.enabled = false;
            biasLabel.color = Color.grey;
            UpdateTable();
            return;
        } else if(layerIndex == network.LayerNum) {
            typeLabel.text = "Type: Output neuron";
        } else {
            typeLabel.text = "Type: Hidden neuron";
        }

        biasLabel.color = new Color(0.1960784f, 0.1960784f, 0.1960784f);
        biasCostLabel.enabled = true;
        UpdateTable();
    }

    public void UpdateTable()
    {
        if(layerIndex == 0) {
            biasLabel.text = "Bias: --";
            return;
        }

        biasLabel.text = "Bias: " + network.Layers[layerIndex].Bias[neuronIndex];
        biasCostLabel.text = "Bias: <color="
            + (-network.Layers[layerIndex].BiasGradient[neuronIndex] > 0 ? "green>" : "red>")
            + -network.Layers[layerIndex].BiasGradient[neuronIndex] + "</color>";
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
