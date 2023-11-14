using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Schema;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;


public class NN_SecUI : MonoBehaviour, ISecUI
{
    [SerializeField] private Sprite nodeSprite;
    public Font font;
    public RectTransform container;
    public NeuronUI neuronUI;
    public WeightUI weightUI;
    private NeuralNetwork network;
    private Button updateBtn;
    private Transform infoTable;
    private Transform infoTableClosed;

    private List<GameObject>[] weightMatrix;
    public int skippedLayerNum;

    private void Start()
    {
        if (!transform.GetChild(1).TryGetComponent<RectTransform>(out container))
            Debug.Log("Could not find graph container");
        infoTable = gameObject.transform.parent.GetChild(2);
        infoTableClosed = gameObject.transform.parent.GetChild(3);
        updateBtn = infoTable.transform.GetChild(5).GetComponent<Button>();
        updateBtn.onClick.AddListener(UpdateAllWeights);

        infoTable.GetChild(6).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(false); });
        infoTableClosed.GetChild(2).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(true); });
        InitNetworkNodes(network.LayerNum);
    }

    public void Init(Main controller)
    {
        network = controller.neuralNetwork;
    }

    private void SetInfoTable(bool active)
    {
        infoTable.gameObject.SetActive(active);
        infoTableClosed.gameObject.SetActive(!active);
    }

    public void UpdateScreen(bool isExecuting, TrainerInfo info, int epoch = 0, int batch = 0)
    {
        UpdateAllWeights();
        if (infoTableClosed.gameObject.activeInHierarchy) return;
        if (isExecuting)
        {
            infoTable.GetChild(1).GetComponent<Text>().text = "Currently executing...";
            infoTable.GetChild(1).GetComponent<Text>().color = new Color(0.91f, 0.43f, 0.43f); //light red
        }
        else
        {
            infoTable.GetChild(1).GetComponent<Text>().text = "Currently not executing";
            infoTable.GetChild(1).GetComponent<Text>().color = new Color(0.42f, 0.91f, 0.43f); //light green
        }
        infoTable.GetChild(2).GetComponent<Text>().text = "Epoch: " + (epoch + 1).ToString() + "/" + info.epochAmount;
        infoTable.GetChild(3).GetComponent<Text>().text = "Mini-batch: " + batch.ToString() + "/" + info.batchAmount;

        float progress = (float)epoch / info.epochAmount;
        progress += 1f / info.epochAmount * ((float)batch / info.batchAmount);
        infoTable.GetChild(4).GetComponent<Text>().text = "Progress: " + Math.Round(progress * 100, 2) + "%";
    }

    private void UpdateAllWeights()
    {
        for (int i = skippedLayerNum + 1; i < network.LayerNum; i++)
        {
            UpdateWeightShape(i, i - skippedLayerNum - 1);
        }
    }

    private void UpdateWeightShape(int layer, int weightMatrixIndex)
    {
        Layer currentLayer;
        currentLayer = network.Layers[layer];
        List<GameObject> correspondingWeights = weightMatrix[weightMatrixIndex];

        int x = currentLayer.Size;
        int y = network.Layers[layer-1].Size;
        for (int i = 0; i < x * y; i++)
        {
            correspondingWeights[i].GetComponent<Image>().color = currentLayer.Weights[i / y][i % y] > 0 ? Color.green : Color.red;
            RectTransform rectTransform = correspondingWeights[i].GetComponent<RectTransform>();
            float width = Mathf.Abs(currentLayer.Weights[i / y][i % y]);
            correspondingWeights[i].GetComponent<RectTransform>().sizeDelta = new Vector2(rectTransform.sizeDelta.x, width * 2);
        }
    }

    private void InitNetworkNodes(int layerNum)
    {
        int totalLayerNum = layerNum + 1;
        float totalSegmentNum = totalLayerNum + 1f;
        weightMatrix = new List<GameObject>[layerNum- skippedLayerNum - 1];
        for (int i = skippedLayerNum + 1; i < layerNum; i++)
        {
            DrawLayerWeights(network.Layers[i].Size, (i + 2) / totalSegmentNum, network.Layers[i-1].Size, (i + 1) / totalSegmentNum, i);
        }

        for (int i = skippedLayerNum; i < layerNum; i++)
        {
            DrawLayer(network.Layers[i].Size, (i + 2) / totalSegmentNum, i);
        }
    }

    private void DrawLayerWeights(int numC, float xPosC, int numP, float xPosP, int layerIndex)
    {
        weightMatrix[layerIndex - skippedLayerNum - 1] = new();
        for (int i = 1; i <= numC; i++)
        {
            Vector2 posA;
            posA.x = 1920 * xPosC;
            posA.y = i * (900f / (numC+1));
            for (int j = 1; j <= numP; j++)
            {
                Vector2 posB;
                posB.x = 1920 * xPosP;
                posB.y = j * (900f / (numP+1));
                CreateWeight(posA, posB, layerIndex, new Vector2Int(i, j));
            }
        }
    }

    private void DrawLayer(int num, float xPosition, int layerIndex)
    {
        num++;
        for (int i = 1; i < num; i++)
        {
            if (num > 100)
                CreateNode(new Vector2(1920 * xPosition, i * (900f / num)), new Vector2Int(layerIndex + 1, i), 4);
            else
                CreateNode(new Vector2(1920 * xPosition, i * (900f / num)), new Vector2Int(layerIndex + 1, i));
        }
    }

    private void CreateWeight(Vector2 posA, Vector2 posB, int layerIndex, Vector2Int neuronIndex)
    {
        GameObject gameObject = new("weight", typeof(Image));
        // UI:
        gameObject.AddComponent<Button>();
        WeightInspector ins = gameObject.AddComponent<WeightInspector>();
        ins.lIndex = layerIndex + 1;
        ins.outputIndex = neuronIndex.x;
        ins.inputIndex = neuronIndex.y;
        ins.window = weightUI;
        ins.Init();
        // Transform:
        gameObject.transform.SetParent(container, false);
        gameObject.GetComponent<Image>().color = Color.red;
        RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
        Vector2 dir = (posB - posA).normalized;
        float dist = Vector2.Distance(posA, posB);
        rectTransform.localEulerAngles = new Vector3(0, 0, Mathf.Atan(dir.y / dir.x) * (180 / Mathf.PI));
        rectTransform.localPosition = posA + .5f * dist * dir;
        rectTransform.sizeDelta = new Vector2(dist, 3);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(0, 0);
        // Save:
        weightMatrix[layerIndex - skippedLayerNum - 1].Add(gameObject);
    }

    private void CreateNode(Vector2 position, Vector2Int index, float scale = 50f)
    {
        GameObject gameObject = new("neuron", typeof(Image));
        // NeuronUI:
        gameObject.AddComponent<Button>();
        NeuronInspector ins = gameObject.AddComponent<NeuronInspector>();
        ins.lIndex = index.x;
        ins.nIndex = index.y;
        ins.window = neuronUI;
        ins.Init();
        // Transform:
        gameObject.transform.SetParent(container, false);
        gameObject.GetComponent<Image>().sprite = nodeSprite;
        RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
        rectTransform.localPosition = position;
        rectTransform.sizeDelta = new Vector2(scale, scale);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(0, 0);
    }

    public void ToggleActive() {
        transform.parent.gameObject.SetActive(!transform.parent.gameObject.activeInHierarchy);
    }
}
