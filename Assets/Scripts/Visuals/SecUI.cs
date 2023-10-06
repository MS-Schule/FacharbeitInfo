using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class SecUI : MonoBehaviour
{
    [SerializeField] private Sprite nodeSprite;
    public Font font;
    private RectTransform container;
    private List<Text> outputs;
    public NeuronUI neuronUI;
    public WeightUI weightUI;
    private Neural_Network network;
    private Button updateBtn;
    private Transform infoTable;
    private Transform infoTableClosed;

    private List<GameObject>[] weightMatrix;
    private int skippedLayerNum;

    private void Start()
    {
        if (!transform.Find("graphContainer").TryGetComponent<RectTransform>(out container))
            Debug.Log("Could not find graph container");
        infoTable = gameObject.transform.parent.GetChild(2);
        infoTableClosed = gameObject.transform.parent.GetChild(3);
        updateBtn = infoTable.transform.GetChild(6).GetComponent<Button>();
        updateBtn.onClick.AddListener(UpdateAllWeights);
        outputs = new();

        infoTable.GetChild(7).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(false); });
        infoTableClosed.GetChild(2).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(true); });
        InitNetworkNodes(network.layerNum);
    }

    public void Init(Main controller, int skipLayer)
    {
        skippedLayerNum = skipLayer;
        network = controller.neuralNetwork;
    }

    private void SetInfoTable(bool active)
    {
        infoTable.gameObject.SetActive(active);
        infoTableClosed.gameObject.SetActive(!active);
    }

    public void UpdateScreen(bool isExecuting, int epochAmount, int fullBatchSize, int miniBatchSize, int epoch = 0, int batch = 0)
    {
        UpdateOutputLabels();
        UpdateAllWeights();
        if (infoTableClosed.gameObject.activeInHierarchy) return;
        float batchAmount = fullBatchSize / miniBatchSize;
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
        infoTable.GetChild(2).GetComponent<Text>().text = "Epoch: " + (epoch + 1).ToString() + "/" + epochAmount;
        infoTable.GetChild(3).GetComponent<Text>().text = "Mini-batch: " + batch.ToString() + "/" + batchAmount;

        float progress = (float)epoch / epochAmount;
        progress += 1f / epochAmount * ((float)batch / batchAmount);
        infoTable.GetChild(4).GetComponent<Text>().text = "Progress: " + Math.Round(progress * 100, 2) + "%";
    }

    private void UpdateOutputLabels()
    {
        for (int i = 0; i < network.GetOutputLayer().Size; i++)
        {
            outputs[i].text = "Match: " + Math.Round((network.GetOutputLayer().activation[i] + 1) / 2 * 100, 2) + "%";
            if (network.GetOutputLayer().desiredOutputs[i] > 0)
                outputs[i].color = Color.green;
            else
                outputs[i].color = new Color(0.1960784f, 0.1960784f, 0.1960784f);
        }
    }

    private void UpdateAllWeights()
    {
        for (int i = skippedLayerNum + 1; i < network.layerNum; i++)
        {
            UpdateWeightShape(i, i - skippedLayerNum - 1);
        }
    }

    private void UpdateWeightShape(int layer, int weightMatrixIndex)
    {
        Layer currentLayer;
        currentLayer = network.layers[layer];
        List<GameObject> correspondingWeights = weightMatrix[weightMatrixIndex];

        int x = currentLayer.Size;
        int y = currentLayer.PreviousLayer.Size;
        if (currentLayer.LayerType == LayerType.semiConnected)
            y = currentLayer.PreviousLayer.chunkSize;
        for (int i = 0; i < x * y; i++)
        {
            correspondingWeights[i].GetComponent<Image>().color = currentLayer.weights[i % x][i / x] > 0 ? Color.green : Color.red;
            RectTransform rectTransform = correspondingWeights[i].GetComponent<RectTransform>();
            float width = Mathf.Abs(currentLayer.weights[i % x][i / x]);//Mathf.Max(Mathf.Abs(currentLayer.weights[i%x][i/x]), 1);
            correspondingWeights[i].GetComponent<RectTransform>().sizeDelta = new Vector2(rectTransform.sizeDelta.x, width * 2);
        }
    }

    private void InitNetworkNodes(int layerNum)
    {
        float totalSegmentNum = layerNum + 1f;
        weightMatrix = new List<GameObject>[layerNum - skippedLayerNum - 1];
        for (int i = skippedLayerNum + 1; i < layerNum; i++)
        {
            if (network.layers[i].LayerType == LayerType.semiConnected)
                DrawLayerWeights(network.layers[i].Size, (i + 1) / totalSegmentNum, network.layers[i].PreviousLayer.Size, i / totalSegmentNum, i, network.layers[1].PreviousLayer.chunkSize);
            else
                DrawLayerWeights(network.layers[i].Size, (i + 1) / totalSegmentNum, network.layers[i].PreviousLayer.Size, i / totalSegmentNum, i);
        }

        for (int i = skippedLayerNum; i < layerNum; i++)
        {
            DrawLayer(network.layers[i].Size, (i + 1) / totalSegmentNum, i);
        }
    }

    private void DrawLayerWeights(int numC, float xPosC, int numP, float xPosP, int layerIndex, int numPchunk = 0)
    {
        numC++;
        numP++;
        weightMatrix[layerIndex - skippedLayerNum - 1] = new();
        if (numPchunk == 0)
        {
            for (int i = 1; i < numC; i++)
            {
                Vector2 posA;
                posA.x = 1920 * xPosC;
                posA.y = i * (900 / numC);
                for (int j = 1; j < numP; j++)
                {
                    Vector2 posB;
                    posB.x = 1920 * xPosP;
                    posB.y = j * (900f / numP);
                    CreateWeight(posA, posB, layerIndex, new Vector2Int(i, j));
                }
            }
            return;
        }
        for (int i = 1; i < numC; i++)
        {
            Vector2 posA;
            posA.x = 1920 * xPosC;
            posA.y = i * (900 / numC);
            for (int j = 1; j <= numPchunk; j++)
            {
                Vector2 posB;
                posB.x = 1920 * xPosP;
                posB.y = j * (900f / numP) + (i - 1) * numPchunk * (900f / numP);
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
        // UI:
        gameObject.AddComponent<Button>();
        NeuronInspector ins = gameObject.AddComponent<NeuronInspector>();
        ins.lIndex = index.x;
        ins.nIndex = index.y;
        ins.window = neuronUI;
        ins.Init();
        // Output text
        if (index.x == network.layerNum)
        {
            GameObject outputText = new("outputText", typeof(Text));
            outputs.Add(outputText.GetComponent<Text>());
            outputText.transform.SetParent(gameObject.transform, false);
            outputText.transform.localPosition = new Vector2(50f, 0f);
            outputs.Last().fontSize = 20;
            outputs.Last().text = "Match: ";
            outputs.Last().font = font;
            outputs.Last().color = new Color(0.1960784f, 0.1960784f, 0.1960784f);
        }
        // Transform:
        gameObject.transform.SetParent(container, false);
        gameObject.GetComponent<Image>().sprite = nodeSprite;
        RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
        rectTransform.localPosition = position;
        rectTransform.sizeDelta = new Vector2(scale, scale);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(0, 0);
    }
}
