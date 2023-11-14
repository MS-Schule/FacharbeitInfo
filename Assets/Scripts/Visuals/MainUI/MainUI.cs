using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public struct TrainerInfo {
    public int epochAmount;
    public int fullBatchSize;
    public int miniBatchSize;
    public int batchAmount;
}

public class MainUI : MonoBehaviour
{
    #region Public Fields

    public Main controller;

    [Space()]
    [Header("Buttons")]
    public Transform testButtonGroup;
    public Transform trainButtonGroup;
    public Transform saveButtonGroup;

    private Toggle drawModeToggle;

    [Space()]
    [Header("Input fields")]
    public InputField epochAmount;
    public InputField miniBatchSize;
    public InputField fullBatchSize;

    [Space()]
    [Header("Display")]
    public Font defaultFont;
    public GameObject percentageTabel;
    public GameObject display;
    public int displayedImageIndex;

    public GameObject drawingDisplay;

    [HideInInspector]
    public bool errorGraphActive;

    #endregion

    #region Private Fields

    //Buttons
    private Button forceUpdate;
    private Button allEpochsBtn;
    private Button oneEpochBtn;
    private Button trainSampleBtn;
    private Button testBtn;
    private Button accuracyBtn;

    //Display
    private RawImage displayImage;
    private Text displayLabel;
    private Text displayClassLabel;

    //Second screen
    private Button screenChangeBtn;

    #endregion

    public void Start() {
        testBtn = testButtonGroup.GetChild(0).GetComponent<Button>();
        forceUpdate = testButtonGroup.GetChild(1).GetComponent<Button>();
        drawModeToggle = testButtonGroup.GetChild(2).GetComponent<Toggle>();
        screenChangeBtn = testButtonGroup.GetChild(3).GetComponent<Button>();
        accuracyBtn = testButtonGroup.GetChild(4).GetComponent<Button>();
        errorGraphActive = false;

        allEpochsBtn = trainButtonGroup.GetChild(0).GetComponent<Button>();
        oneEpochBtn = trainButtonGroup.GetChild(1).GetComponent<Button>();
        trainSampleBtn = trainButtonGroup.GetChild(2).GetComponent<Button>();

        displayImage = display.GetComponent<RawImage>();
        displayLabel = display.transform.GetChild(0).GetComponent<Text>();
        displayClassLabel = display.transform.GetChild(1).GetComponent<Text>();

        saveButtonGroup.GetChild(0).GetComponent<Button>().onClick.AddListener(controller.saver.Save);
        var loadIndex_TextField = saveButtonGroup.GetChild(2).GetComponent<InputField>();
        saveButtonGroup.GetChild(1).GetComponent<Button>().onClick.AddListener(delegate { controller.saver.Load(int.Parse(loadIndex_TextField.text)); });

        InitScreen();
    }

    public void DisplayImage((Dataset image, float[] outputs) classifiedData) {
        Dataset image = classifiedData.image;
        List<float> outputs = classifiedData.outputs.ToList();

        Texture2D sampleImage = displayImage.texture as Texture2D;
        ProjectOnTexture(image, ref sampleImage);
        displayImage.texture = sampleImage;

        displayLabel.text = "Label: " + System.Convert.ToString(image.Label);
        int classifiedIndex = outputs.IndexOf(outputs.Max());
        displayClassLabel.text = "Classified as: " + System.Convert.ToString(classifiedIndex);
        displayClassLabel.color = image.Label == classifiedIndex ? Color.green : Color.red;

        if(controller.learnType == LearnType.NeuralNetwork) {
            List<float> orderedOutputs = outputs.OrderByDescending(o => o).ToList();
            for(int i = 0; i < 10; i++) {
                float classIndex = outputs.IndexOf(orderedOutputs[i]);
                percentageTabel.transform.GetChild(i).GetComponent<Text>().text = "Match for " + classIndex + ": " + (100 * (orderedOutputs[i] + 1) / 2).ToString("0.##") + "%";
                percentageTabel.transform.GetChild(i).GetComponent<Text>().color = new Color(0.4f, 0.4f, 0.4f);
            }
            percentageTabel.transform.GetChild(0).GetComponent<Text>().color = new Color(0.7f, 0.7f, 0.7f);
        }
    }

    private static void ProjectOnTexture(Dataset img, ref Texture2D tex)
    {
        for (int i = 0; i < img.Data.GetLength(0); i++)
        {
            for (int j = 0; j < img.Data.GetLength(1); j++)
            {
                float intensity = (float)img.Data[i, j];
                tex.SetPixel(j, img.Data.GetLength(0) - i, new Color(intensity, intensity, intensity, 1));
            }
        }
        tex.Apply();
    }

    public void InitScreen() {
        DefTestButtons();
        DefTrainButtons();

        if(controller.learnType == LearnType.Clustering) {
            percentageTabel.SetActive(false);
            return;
        } else {
            for(int i = 0; i < controller.neuralNetwork.Layers[^1].Size; i++) {
                GameObject text = new("Output " + i, typeof(Text));
                text.transform.SetParent(percentageTabel.transform, false);
                text.transform.localPosition = new Vector2(0, 195 - i * 43.3333f);
                text.GetComponent<Text>().text = "Match for " + i + ": ";
                text.GetComponent<Text>().font = defaultFont;
                text.GetComponent<Text>().color = new Color(0.4f, 0.4f, 0.4f);
                text.GetComponent<Text>().alignment = TextAnchor.MiddleLeft;
                text.GetComponent<Text>().fontSize = 16;
                text.GetComponent<RectTransform>().sizeDelta = new Vector2(140, 30);
            }
        }
    }

    private void DefTestButtons() {
        testBtn.onClick.AddListener(controller.InsertTestSample);
        forceUpdate.onClick.AddListener(controller.ComputeDisplayedSample);

        drawModeToggle.onValueChanged.AddListener(delegate
        {
            display.SetActive(!display.activeInHierarchy);
            drawingDisplay.SetActive(!drawingDisplay.activeInHierarchy);
        });

        screenChangeBtn.onClick.AddListener(delegate
        {
            controller.ToggleSecUI();
            errorGraphActive = !errorGraphActive;
        });

        accuracyBtn.onClick.AddListener(controller.LogAccuracy);
    }

    private void DefTrainButtons() {
        allEpochsBtn.onClick.AddListener(delegate
        {
            ToggleButtons();
            TrainerInfo info = new() {
                epochAmount = int.Parse(epochAmount.text),
                fullBatchSize = int.Parse(fullBatchSize.text),
                miniBatchSize = int.Parse(miniBatchSize.text)
            };
            info.batchAmount = info.fullBatchSize / info.miniBatchSize;
            StartCoroutine(controller.StartTrainer(info));
            ToggleButtons();
        });

        oneEpochBtn.onClick.AddListener(delegate
        {
            ToggleButtons();
            TrainerInfo info = new() {
                epochAmount = 1,
                fullBatchSize = int.Parse(fullBatchSize.text),
                miniBatchSize = int.Parse(miniBatchSize.text)
            };
            info.batchAmount = info.fullBatchSize / info.miniBatchSize;
            StartCoroutine(controller.StartTrainer(info));
            ToggleButtons();
        });

        trainSampleBtn.onClick.AddListener(delegate { controller.InsertTrainSample(int.Parse(fullBatchSize.text)); });
    }

    private void ToggleButtons()
    {
        forceUpdate.interactable = !forceUpdate.interactable;
        allEpochsBtn.interactable = !allEpochsBtn.interactable;
        oneEpochBtn.interactable = !oneEpochBtn.interactable;
        testBtn.interactable = !testBtn.interactable;
        epochAmount.interactable = !epochAmount.interactable;
        fullBatchSize.interactable = !fullBatchSize.interactable;
        miniBatchSize.interactable = !miniBatchSize.interactable;
        saveButtonGroup.GetChild(0).GetComponent<Button>().interactable = !saveButtonGroup.GetChild(0).GetComponent<Button>().interactable;
        saveButtonGroup.GetChild(1).GetComponent<Button>().interactable = !saveButtonGroup.GetChild(1).GetComponent<Button>().interactable;
    }
}

/*

private void CreateLog(object obj) {
    int index = 0;
    string fileName = string.Format("Assets/Logs/log_{0}", index);
    while(File.Exists(fileName)) {
        index++;
        fileName = string.Format("Assets/Logs/log_{0}", index);
    }
    Debug.Log("Saved to index: " + index);
    File.WriteAllText(fileName, JsonConvert.SerializeObject(obj, Formatting.Indented));
}

*/