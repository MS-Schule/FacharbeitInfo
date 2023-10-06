using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Main : MonoBehaviour
{
    public RectTransform costGraphContainer;

    [Space()]
    [Header("Buttons")]
    public Button forceUpdate;
    public Button allEpochsBtn;
    public Button oneEpochBtn;
    public Button testBtn;
    public Button continueBtn;

    [Space()]
    [Header("Input fields")]
    public InputField epochAmount;
    public InputField miniBatchSize;
    public InputField fullBatchSize;

    [Space()]
    [Header("Display")]
    public RawImage display;
    public Text displayLabel;
    private Texture2D sampleImage;

    [Space()]
    [Header("Shader")]
    public ComputeShader basicDrawer;
    public ComputeShader layerShader;

    [HideInInspector]
    public Neural_Network neuralNetwork;
    private MNISTTrainer trainer;
    private SecUI secUI;

    public bool executeNextBatch = false;
    void Awake()
    {
        if (!GameObject.Find("Secondary UI").TryGetComponent<SecUI>(out secUI))
            Debug.Log("Could not find sec UI");
        int[] layerSizes = new int[] {
            28 * 28,
            64,
            32,
            10
        };
        neuralNetwork = new Neural_Network(layerSizes, layerShader);
        trainer = new(this);
        sampleImage = new Texture2D(28, 28, TextureFormat.ARGB32, false) { filterMode = FilterMode.Point };
        forceUpdate.onClick.AddListener(neuralNetwork.Compute);
        secUI.Init(this, 3);

        allEpochsBtn.onClick.AddListener(delegate
        {
            StartCoroutine(StartTrainer());
        });

        oneEpochBtn.onClick.AddListener(delegate
        {
            epochAmount.text = "1";
            StartCoroutine(StartTrainer());
        });

        testBtn.onClick.AddListener(delegate
        {
            int sampleIndex = Random.Range(0, int.Parse(fullBatchSize.text));
            trainer.WriteSample(sampleIndex);
            Dataset image = trainer.PassSample(sampleIndex);
            ProjectOnTexture(image, ref sampleImage);
            display.texture = sampleImage;
            displayLabel.text = "Label: " + System.Convert.ToString(image.Label);
            secUI.UpdateScreen(false, int.Parse(epochAmount.text), int.Parse(fullBatchSize.text), int.Parse(miniBatchSize.text));
        });

        continueBtn.onClick.AddListener(delegate
        {
            //executeNextBatch = true;
            foreach (var layer in neuralNetwork.layers)
                layer.ApplyGradients(HyperParameters.learnRate, HyperParameters.regularization, HyperParameters.momentum);
        });
    }

    private IEnumerator StartTrainer()
    {
        ToggleButtons();
        yield return StartCoroutine(trainer.ExecuteEpochs(
            int.Parse(epochAmount.text),
            int.Parse(fullBatchSize.text),
            int.Parse(miniBatchSize.text),
            secUI.UpdateScreen
        ));
        secUI.UpdateScreen(false, int.Parse(epochAmount.text), int.Parse(fullBatchSize.text), int.Parse(miniBatchSize.text));
        ToggleButtons();
    }

    private void ProjectOnTexture(Dataset img, ref Texture2D tex)
    {
        for (int i = 0; i < img.Data.GetLength(0); i++)
        {
            for (int j = 0; j < img.Data.GetLength(1); j++)
            {
                float intensity = (float)img.Data[i, j] / 255f;
                tex.SetPixel(j, 28 - i, new Color(intensity, intensity, intensity, 1));
            }
        }
        tex.Apply();
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
    }
}
