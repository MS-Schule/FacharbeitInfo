using System.Collections;
using System.Linq;
using UnityEngine;

public enum LearnType
{
    NeuralNetwork,
    Clustering
}

public class Main : MonoBehaviour
{
    #region Public fields

    [Header("Settings")]
    public LearnType learnType;
    public ActivationType activationType;
    public CostType costType;

    public MainUI ui;
    public ErrorGraph errorGraph;

    [HideInInspector]
    public NeuralNetwork neuralNetwork;

    [HideInInspector]
    public Clusterer clusterer;

    #endregion

    #region Private fields

    //Interfaces
    private ISecUI secUI;
    public ISaver saver;
    private ITrainer trainer;
    private ITester tester;

    #endregion

    void Awake()
    {

        if(learnType == LearnType.NeuralNetwork) {
            InitializeNetworkSim();
        } else {
            InitializeClusteringSim();
        }

        trainer.Init(this);
        tester.Init(this);
        secUI.Init(this);
    }

    //---------Training methods-----------

    #region Training

    public IEnumerator StartTrainer(TrainerInfo trainerInfo)
    {
        yield return StartCoroutine(trainer.ExecuteEpochs(
            trainerInfo,
            FinishFrame,
            DocumentErrorRate
        ));
        FinishFrame(false, trainerInfo);
    }

    private void FinishFrame(bool isExecuting, TrainerInfo info, int currentEpoch = 0, int currentBatch = 0) {
        if(ui.errorGraphActive)
            errorGraph.UpdateScreen(isExecuting, info, currentEpoch, currentBatch);
        else
            secUI.UpdateScreen(isExecuting, info, currentEpoch, currentBatch);
    }

    private void DocumentErrorRate() {
        bool status = errorGraph.transform.parent.gameObject.activeInHierarchy;
        errorGraph.transform.parent.gameObject.SetActive(true);
        errorGraph.AddTestNode(tester.CalculateAccuracy());
        errorGraph.AddTrainNode(trainer.CalculateAccuracy());
        errorGraph.transform.parent.gameObject.SetActive(status);
    }

    #endregion

    //--------Button methods--------

    #region Button methods

    public void ToggleSecUI() {
        secUI.ToggleActive();
        errorGraph.ToggleActive();
    }

    public void InsertTestSample() {
        int sampleIndex = Random.Range(0, tester.ImageCount());
        ui.DisplayImage(tester.Classify(sampleIndex));
        ui.displayedImageIndex = sampleIndex;
    }

    public void InsertTrainSample(int trainSamples) {
        int sampleIndex = Random.Range(0, trainSamples);
        ui.DisplayImage(trainer.Classify(sampleIndex));
        ui.displayedImageIndex = sampleIndex;
    }

    public void LogAccuracy() {
        Debug.Log("Calculating full set accuracy. This may take some time...");
        tester.FullSetAccuracy();
    }

    public void ComputeDisplayedSample() {
        int label;
        label = tester.Classify(ui.displayedImageIndex).img.Label;
    }

    #endregion

        //------Initializations----------

    #region Initializations

    private void InitializeNetworkSim() {
        var clusteringUI = GameObject.Find("Secondary UICL");
        if (!clusteringUI.TryGetComponent<CL_SecUI>(out CL_SecUI otherUI))
            Debug.Log("Could not find sec UI CL");
        clusteringUI.transform.parent.gameObject.SetActive(false);

        var networkUI = GameObject.Find("Secondary UINN");
        if (!networkUI.TryGetComponent<ISecUI>(out secUI))
            Debug.Log("Could not find sec UI NN");

        networkUI.transform.parent.gameObject.SetActive(true);

        errorGraph.transform.parent.gameObject.SetActive(false);

        trainer = new NetworkMNISTTrainer(this);
        tester = new NetworkTester();

        int[] layerSizes = new int[3] {
            64,
            32,
            10
        };

        neuralNetwork = new NeuralNetwork(layerSizes, new Vector2Int(28, 28), activationType, costType);
        saver = new NetworkSaver(neuralNetwork);
    }

    private void InitializeClusteringSim() {
        var clusteringUI = GameObject.Find("Secondary UICL");
        if (!clusteringUI.TryGetComponent<ISecUI>(out secUI))
            Debug.Log("Could not find sec UI CL");
        clusteringUI.transform.parent.gameObject.SetActive(true);

        var networkUI = GameObject.Find("Secondary UINN");
        if (!networkUI.TryGetComponent<NN_SecUI>(out NN_SecUI otherUI))
            Debug.Log("Could not find sec UI NN");
        networkUI.transform.parent.gameObject.SetActive(false);

        errorGraph.transform.parent.gameObject.SetActive(false);

        trainer = new ClusterTrainer(this);
        tester = new ClusterTester();

        clusterer = new Clusterer(30, new Vector2Int(28,28), ((ClusterTrainer)trainer).GetRndSamples(90).ToList());
        saver = new ClusterSaver(clusterer);
    }

    #endregion
}
