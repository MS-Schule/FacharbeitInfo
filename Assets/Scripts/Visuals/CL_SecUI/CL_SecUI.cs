using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class CL_SecUI : MonoBehaviour, ISecUI
{
    public Font font;
    public RectTransform container;
    private List<RawImage> images;
    private Clusterer clusterer;
    private Transform infoTable;
    private Transform infoTableClosed;

    public void Start()
    {
        if (!transform.GetChild(1).TryGetComponent<RectTransform>(out container))
            Debug.Log("Could not find image container");
        infoTable = gameObject.transform.parent.GetChild(1);
        infoTableClosed = gameObject.transform.parent.GetChild(2);

        infoTable.GetChild(5).GetComponent<Button>().onClick.AddListener(UpdateImages);
        infoTable.GetChild(6).GetComponent<Button>().onClick.AddListener(ChangeVisibility);

        infoTable.GetChild(7).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(false); });
        infoTableClosed.GetChild(2).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(true); });
        images = new();
        InitCentroidImages(clusterer.clusterAmount);
    }

    public void Init(Main controller)
    {
        clusterer = controller.clusterer;
    }

    private void SetInfoTable(bool active)
    {
        infoTable.gameObject.SetActive(active);
        infoTableClosed.gameObject.SetActive(!active);
    }

    private void ChangeVisibility() {
        foreach(var img in images) {
            img.gameObject.SetActive(!img.gameObject.activeInHierarchy);
        }
    }

    private void InitCentroidImages(int cluster)
    {
        images = new();
        int rows = Mathf.FloorToInt(Mathf.Sqrt(cluster));
        int columns = Mathf.CeilToInt(cluster / (float)rows);
        int rowHeight = Mathf.RoundToInt(700f / rows);
        int colWidth = Mathf.RoundToInt(1600f / columns);
        int sideLength = Mathf.Min(rowHeight, colWidth) - 10;
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                int index = row * columns + col;
                if (index >= cluster) break;
                GameObject display = new("display(" + col + "," + row + ")", typeof(RawImage));
                display.transform.SetParent(container, false);
                images.Add(display.GetComponent<RawImage>());
                images[index].texture = new Texture2D(clusterer.DataSize.x, clusterer.DataSize.y, TextureFormat.RGBA32, true) {filterMode = FilterMode.Point};
                Vector2 position;
                position.x = col * colWidth + colWidth/2f;
                position.y = row * rowHeight + rowHeight/2f;
                RectTransform rectTransform = display.GetComponent<RectTransform>();
                rectTransform.localPosition = position;
                rectTransform.sizeDelta = new Vector2(sideLength, sideLength);
                rectTransform.anchorMin = new Vector2(0, 0);
                rectTransform.anchorMax = new Vector2(0, 0);
            }
        }
    }

    public void UpdateScreen(bool isExecuting, TrainerInfo info, int epoch = 0, int batch = 0)
    {
        UpdateImages();
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

    public void UpdateImages()
    {
        List<float[,]> centroids = clusterer.GetCentroids().ToList();
        for (int i = 0; i < clusterer.clusterAmount; i++)
        {
            var tex = images[i].texture as Texture2D;
            ProjectOnTexture(centroids[i], ref tex);
            images[i].texture = tex;
        }
    }

    private static void ProjectOnTexture(float[,] img, ref Texture2D tex)
    {
        for (int i = 0; i < img.GetLength(0); i++)
        {
            for (int j = 0; j < img.GetLength(1); j++)
            {
                float intensity = (float)img[i, j];
                tex.SetPixel(j, 28 - i, new Color(intensity, intensity, intensity, 1));
            }
        }
        tex.Apply();
    }

    public void ToggleActive() {
        transform.parent.gameObject.SetActive(!transform.parent.gameObject.activeInHierarchy);
    }
}
