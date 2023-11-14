using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using UnityEngine;
using System;

public class ErrorGraph : MonoBehaviour
{
    private RectTransform container;
    private int testNodeCount;
    private Vector2 previousTestPos;

    private int trainNodeCount;
    private Vector2 previousTrainPos;

    private Transform infoTable;
    private Transform infoTableClosed;

    void Start()
    {
        testNodeCount = 0;
        trainNodeCount = 0;
        container = transform.GetChild(1).GetComponent<RectTransform>();
        previousTestPos = new Vector2(0, 0);
        previousTrainPos = new Vector2(0, 0);

        infoTable = transform.parent.GetChild(1);
        infoTableClosed = transform.parent.GetChild(2);

        infoTable.GetChild(5).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(false); });
        infoTableClosed.GetChild(2).GetComponent<Button>().onClick.AddListener(delegate { SetInfoTable(true); });
    }

    public void Init(Main control) {
        //no need for that here
    }

    private void SetInfoTable(bool active)
    {
        infoTable.gameObject.SetActive(active);
        infoTableClosed.gameObject.SetActive(!active);
    }

    public void UpdateScreen(bool isExecuting, TrainerInfo info, int epoch = 0, int batch = 0)
    {
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

    public void AddTestNode(float accuracy) {
        Vector2 pos = new(++testNodeCount * 30, accuracy * 700);
        GameObject path = new("path" + testNodeCount, typeof(Image));
        path.transform.SetParent(container, false);
        path.GetComponent<Image>().color = Color.blue;
        RectTransform rectTransform = path.GetComponent<RectTransform>();
        Vector2 dir = (pos - previousTestPos).normalized;
        float dist = Vector2.Distance(previousTestPos, pos);
        rectTransform.localEulerAngles = new Vector3(0, 0, Mathf.Atan(dir.y / dir.x) * (180 / Mathf.PI));
        rectTransform.localPosition = previousTestPos + .5f * dist * dir;
        rectTransform.sizeDelta = new Vector2(dist, 1);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(0, 0);
        previousTestPos = pos;
    }

    public void AddTrainNode(float accuracy) {
        Vector2 pos = new(++trainNodeCount * 30, accuracy * 700);
        GameObject path = new("path" + trainNodeCount, typeof(Image));
        path.transform.SetParent(container, false);
        path.GetComponent<Image>().color = Color.red;
        RectTransform rectTransform = path.GetComponent<RectTransform>();
        Vector2 dir = (pos - previousTrainPos).normalized;
        float dist = Vector2.Distance(previousTrainPos, pos);
        rectTransform.localEulerAngles = new Vector3(0, 0, Mathf.Atan(dir.y / dir.x) * (180 / Mathf.PI));
        rectTransform.localPosition = previousTrainPos + .5f * dist * dir;
        rectTransform.sizeDelta = new Vector2(dist, 1);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(0, 0);
        previousTrainPos = pos;
    }

    public void ToggleActive() {
        transform.parent.gameObject.SetActive(!gameObject.activeInHierarchy);
    }
}
