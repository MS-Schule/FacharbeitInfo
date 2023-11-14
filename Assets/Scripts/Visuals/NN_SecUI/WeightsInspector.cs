using System.Collections;
using System.Collections.Generic;
using System.Reflection.Emit;
using UnityEngine;
using UnityEngine.UI;

public class WeightInspector : MonoBehaviour
{
    public int lIndex;
    public int outputIndex;
    public int inputIndex;
    private Button button;
    public WeightUI window;
    public bool active = false;

    public void Init()
    {
        button = gameObject.GetComponent<Button>();
        button.onClick.AddListener(Clicked);
    }

    private void Clicked()
    {
        active = !active;
        window.SetActive(active, this);
        window.UpdateTable(lIndex, outputIndex, inputIndex);
    }
}
