using System.Collections;
using System.Collections.Generic;
using System.Reflection.Emit;
using UnityEngine;
using UnityEngine.UI;

public class NeuronInspector : MonoBehaviour
{
    public int lIndex;
    public int nIndex;
    private Button button;
    public NeuronUI window;
    public bool active = false;

    public void Init()
    {
        button = gameObject.GetComponent<Button>();
        button.onClick.AddListener(Test);
    }

    private void Test()
    {
        active = !active;
        window.SetActive(active, this);
        window.UpdateTable(nIndex, lIndex);
    }
}
