using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NewBehaviourScript : MonoBehaviour {

    public static int moveNum;
    /*
     * 0: position
     * 1: rotation
     * 2: scale
     */
    public int[] moveType = new int[moveNum];
    public double[,] creditcalPoints = new double[moveNum, 10];

    public static int objNum;
    public GameObject[] obj = new GameObject[objNum];
    private double[,] position = new double[objNum, 3];
    private double[,] rotation = new double[objNum, 3];
    private double[,] scale = new double[objNum, 3];

    // Use this for initialization
    void Start () {
        /*
         * Initializing...
         */
		for(int i = 0; i < objNum; i++)
        {
            position[i, 0] = obj[i].transform.position.x;
            position[i, 1] = obj[i].transform.position.y;
            position[i, 2] = obj[i].transform.position.z;
            rotation[i, 0] = obj[i].transform.rotation.x;
            rotation[i, 1] = obj[i].transform.rotation.y;
            rotation[i, 2] = obj[i].transform.rotation.z;
            scale[i, 0] = obj[i].transform.localScale.x;
            scale[i, 1] = obj[i].transform.localScale.y;
            scale[i, 2] = obj[i].transform.localScale.z;
        }
	}
	
	// Update is called once per frame
	void Update () {
		
	}

}
