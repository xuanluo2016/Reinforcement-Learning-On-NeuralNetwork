package LUTtoNeuralNet;

import java.io.*;
public class LUTtoNN {
static int NumHeading = 4;
static int NumTargetBearing = 6;
static int NumTargetDistance = 5;
static int NumXlabel = 4;
static int NumYlabel = 3;
static int Actions = 6;
static int NumHidden = 12;
static double Q[][][][][][] = new double[NumHeading][NumTargetBearing][NumTargetDistance][NumXlabel][NumYlabel][Actions];
static BPalgorithm BPNN = new BPalgorithm();
public static void main (String arg[]) throws Exception{
	loadQ();
	normQ();
	double x=0;
	BPalgorithm BPNN = new BPalgorithm();
	BPNN.setLearningRate(0.1); // set the learning rate
	BPNN.setNumHidden(NumHidden); // set the number of hidden nodes
	x=BPNN.train(Q);
}
public static void loadQ() throws IOException {
	String buff = new String();
	//BufferedReader br = new BufferedReader(new FileReader("q.txt"));
	BufferedReader br = new BufferedReader(new FileReader("LUT.dat"));
	for (int k1 = 0; k1 < NumHeading; k1++)
	{
		for (int k2 = 0; k2 < NumTargetBearing; k2++)
		{
			for (int k3 = 0; k3< NumTargetDistance; k3++)
			{
				for(int k4 = 0; k4< NumXlabel; k4++)
				{
					for(int k5 = 0; k5< NumYlabel; k5++)
						{
							for(int k6 = 0; k6< Actions; k6++)
							{
								buff = br.readLine();
								if (buff!=null)
								{
									Q[k1][k2][k3][k4][k5][k6] = Double.parseDouble(buff);
								}
							}
						}
				}
			}
		}
	}
	br.close();
}

public static void normQ(){
	double maxvalue = -2f;
	for (int k1 = 0; k1 < NumHeading; k1++)
	{
		for (int k2 = 0; k2 < NumTargetBearing; k2++)
		{
			for (int k3 = 0; k3< NumTargetDistance; k3++)
			{
				for(int k4 = 0; k4< NumXlabel; k4++)
				{
					for(int k5 = 0; k5< NumYlabel; k5++)
					{
						for(int k6 = 0; k6< Actions; k6++)
						{
							if(Math.abs(Q[k1][k2][k3][k4][k5][k6])>maxvalue)
							{
								maxvalue = Math.abs(Q[k1][k2][k3][k4][k5][k6]);
							}
						}
					}
				}
			}
		}
	}
	
for (int k1 = 0; k1 < NumHeading; k1++)
	{
		for (int k2 = 0; k2 < NumTargetBearing; k2++)
		{
			for (int k3 = 0; k3< NumTargetDistance; k3++)
			{
				for(int k4 = 0; k4< NumXlabel; k4++)
				{
					for(int k5 = 0; k5< NumYlabel; k5++)
					{
						for(int k6 = 0; k6< Actions; k6++)
						{
							Q[k1][k2][k3][k4][k5][k6]=Q[k1][k2][k3][k4][k5][k6]/maxvalue;
						}
					}
				}
			}
		}
	}
}

}