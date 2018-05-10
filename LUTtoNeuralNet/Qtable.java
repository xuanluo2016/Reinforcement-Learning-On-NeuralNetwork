package LUTtoNeuralNet;

import java.util.*;
public class Qtable {
	double Q[][][][][][];
	int NumHeading;
	int NumTargetBearing;
	int NumTargetDistance;
	int NumXlabel;
	int NumYlabel;
	int Actions;
//constructor
public Qtable(int a , int b, int c, int d, int e, int f){
this.setPara(a,b,c,d,e,f);
this.initialiseLUT();
}
public void setPara(int a, int b, int c, int d, int e, int f){
// set the parameter
this.NumHeading = a;
this.NumTargetBearing = b;
this.NumTargetDistance = c;
this.NumXlabel = d;
this.NumYlabel = e;
this.Actions = f;
}
/*Initialize the look up table to random matrix */
public void initialiseLUT() {
this.Q = new double[NumHeading][NumTargetBearing][NumTargetDistance][NumXlabel][NumYlabel][Actions];
Random generator = new Random();
for (int k1 = 0; k1 < this.NumHeading; k1++)
{for (int k2 = 0; k2 < this.NumTargetBearing; k2++)
{for (int k3 = 0; k3< this.NumTargetDistance;k3++)
{for(int k4 = 0; k4< this.NumXlabel;k4++)
{for(int k5 = 0; k5< this.NumYlabel;k5++)
{for(int k6 = 0; k6< this.Actions;k6++)
Q[k1][k2][k3][k4][k5][k6] = generator.nextDouble();
}
}
}
}
}
}
//quantization
public int quanHeading(double robotHeading){
double angle = 360 / NumHeading;
double newHeading = robotHeading + angle / 2;
int a = (int)(newHeading /angle);
if(a == 4){a = 0;}
return a;
}
public int quanTargetBearing(double e){
double angle = 360 / NumTargetBearing;
double newBearing = (180+e) + angle/ 2;
int a = (int)(newBearing / angle);
if (a == 4){a = 0;}
return a;
}
public int quanDistance(double e){
return Math.min((int)(e / 120.0),this.NumTargetDistance - 1);
}
public int quanXlabel(double x){
return Math.min((int)(x/200),this.NumXlabel-1);
}
public int quanYlabel(double y){
return Math.min((int)(y/200), this.NumYlabel-1);
}
public int[] quanQ(double a, double b, double c, double d, double e){
int[] result=new int[5];
result[0] = this.quanHeading(a);
result[1] = this.quanTargetBearing(b);
result[2] = this.quanDistance(c);
result[3]= this.quanXlabel(d);
result[4] = this.quanYlabel(e);
return result;
}
// find the action number corresponding to the maximum value in the state-action pair table corresponding to a certain state
public int maxQValue(int i, int j, int k, int m, int n){
double max = -100;
int counter = 0;
int action = 0;
for (counter =0;counter<this.Actions;counter++){
if (this.Q[i][j][k][m][n][counter]>max){
max=this.Q[i][j][k][m][n][counter];
action= counter;
}
}
return action;
}
// pick the action according to epsilon greedy
public int pickAction(int i, int j, int k, int m, int n,double thre){
int result;
Random generator = new Random();
if(generator.nextDouble()<thre)
{result =Math.abs(generator.nextInt())%12;}
else{result = this.maxQValue(i, j, k, m, n);}
return result;
}
}