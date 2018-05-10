package LUTtoNeuralNet;

import java.util.*;
import java.io.*;
class BPalgorithm{
//Initialization
private static int NumInputs= 6;
private static int NumHidden;
private static double Weights1[][] ;
private static double Weights2[];
private static double PreWeiCha1[][];
private static double PreWeiCha2[];
private static double hiddenout[];
private static double hiddenDelta[];
private static double out = 0.0f;
private static double rou; //learningrate
private static double bias =1;
private static double delta;
private static double alpha = 0.02f; // momentum
/*package*/
// setters and getters
public int getNumHidden(){
return NumHidden;
}
public void setNumHidden(int X){
NumHidden = X;
}
public double getLearningRate(){
return rou;
}
public void setLearningRate(double X){
rou = X;
}
public double[][] getWeights1(){
return Weights1;
}
public double[] getWeights2(){
return Weights2;
}
// set those matrix according to the value of NumHidden and NumInputs
public void setData(int NumHidden){
Weights1 = new double [NumInputs+1][NumHidden];
Weights2 = new double[NumHidden+1];
hiddenout = new double[NumHidden];
hiddenDelta = new double[NumHidden];
PreWeiCha1= new double[NumInputs+1][NumHidden];
PreWeiCha2 = new double[NumHidden+1];
}
public void initializeWeights()
{ // Initialization of the starting weight matrix Weights1[][] and Weights2[].
Random generator = new Random();
this.setData(NumHidden);
for (int i = 0; i < NumInputs+1; i++)
{for (int j = 0; j < NumHidden; j++){
Weights1[i][j] = generator.nextDouble()-0.5;
}
}
for (int i = 0; i < NumHidden+1; i++){
Weights2[i] = generator.nextDouble()-0.5;
}
}
// The activation function
public double sigmoid (double x){
return 2*(1/(1+Math.exp(-x))-0.5);
}
public void outputFor (double[] X){
// Calculate the output for the hidden nodes
double sum;
for(int j =0; j < NumHidden; j++){
sum = Weights1[0][j]*bias; //bias term
for (int i = 1; i < NumInputs+1; i++){
sum = sum + Weights1[i][j]*X[i-1];
}
hiddenout[j] = sigmoid(sum); // output
}
// Calculate the output for the output nodes
sum = Weights2[0]*bias; //bias term
for (int i =1; i < NumHidden+1; i++){
sum = sum + Weights2[i] *hiddenout[i-1];
}
out = sigmoid(sum); // output
}
// update the weight matrix
public void weightUpdate(int[] X, double[][][][][][] Q){
double C; // the correct output for X
C = Q[X[0]][X[1]][X[2]][X[3]][X[4]][X[5]];
// weights update for the hidden nodes
//System.out.println("A"+out);
delta = (C-out) * 0.5* (1-out*out);
for (int i = 0; i< NumHidden; i++){
Weights2[i+1] = Weights2[i+1] + alpha*PreWeiCha2[i+1] + rou * delta * hiddenout[i];
PreWeiCha2[i+1] =alpha*PreWeiCha2[i+1] + rou*delta*hiddenout[i];// for the momentum
}
Weights2[0] = Weights2[0] +alpha*PreWeiCha2[0]+ rou*delta*bias;
PreWeiCha2[0] = alpha*PreWeiCha2[0] + rou*delta*bias;
// weights update for the input node
for (int i = 0; i < NumHidden; i++){
hiddenDelta[i] = 0.5* (1-hiddenout[i]*hiddenout[i]) * delta * Weights2[i+1];
}
for (int i = 1; i<NumInputs+1;i++){
for(int j = 0; j<NumHidden; j++){
Weights1[i][j]=Weights1[i][j]+alpha*PreWeiCha1[i][j]+rou*hiddenDelta[j]*X[i-1];
PreWeiCha1[i][j] = alpha * PreWeiCha1[i][j] +rou*hiddenDelta[j]*X[i-1]; // for the momentum
}
}
for (int j = 0; j< NumHidden; j++){
Weights1[0][j] = Weights1[0][j]+ alpha * PreWeiCha1[0][j]+ rou*hiddenDelta[j]*bias; // bias term
PreWeiCha1[0][j] = alpha * PreWeiCha1[0][j]+ rou*hiddenDelta[j]*bias;
}
}
// trainning
public int train(double[][][][][][] Q){
this.initializeWeights();
// four training patterns
int i = 0; // number of epochs
double error = 0;
double[] X = new double[NumInputs];
try{
FileWriter abc = new FileWriter("ab.txt"); // write to a .txt. file
do{
error = 0;
for (int k1 = 0; k1 < Q.length; k1++)
{for (int k2 = 0; k2 < Q[0].length; k2++)
{for (int k3 = 0; k3< Q[0][0].length; k3++)
{for(int k4 = 0; k4< Q[0][0][0].length; k4++)
{for(int k5 = 0; k5< Q[0][0][0][0].length; k5++)
{for(int k6 = 0; k6< Q[0][0][0][0][0].length; k6++)
{
X[0] = -1 + k1*2/(Q.length - 1);
X[1] = -1 + k2*2/(Q[0].length - 1);
X[2] = -1 + k3*2/(Q[0][0].length - 1);
X[3] = -1 + k4*2/(Q[0][0][0].length - 1);
X[4] = -1 + k5*2/(Q[0][0][0][0].length -1);
X[5] = -1 + k6*2/(Q[0][0][0][0][0].length -1);
this.outputFor(X);
error = error + 0.5*(Q[k1][k2][k3][k4][k5][k6]-out)*(Q[k1][k2][k3][k4][k5][k6]-out);
int [] Y = {k1,k2,k3,k4,k5,k6};
this.weightUpdate(Y,Q);
}
}
}
}
}
}
abc.write(Double.toString(error)+"\r\n");
abc.flush();
i++;
System.out.println(i);
System.out.println(error);
}while(error >0.05 && i<70000); // stopping criterion
abc.close();
}catch (Exception e){}
return i;
}
}