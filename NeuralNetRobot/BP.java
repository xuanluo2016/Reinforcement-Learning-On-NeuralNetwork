package NeuralNetRobot;

class BP{
	private static int argNumInputs;
	private static int argNumHidden;	
	
	private static double argLearningRate; 
	private static double argMomentumTerm;
	private static double argDiscountRate;
	private static double bias =1;
	private static double delta;
	
	private static double weight1[][] ;
	private static double weights2[];
	private static double preWeights1[][];
	private static double preWeights2[];
	private static double hiddenOutputs[];
	private static double hiddenDeltas[];
	
	public int getArgNumInputs() {
		return argNumInputs;
	}
	public void setArgNumInputs(int argNumInputs) {
		BP.argNumInputs = argNumInputs;
	}
	public int getArgNumHidden() {
		return argNumHidden;
	}
	public void setArgNumHidden(int argNumHidden) {
		BP.argNumHidden = argNumHidden;
	}
	public double getArgLearningRate() {
		return argLearningRate;
	}
	public void setArgLearningRate(double argLearningRate) {
		BP.argLearningRate = argLearningRate;
	}
	public double getArgMomentumTerm() {
		return argMomentumTerm;
	}
	public void setArgMomentumTerm(double argMomentumTerm) {
		BP.argMomentumTerm = argMomentumTerm;
	}
	public double getArgDiscountRate() {
		return argDiscountRate;
	}
	public void setArgDiscountRate(double argDiscountRate) {
		BP.argDiscountRate = argDiscountRate;
	}
			
	// set those matrix according to the value of argNumHidden and argNumInputs
	public void initializeWeights()
	{ 
		// Initialization of the starting weight matrix weight1[][] and weights2[].
		weight1 = new double [argNumInputs+1][argNumHidden];
		weights2 = new double[argNumHidden+1];
		hiddenOutputs = new double[argNumHidden];
		hiddenDeltas = new double[argNumHidden];
		preWeights1= new double[argNumInputs+1][argNumHidden];
		preWeights2 = new double[argNumHidden+1];
		
		// initialize weights to 0
		for (int i = 0; i < argNumInputs+1; i++){
			for (int j = 0; j < argNumHidden; j++){
				weight1[i][j] = 0;
			}
		}
		
		for (int i = 0; i < argNumHidden+1; i++){
			weights2[i] = 0.0f;
		}
	}
	// The activation function
	public double sigmoid (double x){
		return 2*(1/(1+Math.exp(-x))-0.5);
	}
	
	public double outputFor (double[] X){
		// Calculate the output for the hidden nodes
		double sum;
		for(int j =0; j < argNumHidden; j++){
		sum = weight1[0][j]*bias; //bias term
			for (int i = 1; i < argNumInputs+1; i++){
				sum = sum + weight1[i][j]*X[i-1];
			}
			hiddenOutputs[j] = sigmoid(sum); // output
		}
		// Calculate the output for the output nodes
		sum = weights2[0]*bias; //bias term
		for (int i =1; i < argNumHidden+1; i++){
			sum = sum + weights2[i] *hiddenOutputs[i-1];
		}
		return sigmoid(sum); // output
	}
	
	// get the absolute error
	public double computeAbsoluteError(double[] X, double Qmax, double reward) {
		double CHat = this.outputFor(X);
		if (CHat >1){
			CHat = 1;
		}
		if (CHat <-1){
			CHat = -1;
		}
		
		// the optimal QValue
		double C = CHat + argLearningRate*(reward + argDiscountRate * Qmax - CHat); 
		if (C >1){
			C = 1;
		}
		if (C <-1){
			C = -1;
		}
		
		double error = Math.abs(C - CHat);
		return error;
	}
	// update the weight matrix
	public double weightUpdate(double[] X, double Qmax, double reward){
		double CHat = this.outputFor(X);
		if (CHat >1){
			CHat = 1;
		}
		if (CHat <-1){
			CHat = -1;
		}
		
		// the optimal QValue
		double C = CHat + argLearningRate*(reward + argDiscountRate * Qmax - CHat); 
		if (C >1){
			C = 1;
		}
		if (C <-1){
			C = -1;
		}
		double result = C-CHat;
		// weights update for the hidden nodes
		//System.out.println("A"+out);
		delta = (C-CHat) * 0.5* (1-CHat)*(1+CHat);
		
		for (int i = 0; i< argNumHidden; i++){
		weights2[i+1] = weights2[i+1] + argMomentumTerm*preWeights2[i+1] + argLearningRate * delta * hiddenOutputs[i];
		preWeights2[i+1] =argMomentumTerm*preWeights2[i+1] + argLearningRate*delta*hiddenOutputs[i];// for the momentum
		}
		
		weights2[0] = weights2[0] +argMomentumTerm*preWeights2[0]+ argLearningRate*delta*bias;
		preWeights2[0] = argMomentumTerm*preWeights2[0] + argLearningRate*delta*bias;
		
		// weights update for the input node
		for (int i = 0; i < argNumHidden; i++){
			hiddenDeltas[i] = 0.5* (1-hiddenOutputs[i]*hiddenOutputs[i]) * delta * weights2[i+1];
		}
		
		for (int i = 1; i<argNumInputs+1;i++){
			for(int j = 0; j<argNumHidden; j++){
				weight1[i][j]=weight1[i][j]+argMomentumTerm*preWeights1[i][j]+argLearningRate*hiddenDeltas[j]*X[i-1];
				preWeights1[i][j] = argMomentumTerm * preWeights1[i][j] +argLearningRate*hiddenDeltas[j]*X[i-1]; // for the momentum
			}
		}
	
		for (int j = 0; j< argNumHidden; j++){
			weight1[0][j] = weight1[0][j]+ argMomentumTerm * preWeights1[0][j]+ argLearningRate*hiddenDeltas[j]*bias; // bias term
			preWeights1[0][j] = argMomentumTerm * preWeights1[0][j]+ argLearningRate*hiddenDeltas[j]*bias;
		}
		
		return result;
	}

}