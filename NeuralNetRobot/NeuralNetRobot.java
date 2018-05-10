package NeuralNetRobot;
import java.io.*;
import java.util.Random;

import robocode.*;
import Sarb.*;

import static robocode.util.Utils.normalRelativeAngleDegrees;

public class NeuralNetRobot extends AdvancedRobot implements LUTInterface  {
	public double reward,rewardHitByBullet,rewardHitRobot,rewardHitWall,rewardBulletHit;
	
	public static int argHeading = 4;
	public static int argEnemyBearing = 6;
	public static int argDist = 5;
	public static int argPositionX = 4;
	public static int argPositionY = 3;
	public static int argActions = 6;
	public static int argNumInputs = 6;
	public static int argNumHidden = 12;
	
	public static int totalRounds = 8000; // number of rounds	
	public static int threshold = 8000;
	public static int countBattle = -1;   
	public static int action = 0;

	public static double argLearningRate = 0.01;
	public static double argDiscountRate = 0.8;
	public static double argMomentumTerm = 0.01f;	
	public static double epsilon = 0.05;// used in epsilon greedy 
	public static double qMaxValue = 0.0;
	public static double absError1 = 0.00;
	public static double absError2 = 0.00;
	public static double absError3 = 0.00;
	public static double absError = 0.00;


	public static double[] state = new double[6];
	public static double[] newState = new double[6];
	public static double[] winHistory =new double[totalRounds]; // winning history
	public static double[] errorHistory1 =new double[totalRounds]; // winning history
	public static double[] errorHistory2 =new double[totalRounds]; // winning history
	public static double[] errorHistory3 =new double[totalRounds]; // winning history
	public static double[] errorHistory =new double[totalRounds]; // winning history

	
	public static ScannedRobotEvent scannedRobot;
	public static LUTable lutTable = new LUTable(argHeading, argEnemyBearing, argDist, argPositionX, argPositionY, argActions);
	
	// newly added
	public static BP BPNeuralNet = new BP();
	public static double[] X = 	new double[argNumInputs+1];

	
	public void run() {
		
		absError1 = 0;
		absError2 = 0;
		absError3 = 0;
		absError = 0;

		setAdjustRadarForGunTurn(true);
		setAdjustRadarForRobotTurn(true);
		setAdjustGunForRobotTurn(true);
		turnRadarRight(360);
		
		countBattle = countBattle + 1; 
		setState(getHeading(),scannedRobot.getBearing(),scannedRobot.getDistance(),getX(),getY());
		
		// to initialize BPNeuralNet when using it for the first time
		if (countBattle == 0){
			BPNeuralNet.setArgNumInputs(argNumInputs);
			BPNeuralNet.setArgNumHidden(argNumHidden); 
			BPNeuralNet.setArgLearningRate(argLearningRate); 
			BPNeuralNet.setArgMomentumTerm(argMomentumTerm);
			BPNeuralNet.setArgDiscountRate(argDiscountRate);
			BPNeuralNet.initializeWeights();
		}		

		while(true) {				

			turnRadarRight(360);
			
			rewardHitByBullet = 0;
			rewardHitRobot = 0;
			rewardHitWall = 0;
			rewardBulletHit = 0;			

			// to get the cumulative win rate and turn off exploration after converge
			if(countBattle >= threshold) {
				action = this.selectAction(state[0],state[1],state[2],state[3],state[4],0);		
				this.takeAction(action); 
				setNewState(getHeading(),scannedRobot.getBearing(),scannedRobot.getDistance(),getX(),getY());  
			}
			else {
				// take the action that returns the highest Q value
				action = this.selectAction(state[0],state[1],state[2],state[3],state[4],epsilon);
				this.takeAction(action);   
				
				// get new State
				setNewState(getHeading(),scannedRobot.getBearing(),scannedRobot.getDistance(),getX(),getY());  
				
				// get the maximized Q-value				
				qMaxValue = this.maxValue(newState[0], newState[1],newState[2],newState[3],newState[4]);
				
				//get immediate reward
				reward = rewardHitByBullet + rewardHitRobot + rewardHitWall + rewardBulletHit;
				
				if(countBattle < threshold) {
					setX(state,action);
					absError = absError + BPNeuralNet.computeAbsoluteError(X, qMaxValue, reward);
					computeError(X,qMaxValue,reward);
					BPNeuralNet.weightUpdate(X,qMaxValue,reward);
				}
				state = newState;
			}
		}
	}
	
	public static void computeError(double[] X, double Qmax, double reward) {
		if(X[0]==0) {
			if(X[1]==1) {
				if(X[2]==2) {
					if(X[3]==1) {
						if(X[4]==2) {
							if(X[5]==1) {
								if(X[6]== 2) {
									absError1 = absError1 + BPNeuralNet.computeAbsoluteError(X, qMaxValue, reward);
								}

							}
						}
					}
				}
			}
		}
		
		if(X[0]==1) {
			if(X[1]==2) {
				if(X[2]==3) {
					if(X[3]==1) {
						if(X[4]==2) {
							if(X[5]==2) {
								if(X[6]==3) {
									absError2 = absError2 + BPNeuralNet.computeAbsoluteError(X, qMaxValue, reward);
								}

							}
						}
					}
				}
			}
		}
		
		if(X[0]==2) {
			if(X[1]==4) {
				if(X[2]==2) {
					if(X[3]==3) {
						if(X[4]==1) {
							if(X[5]==1) {
								if(X[6]==4) {
									absError3 = absError3 + BPNeuralNet.computeAbsoluteError(X, qMaxValue, reward);
								}
							}
						}
					}
				}
			}
		}

	}
	// set current state 
	public static void setState(double argHeading, double argEnemyBearing, double argDist, double argPositionX, double argPositionY) {
		state[0] = argHeading;
		state[1] = argEnemyBearing;
		state[2] = argDist;
		state[3] = argPositionX;
		state[4] = argPositionY;
	}
	
	// set new state 
	public static void setNewState(double argHeading, double argEnemyBearing, double argDist, double argPositionX, double argPositionY) {
		state[0] = argHeading;
		state[1] = argEnemyBearing;
		state[2] = argDist;
		state[3] = argPositionX;
		state[4] = argPositionY;
	}
	
	// set new state 
	public static void setX(double[] state, double argActions) {
		X[0] = state[0];
		X[1] = state[1];
		X[2] = state[2];
		X[3] = state[3];
		X[4] = state[4];
		X[5] = state[5];
		X[6] = argActions;

	}

	// select the action according to epsilon greedy
	public int selectAction(double i, double j, double k, double m, double n, double epsilon){
		int result = 0;
		Random generator1 = new Random();
		Random generator2 = new Random();
		if(generator1.nextDouble() < epsilon){
			result = generator2.nextInt(argActions);			
		}
		else{
			result = this.maxValue(i, j, k, m, n);
		}
		return result;
	}
		
	// find the action number corresponding to the maximum value in the state-action pair table corresponding to a certain state
		public int maxValue(double i, double j, double k, double m, double n){			
			double max =  -100;
			int counter1 = 0;
			int action = 0;
			double current = 0.0;
			for (counter1 = 0; counter1 < argActions; counter1++){
				double[] X = {i, j, k, m, n, counter1};
				current = BPNeuralNet.outputFor(X);
				if (current > max){
					max = current;
					action= counter1;
				}		   
			}

			return action;
		}
	
	public void onScannedRobot(ScannedRobotEvent e) {
		scannedRobot = e;
		
		// Calculate exact location of the robot
		double absoluteBearing = getHeading() + e.getBearing();
		double bearingFromGun = normalRelativeAngleDegrees(absoluteBearing - getGunHeading());

		// If it's close enough, fire!
		if (Math.abs(bearingFromGun) <= 3) {
			turnGunRight(bearingFromGun);
			// We check gun heat here, because calling fire()
			// uses a turn, which could cause us to lose track
			// of the other robot.
			if (getGunHeat() == 0) {
				fire(Math.min(3 - Math.abs(bearingFromGun), getEnergy() - .1));
			}
		} // otherwise just set the gun to turn.
		// Note:  This will have no effect until we call scan()
		else {
			turnGunRight(bearingFromGun);
		}
		// Generates another scan event if we see a robot.
		// We only need to call this if the gun (and therefore radar)
		// are not turning.  Otherwise, scan is called automatically.
		if (bearingFromGun == 0) {
			scan();
		}
		
	}
	
	public void onHitByBullet(HitByBulletEvent e) {
		rewardHitByBullet = -5 ;
	}

	public void onHitWall(HitWallEvent e) {
        rewardHitWall = -2;
	}

	public void onHitRobot(HitRobotEvent e){
		rewardHitRobot = -3;
	}

	public void onBulletHit(BulletHitEvent e){
		rewardBulletHit = 4 ;
	}

	public void takeAction(int counter){
		// 6 actions		
		
		switch(counter){
			case (0): {
				ahead(90);
				break;				
			}
			case (1): {
				back(90);					
				break;
			}
			case (2): {
				setAhead(90);
				setTurnLeft(90);
				execute();
				break;
			}
			case (3): {
				setAhead(90);
				setTurnRight(90);
				execute();
				break;
			}
			case (4): {
				setBack(90);
				setTurnLeft(90);
				execute();
				break;
			}
			case (5): {
				setBack(90);
				setTurnRight(90);
				execute();
				break;
			}
		}
			
	}
	
	public void onWin(WinEvent event){
		winHistory[countBattle] = 1.0;
		errorHistory[countBattle]= absError;
		errorHistory1[countBattle]= absError1;
		errorHistory2[countBattle]= absError2;
		errorHistory3[countBattle]= absError3;
	}
	
	public void onDeath(DeathEvent event){
		errorHistory[countBattle]= absError;
		errorHistory1[countBattle]= absError1;
		errorHistory2[countBattle]= absError2;
		errorHistory3[countBattle]= absError3;
	}	
	
	// Store information to file system
	public void onBattleEnded(BattleEndedEvent event){
		out.println("End of the battle");
		
		//Save Lookup Table to file system
		File fileLUT = new File("LUT.dat");
		save(fileLUT);
		
		File fileWinRate = new File("winRate.txt");
		saveWinRate(fileWinRate);
		
		File fileWinRate2 = new File("winRate2.txt");
		saveWinRate2(fileWinRate2);
					
		File fileTotalRuns = new File("totalRuns.txt");
		saveTotalRounds(fileTotalRuns);
		
		File fileAbsError = new File("abserror.txt");
		saveAbsError1(fileAbsError);		
		File fileAbsError1 = new File("abserror1.txt");
		saveAbsError1(fileAbsError1);
		File fileAbsError2 = new File("abserror2.txt");
		saveAbsError1(fileAbsError2);
		File fileAbsError3 = new File("abserror3.txt");
		saveAbsError1(fileAbsError3);
		
	}

	@Override
	public double outputFor(double[] X) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double train(double[] X, double argValue) {
		// update the LUT table via learning
		return 0;
	}
	
	// store LUT.dat into the file system
	@Override
	public void save(File argFile) {
		// iterate every element in Lookup table and save them to file system
		try{
			FileWriter writer = new FileWriter(argFile); 
			for (int i1 = 0; i1 < argHeading; i1++){
				for (int i2 = 0; i2 < argEnemyBearing; i2++){
					for (int i3 = 0; i3< argDist;i3++){
						for(int i4 = 0; i4< argPositionX;i4++){
							for(int i5 = 0; i5< argPositionY;i5++){
								for(int i6 = 0; i6< argActions;i6++){
									writer.write(Double.toString(lutTable.getElement(i1, i2, i3, i4, i5, i6))+"\r\n");
								}
							}
						}
					}
				}
			}
			writer.flush();
			writer.close();
		}catch (Exception e){
			out.println("exception in save function");
		}			
		
	}
	
	// store win rate to file system		
		public void saveAbsError(File argFile) {
					
			try{			
				FileWriter writer = new FileWriter(argFile,false); 
				for (int i = 0; i<totalRounds; i++){
					writer.write(Double.toString(errorHistory[i])+"\r\n");
					writer.flush();
				}
				writer.close();
			}catch (Exception e){				
				out.println("exception in saveAbsError function");	
			}			
		}
		
	// store win rate to file system		
	public void saveAbsError1(File argFile) {
				
		try{			
			FileWriter writer = new FileWriter(argFile,false); 
			for (int i = 0; i<totalRounds; i++){
				writer.write(Double.toString(errorHistory1[i])+"\r\n");
				writer.flush();
			}
			writer.close();
		}catch (Exception e){				
			out.println("exception in saveAbsError function");	
		}			
	}
	
	// store win rate to file system		
	public void saveAbsError2(File argFile) {
				
		try{			
			FileWriter writer = new FileWriter(argFile,false); 
			for (int i = 0; i<totalRounds; i++){
				writer.write(Double.toString(errorHistory2[i])+"\r\n");
				writer.flush();
			}
			writer.close();
		}catch (Exception e){				
			out.println("exception in saveAbsError function");	
		}			
	}		
	
	// store win rate to file system		
	public void saveAbsError3(File argFile) {
				
		try{			
			FileWriter writer = new FileWriter(argFile,false); 
			for (int i = 0; i<totalRounds; i++){
				writer.write(Double.toString(errorHistory3[i])+"\r\n");
				writer.flush();
			}
			writer.close();
		}catch (Exception e){				
			out.println("exception in saveAbsError function");	
		}			
	}		
	
	// store abs error to file system		
	public void saveWinRate(File argFile) {
		
		// Calculate the total win rate
		double totalWins = 0;
		double[] winRate =new double[totalRounds];
		double[] totalBattles = new double[totalRounds];

		for (int i = 0; i<totalRounds; i++){
				totalWins = totalWins + winHistory[i];
				winRate[i] = totalWins * 100.0f/(i+1);
				totalBattles[i] = i;
			}
		
		try{			
			FileWriter writer = new FileWriter(argFile,false); 
			for (int i = 0; i<totalRounds; i++){
				writer.write(Double.toString(winRate[i]/100)+"\r\n");
				writer.flush();
			}
			writer.close();
		}catch (Exception e){				
			out.println("exception in saveWinRate function");	
		}			
	}	
			
	// store win rate for every few rounds to file system		
	public void saveWinRate2(File argFile) {
		// compute the average win rate for every 50 battles
	 	int numforoneRound = 50;
	 	int N2 = totalRounds/numforoneRound;
		double totalWins2 = 0;
		double[] winRate2 =new double[N2];
		double[] totalBattles2 = new double[N2];
		for (int i = 0; i<N2; i++){
			totalWins2 = 0;
			for (int j=0; j<numforoneRound;j++) {
		       totalWins2 = totalWins2 + winHistory[j+i*numforoneRound];
			}
	       winRate2[i] = totalWins2 * 100.0f/numforoneRound;
	       totalBattles2[i] = i+1;
		}
	 
	 try{  
		    FileWriter writer = new FileWriter(argFile); // write to a .txt. file
		    for (int i = 0; i<N2; i++){
		    	writer.write(Double.toString(winRate2[i]/100)+"\r\n");
		    	writer.flush();
			 }
		    writer.close();
	 	}catch (Exception e){				
			out.println("exception in saveWinRate2 function");	
		}
		
	}			
	
	
	// store the total runs to file system		
	public void saveTotalRounds(File argFile) {	
		try{			
			FileWriter writer = new FileWriter(argFile); 
			for (int i = 0; i<totalRounds; i++){
				writer.write(Integer.toString(i+1)+"\r\n");
				writer.flush();
			}
			writer.close();
		}catch (Exception e){				
			out.println("exception in saveTotalRounds function");	
		}
		
	}
				

	@Override
	public void load(String argFileName) throws IOException {
		return;
	}

	@Override
	public void initialiseLUT() {
		lutTable.initialize();			
	}

	@Override
	public int indexFor(double[] X) {
		// TODO Auto-generated method stub
		return 0;
	}
}
