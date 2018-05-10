package LUTtoNeuralNet;
import java.io.*;
import robocode.*;
public class LUTrobot extends AdvancedRobot {
//Initialization
public static double reward;
public static double rewardhitted;
public static double rewardhitrobot;
public static double rewardwall;
public static double rewardmissed;
public static double rewardbullethit;
public static ScannedRobotEvent scannedRobot;
public static int[] oldstate;
public static int[] newstate;
public static int oldaction;
public static int newaction;
public static int NumHeading = 4;
public static int NumTargetBearing = 4;
public static int NumTargetDistance = 3;
public static int NumXlabel = 4;
public static int NumYlabel = 3;
public static int Actions = 12;
public static double LearningRate = 0.2;
public static double DiscountRate = 0.8;
//Intialization using the constructor
public static Qtable Qt = new Qtable(NumHeading, NumTargetBearing, NumTargetDistance, NumXlabel, NumYlabel, Actions);
public static int N = 10000; // number of battles
public static double[] winnum =new double[N]; // used to keep track of the winning history
public static double[] qvv = new double[N]; // used to keep track of the losing history
public static int counterofBattle = -1; // counter
public static double epsi = 0.1; // the epsilon in epsilon greedy
public void run() {
setAdjustRadarForGunTurn(true);
setAdjustRadarForRobotTurn(true);
setAdjustGunForRobotTurn(true);
rewardhitted = 0;
rewardhitrobot = 0;
rewardmissed = 0;
rewardwall = 0;
rewardbullethit = 0;
counterofBattle = counterofBattle +1; // counter plus one
turnRadarRight(360);
oldstate = Qt.quanQ(getHeading(),scannedRobot.getBearing(),scannedRobot.getDistance(),getX(),getY());
oldaction = Qt.pickAction(oldstate[0],oldstate[1],oldstate[2],oldstate[3],oldstate[4],epsi);
this.takeAction(oldaction, scannedRobot.getDistance());
while(true) {
setAdjustRadarForGunTurn(true);
setAdjustRadarForRobotTurn(true);
setAdjustGunForRobotTurn(true);
turnRadarRight(360);
newstate = Qt.quanQ(getHeading(),scannedRobot.getBearing(),scannedRobot.getDistance(),getX(),getY());
newaction =Qt.pickAction(newstate[0],newstate[1],newstate[2],newstate[3],newstate[4],0);
reward = rewardhitted + rewardhitrobot + rewardwall + rewardbullethit + rewardmissed;
this.updateQ(oldstate, oldaction, newstate, newaction, reward);
rewardhitted = 0;
rewardhitrobot = 0;
rewardmissed = 0;
rewardwall = 0;
rewardbullethit = 0;
oldstate = newstate;
oldaction = Qt.pickAction(newstate[0],newstate[1],newstate[2],newstate[3],newstate[4],epsi);
this.takeAction(oldaction, scannedRobot.getDistance());
}
}
public void onScannedRobot(ScannedRobotEvent e) {
scannedRobot = e;
this.targetOppo(getHeading(), getGunHeading(), e.getBearing());
}
public void updateQ(int[] oldstate, int oldaction, int[] newstate, int newaction, double reward )
{ // function to update the Q table
double a = Qt.Q[oldstate[0]][oldstate[1]][oldstate[2]][oldstate[3]][oldstate[4]][oldaction];
double b = Qt.Q[newstate[0]][newstate[1]][newstate[2]][newstate[3]][newstate[4]][newaction];
Qt.Q[oldstate[0]][oldstate[1]][oldstate[2]][oldstate[3]][oldstate[4]][oldaction] = (1-LearningRate)*a+LearningRate*reward + LearningRate*DiscountRate*b;
}
public void onHitByBullet(HitByBulletEvent e) {
rewardhitted = -3;
}
public void onHitWall(HitWallEvent e) {
rewardwall = -3;
}
public void onHitRobot(HitRobotEvent e){
rewardhitrobot = -2;
}
public void onBulletHit(BulletHitEvent e){
rewardbullethit = 5;
}
public void onBulletMissed(BulletMissedEvent e) {
rewardmissed = -1;
}
public void takeAction(int counter, double distance){
// 6 actions
switch(counter){
case (0): {
fire(3);
setAhead(90);
execute();
break;
}
case (1): {
fire(3);
setTurnLeft(90);
setAhead(90);
execute();
break;
}
case (2): {
fire(3);
setTurnRight(90);
setAhead(90);
execute();
break;
}
case (3): {
setFire(3);
setBack(90);
execute();
break;
}
case (4): {
fire(3);
setTurnLeft(90);
setBack(90);
execute();
break;
}
case (5): {
fire(3);
setTurnRight(90);
setBack(90);
execute();
break;
}
case (6): {
setAhead(90);
execute();
break;
}
case (7): {
setTurnLeft(90);
setAhead(90);
execute();
break;
}
case (8): {
setTurnRight(90);
setAhead(90);
break;
}
case (9): {
back(90);
break;
}
case (10): {
setTurnLeft(90);
setBack(90);
break;
}
case (11): {
setTurnRight(90);
setBack(90);
break;
}
}
}
public void targetOppo(double robotHeading, double robotGunHeading, double TargetBearing){
//find the smallest angle between the gun and the targe and move the gun to target the enemy
setAdjustRadarForGunTurn(true);
setAdjustGunForRobotTurn(true);
double stbullet;
if ((robotHeading>180)&&(TargetBearing<0))
{stbullet = robotHeading+TargetBearing;
}else if ((robotHeading>180)&&(TargetBearing>0))
{ if ( TargetBearing-(360-robotHeading)>0){
stbullet = TargetBearing-(360-robotHeading);}
else{stbullet = TargetBearing +robotHeading;}
}
else if((robotHeading<180)&&(TargetBearing>0))
{stbullet = TargetBearing + robotHeading;}
else{if (getHeading()+TargetBearing<0){stbullet = 360-((-TargetBearing)-robotHeading);}
else{stbullet = getHeading()+TargetBearing;}
}
if (robotGunHeading-stbullet>180){turnGunRight(360-robotGunHeading+stbullet); }
else if ((robotGunHeading-stbullet<180)&&(robotGunHeading-stbullet>0)){turnGunLeft(robotGunHeading-stbullet);}
else if (stbullet-robotGunHeading>180) {turnGunLeft(360-stbullet+robotGunHeading);}
else {turnGunRight(stbullet-robotGunHeading);}
}
public void onWin(WinEvent event){
winnum[counterofBattle] = 1.0;
qvv[counterofBattle] = Qt.Q[1][2][0][1][1][5];
}
public void onDeath(DeathEvent event){
qvv[counterofBattle] =Qt.Q[1][2][0][1][1][5];
}
public void onBattleEnded(BattleEndedEvent event){
//Print to file
out.println("The battle has ended");
try{
FileWriter cer = new FileWriter("qlearningfinal.txt"); // write to a .txt. file
for (int i = 0; i<N; i++){
cer.write(Double.toString(winnum[i])+"\r\n");
cer.flush();
}
cer.close();
FileWriter qv = new FileWriter("qvalueqlearning0.1.txt");
for (int i = 0; i<N; i++){
qv.write(Double.toString(qvv[i])+"\r\n");
qv.flush();
}
qv.close();
saveQ();
}catch (Exception e){}
}
//save Q matrix
public void saveQ() throws Exception {
String buff = new String();
BufferedWriter bw = new BufferedWriter(new FileWriter("q.txt"));
for (int i = 0; i < 19; i++) {
for (int j = 0; j < 8; j++) {
}
}
for (int k1 = 0; k1 < NumHeading; k1++)
{for (int k2 = 0; k2 < NumTargetBearing; k2++)
{for (int k3 = 0; k3< NumTargetDistance;k3++)
{for(int k4 = 0; k4< NumXlabel;k4++)
{for(int k5 = 0; k5< NumYlabel;k5++)
{for(int k6 = 0; k6< Actions;k6++)
buff = Double.toString(Qt.Q[k1][k2][k3][k4][k5][k6]);
bw.write(buff);
bw.newLine();
}}}}}
bw.close();
}
}