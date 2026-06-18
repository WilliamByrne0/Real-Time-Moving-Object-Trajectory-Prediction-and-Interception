/**************************************************************************************************
 **************************************************************************************************
 *                                                                                                *
 *                 Insect bot mini MKII                                                           *
 *                       by Lumi                                                                  *
 *                 http://www.dfrobot.com                                                         *
 *                            &                                                                   *
 *              http://www.letsmakerobots.com                                                     *
 *                     2014/08/13                                                                 *
 *                         V 0.4                                                                  *
 *                                                                                                *
 *       Sort function based on Steve's bubble sort code at:                                      *
 *       http://www.hackshed.co.uk/arduino-sorting-array-integers-with-a-bubble-sort-algorithm/   *
 *                                                                                                *
 **************************************************************************************************
 *************************************************************************************************/
#include <Servo.h>

// creating the servo objects for front, rear and mid servo
Servo frontLeg;
Servo rearLeg;
Servo midLeg;
// setting the servo angle to 90° for startup
byte frontAngle = 90;
byte rearAngle = 90;
byte midAngle = 90;
// setting the delay value
byte delayWalk = 7;
byte delayTurn = 3;
// Analog sensor pins
int distanceSensor = A1;
int lightSensorLeft = A2;
int lightSensorRight = A0;
// Analog sensor reading
int sensorValue = 0;
int leftEye = 0;
int rightEye = 0;
// Values
int sensorValueLeft = 0;
int sensorValueRight = 0;
int left = 0;
int right = 0;
// Change the following value to decrease or increase the sensitivity. 
// Bigger value for lower sensitivity and smaller value for highter sensitivity 
int lightDifference = 60;
// Decrease the danger value when you want to INCREASE the collision trigger distance.
// Increase the danger value when you want to DECREASE the collision trigger distance 
int danger = 450;  // Increase this value when you want to DECREASE the collision trigger distance 
// Arrays for sensor reading average calculation
int leftReadings[11];
int rightReadings[11];
// Booleans for decision
boolean lightLeft = false;
boolean lightRight = false;
// Setup function
void setup(){
  // serial connection for debugging
  Serial.begin(9600);
  // attaching the servos to their pins  
  frontLeg.attach(3);
  rearLeg.attach(5);
  midLeg.attach(4);
  // move servos to center position -> 90°
  frontLeg.write(frontAngle);
  rearLeg.write(rearAngle);
  midLeg.write(midAngle);
  delay(2000);

}
// Light detection and filtering
void scan()
{  
    int i;
        // Take 5 readings on each sensor
	for (i = 0; i < 11; i = i + 1) {
		// read the value from the left and right sensor:
                sensorValueLeft = analogRead(lightSensorLeft);
                sensorValueRight = analogRead(lightSensorRight);
                // add sensor readings of both sides to their respective array
		leftReadings[i] = sensorValueLeft;
                rightReadings[i] = sensorValueRight;
	    }
// calculate an average value for left and right sensor readings
/*
  How to sort and find the average value. Well, that's something you can do in perhaps 1000 ways. 
  An easy way is a bubble sort algorithm which sorts the 11 values for each, 
  the left and the right light sensor in an ascending order. 
  Then you simply take the middle value as the one you are working with. 
  This way, even if there are 2 or 3 strange readings off the scale, 
  like ZERO (0) or 8565 when only 1023 is possible, which you will eliminate by just taking the middle value.
  This is an example for some strange readings. Maybe your sensor reads 765, 786, 2567, 0, 745, 776, 156, 23, 734, 755, 2333
  Sorting the readings would result in an array like that: 0, 23, 156, 734, 745, 755, 765, 776, 786, 2333, 2567
  Now, if you take the middle value of the array which is 755 you should be on the safe side, at least for this kind of application :-)
*/
// Sorting the left light sensor values
sort(leftReadings,11);
// Un-comment the following 8 lines to serial output the sorted values and the middle value of the array
//  Serial.print("Sorted Left Array: ");
//  for(int i=0; i<11; i++) {
//     Serial.print(leftReadings[i]); 
//     Serial.print(",");
//  }
//  Serial.println("");
//  Serial.println(leftReadings[5]);
//  delay(100);
// This is the final value for the left light sensor
    left = leftReadings[5];
//    Serial.print("Middle Left: ");
//    Serial.println(left);
// Sorting the right light sensor values
sort(rightReadings,11);
// Un-comment the following 8 lines to serial output the sorted values and the middle value of the array
//  Serial.print("Sorted Right Array: ");
//  for(int i=0; i<11; i++) {
//     Serial.print(rightReadings[i]); 
//     Serial.print(",");
//  }
//  Serial.println("");
//  Serial.println(rightReadings[5]);
//  delay(100);
// This is the final value for the right light sensor
     right = rightReadings[5];
//  Serial.print("Middle Right: ");
//  Serial.println(right);
// Un-comment the next 9 lines for serial output of the light sensor readings and their difference
//      Serial.print ("Right: ");
//      Serial.print (right);
//      Serial.print (" I "); 
//      Serial.print ("Left: ");
//      Serial.print (left);
//      Serial.print (" I ");
//      Serial.print ("Difference: ");
//      Serial.println (left-right);
//      delay (130);   
}
// Sort the array ascending
void sort(int a[], int size) {
    for(int i=0; i<(size-1); i++) {
        for(int o=0; o<(size-(i+1)); o++) {
                if(a[o] > a[o+1]) {
                    int t = a[o];
                    a[o] = a[o+1];
                    a[o+1] = t;
                }
        }
    }
}
// Desision depending on the light values
void decision(){
// Call the scan function to provide the sensor values of the left and right light sensor
  scan();
// Compare the values and make a decision according the difference value
if (left > right){
    // subtract right from left value to get a value to work with
    left = left-right;
    // check if that previous calculated value is greater than 50
    if (left > lightDifference) // I will call this "Inside IF 1" in further comments.
    {
      /* If the value is greater than 50 then turn left. Why? the 50? 
         Due the readings it's nearly impossible to get the exact same readings 
         from both sensors in at the same time. So if both sensors always have slightly different readings
         the robot would just wiggle left and right without really going forward. By eliminating slight differences
         and set the accepted difference to a value of 50 it's more likely that the robot 
         actually goes to the function forward.
      */
      lightLeft = true;
      lightRight = false;
    }
    // That happens when the left value in "Inside IF 1" is lower than 50
    else{
      /* Is the calculated value for left lower than 50 then go forward. Why? Is that value lower than 50 then the difference 
         between the light on the left and right sensor are not that great. In this case it should be save to go forward.
      */
      lightLeft = true;
      lightRight = true;
    }
  }
  // That happens when left is lower than right 
  else if (left < right){
    // subtract left from right value to get a value to work with
    right = right-left;
    // check if that previous calculated value is greater than 50
    if (right > lightDifference){  // I will call this "Inside IF 2" in further comments.
      lightLeft = false;
      lightRight = true;
    }
    else{ 
      /* Is the calculated value for right lower than 50 then go forward. Why? Is that value lower than 50 then the difference 
         between the light on the left and right sensor are not that great. In this case it should be save to go forward.
      */
      lightLeft = true;
      lightRight = true;
    }
  }
  /* That "else" happens when none of the above conditions occur and the left and right sensor shows the same readings.
     This condition will probably never occur but we need it anyway.
  */
  else{
    // Go forward without questions
      lightLeft = true;
      lightRight = true;
  }
}
// Walk forward //////////////////////////////////////////////////////////
void forward(){

  int angle_MID_1 = 90 - 20;
   int angle_MID_2 = 90 + 20;

   int angle_Forward_1 = 90 - 40;
   int angle_Forward_2 = 90 + 40;


  for (midAngle = angle_MID_1; midAngle < angle_MID_2; midAngle +=1){
    midLeg.write(midAngle);
    delay(delayWalk);
  }
  for (frontAngle = angle_Forward_2; frontAngle > angle_Forward_1; frontAngle -= 1){
    frontLeg.write(frontAngle);
    rearLeg.write(frontAngle);
    delay(delayWalk);
  }
  for (midAngle = angle_MID_2; midAngle > angle_MID_1; midAngle -=1){
    midLeg.write(midAngle);
    delay(delayWalk);
  }
  for (frontAngle = angle_Forward_1; frontAngle < angle_Forward_2; frontAngle += 1){
    frontLeg.write(frontAngle);
    rearLeg.write(frontAngle);
    delay(delayWalk);
  }
}
// Walk reverse //////////////////////////////////////////////////////////
void reverse(){

  for (midAngle = 70; midAngle < 100; midAngle +=1){
    midLeg.write(midAngle);
    delay(delayWalk);
  }
  for (frontAngle = 50; frontAngle < 120; frontAngle += 1){
    frontLeg.write(frontAngle);
    rearLeg.write(frontAngle);
    delay(delayWalk);
  }

  for (midAngle = 100; midAngle > 70; midAngle -=1){
    midLeg.write(midAngle);
    delay(delayWalk);
  }
  for (frontAngle = 120; frontAngle > 50; frontAngle -= 1){
    frontLeg.write(frontAngle);
    rearLeg.write(frontAngle);
    delay(delayWalk);
  }
}
// Left Turn //////////////////////////////////////////////////////////
void leftTurn(){

  rearLeg.write(90);
  for (midAngle = 70; midAngle < 110; midAngle += 1){
    midLeg.write(midAngle);
    delay(delayTurn); 
  } 
  for (frontAngle = 70; frontAngle < 110; frontAngle +=1){
    frontLeg.write(frontAngle);
    delay(delayTurn); 
  }
  for (rearAngle = 110; rearAngle > 70; rearAngle -=1){
    rearLeg.write(rearAngle);
    delay(delayTurn); 
  }
  for (midAngle = 110; midAngle > 70; midAngle -= 1){
    midLeg.write(midAngle);
    delay(delayTurn); 
  }
  for (frontAngle = 110; frontAngle > 70; frontAngle -=1){
    frontLeg.write(frontAngle);
    delay(delayTurn); 
  }
  for (rearAngle = 70; rearAngle < 110; rearAngle +=1){
    rearLeg.write(rearAngle);
    delay(delayTurn); 
  }
}
// Right Turn //////////////////////////////////////////////////////////
void rightTurn(){

  frontLeg.write(90);
  for (midAngle = 70; midAngle < 110; midAngle += 1){
    midLeg.write(midAngle);
    delay(delayTurn); 
  }
  for (rearAngle = 70; rearAngle < 110; rearAngle +=1){
    rearLeg.write(rearAngle);
    delay(delayTurn); 
  }
  for (frontAngle = 110; frontAngle > 70; frontAngle -=1){
    frontLeg.write(frontAngle);
    delay(delayTurn); 
  }
  for (midAngle = 110; midAngle > 70; midAngle -= 1){
    midLeg.write(midAngle);
    delay(delayTurn); 
  } 
  for (rearAngle = 110; rearAngle > 70; rearAngle -=1){
    rearLeg.write(rearAngle);
    delay(delayTurn); 
  }
  for (frontAngle = 70; frontAngle < 110; frontAngle +=1){
    frontLeg.write(frontAngle);
    delay(delayTurn); 
  }
}
// Stop walking
void stay(){
    frontLeg.write(90);
    midLeg.write(90);
    rearLeg.write(90);
}
//////////////////////////////////////////////////////////////
void loop(){
  forward();
}





