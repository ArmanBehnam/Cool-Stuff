// HomeShh - IoT Specialization Capstone Project
// Smart home for deaf people
// Apache License
// Copyright 2017 Niam Moltta

// Simulator "Заткнись дом": https://www.tinkercad.com/things/afNAJcOk2rS-sim-zatknis-dom-/editel?sharecode=b4-hxomIlO_lSzW6PpGuVMrkRsW0HRKllROT_KIAYx4=

// Copy the circuit from the simulator above and use the code below to perform the simulation.

/* 
The simulation includes a house (drawn as cables), you can see the reference at the README.md file in this repository.
Once you are running the simulation, you can activate/deactivate the sensors manually and see how the HomeShh works.
There are gif files at the README.md where you can see a couple of demostrations. 
*/

// Light settings:
const int G =12;  // green
const int B =11;  // blue
const int R =13;  // red
const int S = 10;  // buzzer
const int A = 8; // red
const int C = 7; // green
const int D = 6; // blue
const int K = 5; // PIR kitchen and garage
const int M = 9; // fan
const int E = 4; // PIR garage light
const int F = 3; // relay outside lights
const int L = 2; // relay garage light
const int Y = A3; // phone
const int P = A2; // doorbell
const int V = A0; // fire in the kitchen
const int W = A1; // fire in the garage

int s = 0; 
int c = 0; 
int ms = 0; 
int db = 0; 

void setup() 
{
  pinMode(G, OUTPUT); 
  pinMode(B, OUTPUT); 
  pinMode(R, OUTPUT); 
  pinMode(S, OUTPUT); 
  pinMode(M, OUTPUT); 
  pinMode(A, OUTPUT);
  pinMode(C, OUTPUT);
  pinMode(D, OUTPUT);
  pinMode(F, OUTPUT);
  pinMode(L, OUTPUT); 
  Serial.begin(9600);
  pinMode(K, INPUT); 
  pinMode(V, INPUT); 
  pinMode(E, INPUT); 
  pinMode(W, INPUT); 
  pinMode(Y, INPUT);
  pinMode(P, INPUT); 
  
  Nothing();
  OffS();
  digitalWrite(M, LOW);
  digitalWrite(L, LOW);
  digitalWrite(F, LOW);
  digitalWrite(S, LOW);
  noTone(S);
  
}

void loop() {
  
   // Car in the garage (when motion sensor is activated in simulator)
  
  long car = digitalRead(E);
 
 if(car == HIGH) {
   
   digitalWrite(L, HIGH);
   delay(5000); //time for simulation purposes
  
 }
  
  else
    
   {       
    
    digitalWrite(L, LOW);
   
  }
   
  //Fire simulation (smoke detector is activated in the simulator)
  
  s = analogRead(V);
  Serial.println("Smoke Signal K");
  Serial.println(s);
  
  if (s > 100) {
    
    Fire();
    FireS();
    tone(S, 1000);
    delay(2000);
    analogWrite(M, 255);
    
  }
  
  else {
    
    noTone(S);
    Nothing();
    OffS();
    
  }
  
  ms = analogRead(W);
  Serial.println("Smoke Signal G");
  Serial.println(ms);
  
  if (ms > 100) {
    
    Fire();
    FireS();
    tone(S, 1000);
    delay(2000);
    analogWrite(M, 255);
    
  }
  
  else {
    
    noTone(S);
    Nothing();
    OffS();
  }
  
  // Phone simulation (pushbutton is pressed in the simulator)
  
  c = analogRead(Y);
  Serial.println("call");
  Serial.println(c);
  
  if (c == 1023) {
    
    Phone();
    PhoneS();
    tone(S, 700);
    delay(1000);
    analogWrite(M, 255);
    
  }
  
  else {
    
    noTone(S);
    Nothing();
    OffS();
    
  }
  
  // Doorbell simulation (pushbutton is pressed in the simulator)
  
  db = analogRead(P);
  Serial.println("Signal");
  Serial.println(db);
  if (db == 1023) {
    
    Doorbell();
    DoorbellS();
    tone(S, 330);
    delay(350);
    tone(S, 261);
    delay(350);
    analogWrite(M, 255);
    
  }
  
  else {
    
    noTone(S);
    Nothing();
    OffS();
    
  }
  
  // Burglar Simulation (one or both of the motion sensors at the house entrances are activated in the simulator)

long b = digitalRead(K);
 
 if(b == HIGH) {
   
   Robbery();
   RobberyS();
   tone(S, 440, 5000);
   digitalWrite(F, HIGH);
   Serial.println(b);
   delay(200);
   analogWrite(M, 255);
   
  }
  
  else
    
   {  
    
    OffS();
    noTone(S);
    
  }
 
}

// Functions for RGB lights:

void Fire() {
  //RED!
  digitalWrite(R, HIGH);
  digitalWrite(B, LOW);
  digitalWrite(G, LOW);
}
void Doorbell() {
  //GREEN
  digitalWrite(R, LOW);
  digitalWrite(B, LOW);
  digitalWrite(G, HIGH);
}
void Robbery() {
  //BLUE
  digitalWrite(R, LOW);
  digitalWrite(B, HIGH);
  digitalWrite(G, LOW);
}
void Noise() { //ABOVE 120db!!!
  //MAGENTA 
  digitalWrite(R, HIGH);
  digitalWrite(B, HIGH);
  digitalWrite(G, LOW);
}
void Phone() {
  //CYAN
  digitalWrite(R, LOW);
  digitalWrite(B, HIGH);
  digitalWrite(G, HIGH);
}
void Nothing() {
  //YELLOW
  digitalWrite(R, HIGH);
  digitalWrite(B, LOW);
  digitalWrite(G, HIGH);
}

// BEDROOM FAN ALARM RGB (In real life this would have a reset button next to the bed so the fan could be reset by user after waking up)

 void FireS() {
  //RED!
  digitalWrite(A, HIGH);
  digitalWrite(D, LOW);
  digitalWrite(C, LOW);
}
void DoorbellS() {
  //GREEN
  digitalWrite(A, LOW);
  digitalWrite(D, LOW);
  digitalWrite(C, HIGH);
}
void RobberyS() {
  //BLUE
  digitalWrite(A, LOW);
  digitalWrite(D, HIGH);
  digitalWrite(C, LOW);
}
void NoiseS() { //ABOVE 120db!!!
  //MAGENTA (the simulator doesn't have microphones)
  digitalWrite(A, HIGH);
  digitalWrite(D, HIGH);
  digitalWrite(C, LOW);
}
void PhoneS() {
  //CYAN
  digitalWrite(A, LOW);
  digitalWrite(D, HIGH);
  digitalWrite(C, HIGH);
}
void NothingS() {
  //YELLOW
  digitalWrite(A, HIGH);
  digitalWrite(D, LOW);
  digitalWrite(C, HIGH);
}
 void OffS() {
  digitalWrite(A, LOW);
  digitalWrite(D, LOW);
  digitalWrite(C, LOW);
} 

// END of functions
