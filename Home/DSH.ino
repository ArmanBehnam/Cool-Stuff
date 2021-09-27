/* code for no activity:

// HOMESHH - Capstone Prototype
// by Niam Moltta

const int Green =12;
const int Blue =11;
const int Red =13;

void setup() 
{

  pinMode(Green, OUTPUT);
  pinMode(Blue, OUTPUT);
  pinMode(Red, OUTPUT);  
      
}

void loop() {
 
  //Red
  digitalWrite(Red, HIGH);
  digitalWrite(Blue, LOW);
  digitalWrite(Green, LOW);
  delay(1000);
  //Green
  digitalWrite(Red, LOW);
  digitalWrite(Blue, LOW);
  digitalWrite(Green, HIGH);
  delay(1000);
  //Blue
  digitalWrite(Red, LOW);
  digitalWrite(Blue, HIGH);
  digitalWrite(Green, LOW);
  delay(1000);
  //Magenta
  digitalWrite(Red, HIGH);
  digitalWrite(Blue, HIGH);
  digitalWrite(Green, LOW);
  delay(1000);
  //Cyan
  digitalWrite(Red, LOW);
  digitalWrite(Blue, HIGH);
  digitalWrite(Green, HIGH);
  delay(1000);
  //Yellow
  digitalWrite(Red, HIGH);
  digitalWrite(Blue, LOW);
  digitalWrite(Green, HIGH);
  delay(1000);
}

*/

// Activity Code:**********************************************************************:

// HomeShh - IoT Specialization Capstone Project
// Smart Home for deaf people
// by Niam Moltta.

const int G =12; 
const int B =11; 
const int R =13; 
const int S = 10; 
const int A = 8; 
const int C = 7; 
const int D = 6; 
const int K = 5; 
const int M = 9; 
const int E = 4;
const int F = 3;
const int L = 2;
const int Y = A3;
const int P = A2;
const int V = A0;
const int W = A1;

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
  
   //Car present
  long car = digitalRead(E);
 
 if(car == HIGH) {
   
   digitalWrite(L, HIGH);
   delay(1000); //time for simulation purposes
  
 }
  
  else
    
   {       
    
    digitalWrite(L, LOW);
   
  }
   
  //Fire simulation
  s = analogRead(V);
  Serial.println("Smoke Signal K");
  Serial.println(s);
  
  if (s > 20) {
    
    Fire();
    tone(S, 1000);
    delay(100);
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
  
  if (ms > 20) {
    
    Fire();
    FireS();
    tone(S, 1000);
    delay(50);
    analogWrite(M, 255);
    
  }
  
  else {
    
    noTone(S);
    Nothing();
    OffS();
  }
  
  //Phone simulation
  
  c = analogRead(Y);
  Serial.println("call");
  Serial.println(c);
  
  if (c == 1023) {
    
    Phone();
    PhoneS();
    tone(S, 700);
    delay(50);
    analogWrite(M, 255);
    
  }
  
  else {
    
    noTone(S);
    Nothing();
    OffS();
    
  }
  
  //Doorbell simulation
  db = analogRead(P);
  Serial.println("Signal");
  Serial.println(db);
  if (db == 1023) {
    
    Doorbell();
    DoorbellS();
    tone(S, 330);
    delay(35);
    tone(S, 261);
    delay(35);
    analogWrite(M, 255);
    
  }
  
  else {
    
    noTone(S);
    Nothing();
    OffS();
    
  }
  
  //Burglar Simulation

long b = digitalRead(K);
 
 if(b == HIGH) {
   
   Robbery();
   RobberyS();
   tone(S, 440, 1000);
   digitalWrite(F, HIGH);
   Serial.println(b);
   delay(50);
   analogWrite(M, 255);
   
  }
  
  else
    
   {  
    
    OffS();
    noTone(S);
    
  }
 
}

//RGB**************************//

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
  //MAGENTA (OPTIONAL WITH RASPBERRY PI)
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
 //END MAIN RGB's**************************

//BEDROOM FAN ALARM RGB

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
  //MAGENTA (optional for RASPBERRY PI, since this simulator doesn't have microphones)
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

/* ************************END************************************************************/

