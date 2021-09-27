// HomeShh - Capstone Project
// Copyright 2017 Niam Moltta
// License: Apache License

const int G =12;
const int B =11;
const int R =13;
const int S = 7;
const int T = 6;
const int V = A0;
const int Y = A3;
const int W = A1;
int s = 0;
int c = 0;
int ms = 0;

void setup() 
{
  pinMode(G, OUTPUT); //green
  pinMode(B, OUTPUT); //blue
  pinMode(R, OUTPUT); //red
  pinMode(S, OUTPUT); //!!!!!
  pinMode(T, OUTPUT); //!!!!
  Serial.begin(9600);
  pinMode(Y, INPUT); //?...
  pinMode(V, INPUT); //??...
  Nothing();
}
void loop() {
  
  s = analogRead(V);
  Serial.println("Signal");
  Serial.println(s);
  
  if ((s > 400) && (s < 800)) {
    Red();
    tone(T, 1000);
    delay(500);
  }
  else {
    noTone(T);
    Nothing();
  }
  ms = analogRead(W);
  Serial.println("Signal");
  Serial.println(ms);
  
  if ((ms > 400) && (ms < 800)) {
    Red();
    tone(T, 1000);
    delay(500);
  }
  else {
    noTone(T);
    Nothing();
  }
  c = analogRead(Y);
  Serial.println("Signal");
  Serial.println(c);
  if (c < 10) {
    Cian();
    tone(S, 700);
    delay(500);
  }
  else {
    noTone(S);
    Nothing();
  }

}

void Red() {
  
  digitalWrite(R, HIGH);
  digitalWrite(B, LOW);
  digitalWrite(G, LOW);
}
void Cian() {
  
  digitalWrite(R, LOW);
  digitalWrite(B, HIGH);
  digitalWrite(G, HIGH);
}
void Nothing() {
  
  digitalWrite(R, HIGH);
  digitalWrite(G, HIGH);
  digitalWrite(B, LOW);
}
