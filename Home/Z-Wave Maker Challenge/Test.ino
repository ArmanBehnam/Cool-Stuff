// HomeShh TEST - IoT Specialization Capstone Project
// Z-Wave Challenge
// by Niam Moltta.
// December 2017

int Gr =12;
int Bl =11;
int Re =13;
int To = 6;
int V = A0;

float s = 0;

void setup() 
{
  pinMode(Gr, OUTPUT); //green
  pinMode(Bl, OUTPUT); //blue
  pinMode(Re, OUTPUT); //red
  pinMode(To, OUTPUT); //tone
 
  Serial.begin(9600);
  pinMode(V, INPUT_PULLUP);

  digitalWrite(To, HIGH);
  digitalWrite(Gr, LOW);
  digitalWrite(Re, LOW);
  digitalWrite(Bl, LOW);
 
}
void loop() {
  
  s = analogRead(V);
  Serial.println(s, HEX);

  if (s <= 226) {

    Door();
    tone(To, 1000);
    delay(80);
    tone(To, 500);
    delay(100);
  }
  else {
  noTone(To);
  Nothing();

  }
  
}

void Door() {
  digitalWrite(Re, HIGH);
  digitalWrite(Bl, HIGH);
  digitalWrite(Gr, LOW);
}

void Nothing() {
  digitalWrite(Re, HIGH);
  digitalWrite(Gr, HIGH);
  digitalWrite(Bl, HIGH);
}


