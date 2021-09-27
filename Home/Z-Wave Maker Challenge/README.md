# Project updates

<p align="center"><img src="https://user-images.githubusercontent.com/22894897/34063866-218c8e00-e1b2-11e7-93e3-6c520a876965.gif" width="57%"/><img src="https://user-images.githubusercontent.com/22894897/33862841-06f17a0a-dea2-11e7-9b97-01cd8a1dce81.JPG" width="43%"/></p>

**Test Code:**

```Arduino

// HomeShh TEST
// Z-Wave Maker Challenge 11th finalist
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
```

**Final wiring for test:**

<p align="center"><img src="https://user-images.githubusercontent.com/22894897/33864046-01881e42-dea8-11e7-80a5-2ca3382db4da.JPG" width="45.5%"/><img src="https://user-images.githubusercontent.com/22894897/34066098-411638e2-e1c7-11e7-9e28-4551fcdb24af.gif" width="54.5%"/></p>

<br>
<br>
<p align="center"><a href="https://lastralab.github.io/website/timeline/" target="_blank"><br><button><img src="http://i.imgur.com/ERyS5Xn.png" alt="l'astra lab icon" width="50px" background="transparent" opacity="0.5" padding="0;"/></button></a></p><br><br>
