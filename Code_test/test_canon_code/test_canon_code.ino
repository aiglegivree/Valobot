const int moteurPin = 7;

void setup() {
  pinMode(moteurPin, OUTPUT);
  digitalWrite(moteurPin, LOW);  // moteur OFF au démarrage
}

void loop() {
  digitalWrite(moteurPin, HIGH); // moteur ON
  delay(3000);                   // 3 secondes

  digitalWrite(moteurPin, LOW);  // moteur OFF
  delay(3000);                   // pause 1 seconde
}