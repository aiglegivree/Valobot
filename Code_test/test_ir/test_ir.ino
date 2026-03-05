// ===== Déclaration des pins =====
const int enablePin = 2;

const int irPins[5] = {A0, A1, A2, A3, A4};
int irValues[5];

void setup() {
  Serial.begin(115200);

  // Enable en sortie
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, LOW);  // Active les capteurs

  // Les pins analogiques sont INPUT par défaut
}

void loop() {

  // Lecture des 5 capteurs
  for (int i = 0; i < 5; i++) {
    irValues[i] = analogRead(irPins[i]);
  }

  // Affichage
  Serial.print("IR: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(irValues[i]);
    Serial.print("  ");
  }
  Serial.println();

  delay(100);
}