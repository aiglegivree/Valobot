// Moteur A
const int ENA = 5;   // PWM vitesse moteur A
const int IN1 = 8;   // direction A
const int IN2 = 9;   // direction A

// Moteur B
const int ENB = 6;   // PWM vitesse moteur B
const int IN3 = 10;  // direction B
const int IN4 = 11;  // direction B

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // STOP initial des deux moteurs
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void loop() {

  // AVANT à 60% (les deux moteurs)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 150);
  analogWrite(ENB, 150);
  delay(2000);

  // STOP
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  delay(1000);

  // ARRIÈRE à 60% (les deux moteurs)
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENA, 150);
  analogWrite(ENB, 150);
  delay(2000);

  // STOP
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  delay(2000);

  // ARRIÈRE à 20%
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENA, 100);  // 20% de 255
  analogWrite(ENB, 100);
  delay(2000);

  // STOP final
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  delay(2000);

  // Avant à 100%
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 255);  // 100% de 255
  analogWrite(ENB, 255);
  delay(2000);

  // STOP final
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  delay(2000);
}

