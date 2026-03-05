// Moteur A
const int ENA = 5;   // PWM
const int IN1 = 7;
const int IN2 = 8;

// Moteur B
const int ENB = 6;   // PWM
const int IN3 = 9;
const int IN4 = 10;

// Gains pour égaliser la vitesse (à régler)
float gainA = 1.00;
float gainB = 0.90;

int clampPWM(int v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return v;
}

void setForward(int pwm) {
  // Direction avant (à inverser si un moteur est monté à l'envers)
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  int pwmA = clampPWM((int)(pwm * gainA));
  int pwmB = clampPWM((int)(pwm * gainB));

  analogWrite(ENA, pwmA);
  analogWrite(ENB, pwmB);
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotors();
}

void loop() {
  setForward(150);
  delay(2000);

  stopMotors();
  delay(1000);
}