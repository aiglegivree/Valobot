// --------- Pins moteurs (comme ton code) ----------
const int ENA = 5;   // PWM moteur A
const int IN1 = 8;
const int IN2 = 9;

const int ENB = 6;   // PWM moteur B
const int IN3 = 10;
const int IN4 = 11;

// --------- Paramètres ----------
int currentSpeed = 150; // 0..255

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void forward(int spd) {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, spd);
  analogWrite(ENB, spd);
}

void backward(int spd) {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  analogWrite(ENA, spd);
  analogWrite(ENB, spd);
}

// Rotation sur place (optionnel)
void left(int spd) {   // A arrière, B avant
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, spd);
  analogWrite(ENB, spd);
}

void right(int spd) {  // A avant, B arrière
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  analogWrite(ENA, spd);
  analogWrite(ENB, spd);
}

void setup() {
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  Serial.begin(115200);
  stopMotors();
  Serial.println("READY");
}

// Protocole simple (lignes terminées par \n):
// F            -> avant vitesse courante
// B            -> arrière vitesse courante
// L / R        -> pivoter gauche/droite vitesse courante
// S            -> stop
// V <0..255>   -> set vitesse courante
// M <a> <b>    -> set vitesses directes (a,b 0..255) + conserve direction actuelle (simple)
// Exemple: "V 180" puis "F"

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  // Commande + éventuels arguments
  char cmd = line.charAt(0);

  if (cmd == 'S') {
    stopMotors();
    Serial.println("OK S");
  }
  else if (cmd == 'F') {
    forward(currentSpeed);
    Serial.println("OK F");
  }
  else if (cmd == 'B') {
    backward(currentSpeed);
    Serial.println("OK B");
  }
  else if (cmd == 'L') {
    left(currentSpeed);
    Serial.println("OK L");
  }
  else if (cmd == 'R') {
    right(currentSpeed);
    Serial.println("OK R");
  }
  else if (cmd == 'V') {
    int v = line.substring(1).toInt();
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    currentSpeed = v;
    Serial.print("OK V ");
    Serial.println(currentSpeed);
  }
  else {
    Serial.print("ERR unknown: ");
    Serial.println(line);
  }
}
