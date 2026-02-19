// --------- Pins moteurs ----------
const int ENA = 5;
const int IN1 = 8;
const int IN2 = 9;

const int ENB = 6;
const int IN3 = 10;
const int IN4 = 11;

void setMotor(int speed, int EN, int INa, int INb) {
  int pwm = abs(speed);
  if (pwm > 255) pwm = 255;

  if (speed > 0) {          // Avant
    digitalWrite(INa, HIGH);
    digitalWrite(INb, LOW);
  } 
  else if (speed < 0) {     // Arrière
    digitalWrite(INa, LOW);
    digitalWrite(INb, HIGH);
  } 
  else {                    // Stop
    digitalWrite(INa, LOW);
    digitalWrite(INb, LOW);
  }

  analogWrite(EN, pwm);
}

void stopMotors() {
  setMotor(0, ENA, IN1, IN2);
  setMotor(0, ENB, IN3, IN4);
}

void setup() {
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  Serial.begin(115200);
  stopMotors();
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.charAt(0) == 'M') {
    int firstSpace = line.indexOf(' ');
    int secondSpace = line.indexOf(' ', firstSpace + 1);

    if (firstSpace > 0 && secondSpace > firstSpace) {
      int leftSpeed  = line.substring(firstSpace + 1, secondSpace).toInt();
      int rightSpeed = line.substring(secondSpace + 1).toInt();

      setMotor(leftSpeed, ENA, IN1, IN2);
      setMotor(rightSpeed, ENB, IN3, IN4);

      Serial.print("OK M ");
      Serial.print(leftSpeed);
      Serial.print(" ");
      Serial.println(rightSpeed);
    }
  }
  else if (line.charAt(0) == 'S') {
    stopMotors();
    Serial.println("OK S");
  }
}
