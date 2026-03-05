#include <Servo.h>

Servo monServo;

void setup() {
  monServo.attach(9);  // Servo sur D9
}

void loop() {
  // Position 0°
  monServo.write(0);
  delay(1000);

  // Position 90°
  monServo.write(90);
  delay(1000);

  // Position 180°
  monServo.write(180);
  delay(1000);
}