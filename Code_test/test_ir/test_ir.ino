const int irPin = A0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(irPin);

  Serial.print("Analog value: ");
  Serial.println(sensorValue);

  delay(200);
}
