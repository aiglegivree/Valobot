import time
import serial

PORT = "/dev/ttyACM0"
BAUD = 115200

def send(ser, cmd):
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(0.05)
    while ser.in_waiting:
        print(ser.readline().decode().strip())

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    time.sleep(2)

    # Avancer
    send(ser, "M 150 150")
    time.sleep(2)

    # Tourner gauche
    send(ser, "M -100 100")
    time.sleep(2)

    # Tourner droite
    send(ser, "M 100 -100")
    time.sleep(2)

    # Reculer
    send(ser, "M -150 -150")
    time.sleep(2)

    # Stop
    send(ser, "S")
