import time
import serial

PORT = "/dev/ttyACM0"
BAUD = 115200

def send(ser, s: str):
    ser.write((s.strip() + "\n").encode("utf-8"))
    ser.flush()
    # lecture réponse (optionnel)
    time.sleep(0.05)
    while ser.in_waiting:
        print(ser.readline().decode(errors="ignore").strip())

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    time.sleep(2)  # reset Arduino à l'ouverture du port
    send(ser, "V 150")
    send(ser, "F")
    time.sleep(2)

    send(ser, "S")
    time.sleep(1)

    send(ser, "B")
    time.sleep(2)

    send(ser, "S")
