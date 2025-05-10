import serial
from time import sleep

def readserial(comport, baudrate):

    ser = serial.Serial(comport, baudrate, timeout=0.1)         # 1/timeout is the frequency at which the port is read

    ser.write("111\n".encode())

    sleep(2)

    message = ser.readline().strip().decode("utf-8")
    print("message", message)
readserial('COM7', 9600)