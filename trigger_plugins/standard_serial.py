import time

import serial
import serial.tools.list_ports
import numpy as np

COM_PORT = "COM3"

class trigger_condition:

    def __init__(self, m_dict, comport = COM_PORT, rate=9600):
        self.m_dict = m_dict
        
        self.quit = False
        self.ser = serial.Serial(comport,
            rate, timeout=0.1, parity = serial.PARITY_NONE)
        self.ser.flushInput()
        self.ser.flushOutput()
        # self.ser.write(b'# INITIALIZED')
        print("trigger_command")

    def trigger(self, tri_cl, in_cl, arduino, results, now):
        if tri_cl in in_cl:
            self.ser.write(b"1")
        else:
            self.ser.write(b"0")