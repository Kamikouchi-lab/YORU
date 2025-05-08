import time

import serial
import serial.tools.list_ports
import numpy as np

COM_PORT = "COM21"

class trigger_condition:

    def __init__(self, m_dict, comport = COM_PORT, rate=115200):
        self.m_dict = m_dict
        
        self.quit = False
        self.ser = serial.Serial(comport,
            rate, timeout=0.1, parity = serial.PARITY_NONE)
        self.ser.flushInput()
        self.ser.flushOutput()
        # self.ser.write(b'# INITIALIZED')
        print("trigger_command")

    def send_array(self, data=[]):
        data = ','.join(map(str, data))
        self.ser.write(f'{data}\r'.encode())
        self.ser.flush()
        return 

    def trigger(self, tri_cl, in_cl, arduino, results, now):
        if tri_cl in in_cl:
            self.send_array(np.array([1]))
        else:
            self.send_array(np.array([0]))