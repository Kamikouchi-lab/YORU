import time

import serial
import serial.tools.list_ports


class trigger_condition:
    def __init__(self, m_dict, rate=9600):
        print("trigger_command")

        self.quit = False
        comport = m_dict.get("arduino_com")
        self.ser = serial.Serial(comport, rate,
                                 timeout=0.1,
                                 parity=serial.PARITY_NONE)

    def trigger(self, tri_cl, in_cl, arduino, results, now):
        if tri_cl in in_cl:
            self.ser.write(b"1")
        else:
            self.ser.write(b"0")
