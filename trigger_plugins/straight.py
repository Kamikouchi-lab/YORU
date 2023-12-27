import time

import serial
import serial.tools.list_ports


class trigger_condition:
    def __init__(self):
        print("trigger_command")

    def trigger(self, tri_cl, in_cl, ser, results, now):
        if tri_cl in in_cl:
            ser.write(b"1")
        else:
            ser.write(b"0")
