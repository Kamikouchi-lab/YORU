import time

import yoru.libs.arduino as ard
import serial
import serial.tools.list_ports


class trigger_condition:
    def __init__(self):
        print("trigger_command")
        self.myArduino = ard.dio(comport="COM3", doCh_IDs=[13])

    def trigger(self, tri_cl, in_cl, ser, results, now):
        if tri_cl in in_cl:
            self.myArduino.writeDO_all(1)
        else:
            self.myArduino.writeDO_all(0)
