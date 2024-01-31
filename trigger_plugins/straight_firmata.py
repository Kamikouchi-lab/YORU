import time

import serial
import serial.tools.list_ports

import asovi.arduino as ard


class trigger_condition:
    def __init__(self):
        print("trigger_command")
        myArduino = ard.dio(comport="COM3", doCh_IDs=[13])

    def trigger(self, tri_cl, in_cl, ser, results, now):
        if tri_cl in in_cl:
            myArduino.writeDO_all(1)
        else:
            myArduino.writeDO_all(0)
