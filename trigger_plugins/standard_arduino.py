import sys
import time

sys.path.append("../yoru")


import libs.arduino as ard
import serial
import serial.tools.list_ports


class trigger_condition:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        print("trigger_command")

        # self.myArduino = ard.dio(comport="COM3", doCh_IDs=[13])

    def trigger(self, tri_cl, in_cl, arduino, results, now):
        # print(tri_cl, in_cl, arduino, results, now)
        if tri_cl in in_cl:
            arduino.writeDO_all(1)
        else:
            arduino.writeDO_all(0)
