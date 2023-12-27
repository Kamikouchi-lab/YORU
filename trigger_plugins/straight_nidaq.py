import time

import serial
import serial.tools.list_ports

import asovi.nidaq as daq


class trigger_condition:
    def __init__(self):
        print("trigger_command")
        mydaqDO = daq.dio(devID="Dev1", taskType="do", port="port0", lineCh="line0:1")

    def trigger(self, tri_cl, in_cl, ser, results, now):
        if tri_cl in in_cl:
            mydaqDO.writeDO([True, True])
        else:
            mydaqDO.writeDO([False, False])
