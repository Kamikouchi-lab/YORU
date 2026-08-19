import time

import serial
import serial.tools.list_ports

import yoru.libs.nidaq as daq


class trigger_condition:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        print("trigger_command")
        self.mydaqDO = daq.dio(
            devID="Dev1", taskType="do", port="port0", lineCh="line0:1"
        )

    def trigger(self, tri_cl, in_cl, ser, results, now):
        if tri_cl in in_cl:
            self.mydaqDO.writeDO([True, True])
        else:
            self.mydaqDO.writeDO([False, False])
