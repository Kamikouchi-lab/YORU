# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import re
import threading

import numpy as np
import pyfirmata
import serial

_RE_NUMBER = re.compile(b"[+-]?\\d{1,}\r\n")


class dio:
    def __init__(
        self, comport="COM1", diCh_IDs=[2, 3, 4, 5], doCh_IDs=[10, 11, 12, 13]
    ):
        self.board = pyfirmata.Arduino(comport)
        self.diCh_IDs = diCh_IDs
        self.doCh_IDs = doCh_IDs

        self.diChs = []
        self.doChs = []
        self.diState = []
        for i in self.diCh_IDs:
            print("d:" + str(i) + ":i")
            self.diChs.append(self.board.get_pin("d:" + str(i) + ":i"))
            self.board.digital[i].mode = pyfirmata.INPUT
            self.board.digital[i].enable_reporting()
        for di in self.diChs:
            self.diState.append(di.read())
        for i in self.doCh_IDs:
            print("d:" + str(i) + ":o")
            self.doChs.append(self.board.get_pin("d:" + str(i) + ":o"))
            self.board.digital[i].mode = pyfirmata.OUTPUT
        self.it = pyfirmata.util.Iterator(self.board)
        self.it.start()

        self.t1 = threading.Thread(target=self.readDI_all_inf)

    def readDI_all(self):
        for i, di in enumerate(self.diChs):
            self.diState[i] = di.read()

    def readDI_all_inf(self):
        while 1:
            for i, di in enumerate(self.diChs):
                self.diState[i] = di.read()

    def start_readDI_all_inf(self):
        self.t1 = threading.Thread(target=self.readDI_all_inf, daemon=True)
        self.t1.start()

    def writeDO_all(self, num):
        for i, do in enumerate(self.doChs):
            do.write(num)


class ser_recount:
    def __init__(self, comport="COM1", rate=115200):
        print("Start serial COM @" + comport)
        self.quit = False
        self.ser = serial.Serial(
            comport, rate, timeout=0.1, parity=serial.PARITY_NONE
        )
        self._lock = threading.Lock()
        self.r = bytearray()
        self.rbuf = 0.0
        self.t1 = threading.Thread(target=self.fun_readline, args=[self.ser], daemon=True)
        self.t1.start()
        print("==== Serial RE-counter initialized ====")

    def fun_readline(self, ser):
        buf = bytearray()
        self.rbuf = 0
        while not self.quit:
            i = max(1, min(1024, ser.in_waiting))
            data = ser.read(i)
            i = data.find(b"\n")
            if i > 0:
                r = buf + data[: i + 1]
                matches = _RE_NUMBER.findall(r)
                with self._lock:
                    self.r = r
                    if matches:
                        self.rbuf = float(matches[-1])
                buf[0:] = data[i + 1 :]
            else:
                buf.extend(data)
            ser.flush()
        return -1

    def __del__(self):
        self.ser.close()
