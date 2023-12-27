import time

import asovi.arduino as ard
import asovi.nidaq as daq

mydaqDO = daq.dio(devID="Dev1", taskType="do", port="port0", lineCh="line0:1")
mydaqDO.writeDO([True, True])
time.sleep(0.5)
mydaqDO.writeDO([False, False])
