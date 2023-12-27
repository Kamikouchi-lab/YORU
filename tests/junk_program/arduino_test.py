import time

import asovi.arduino as ard

myArduino = ard.dio(comport="COM3", doCh_IDs=[12, 13])

myArduino.writeDO_all(1)
time.sleep(1)
myArduino.writeDO_all(0)
time.sleep(1)
myArduino.writeDO_all(1)
time.sleep(1)
myArduino.writeDO_all(0)
