import sys
import time
from enum import auto

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping
from nidaqmx.errors import DaqError


# Mainly for NI USB-6001 and similar type of DAQs
class dio:
    def __init__(self, devID="Dev1", taskType="di", port="port0", lineCh="line4:7"):
        self.devID = devID
        self.taskType = taskType
        self.port = port
        self.lineCh = lineCh
        self.task = nidaqmx.Task()
        if taskType == "di":
            self.task.di_channels.add_di_chan(
                "/" + devID + "/" + port + "/" + lineCh,
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )
            self.nCh = len(self.task.di_channels.channel_names)
            print("Configured: " + devID + "/" + port + "/" + lineCh + " (DI)")
        elif taskType == "do":
            self.task.do_channels.add_do_chan(
                "/" + devID + "/" + port + "/" + lineCh,
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )
            self.nCh = len(self.task.do_channels.channel_names)
            print("Configured: " + devID + "/" + port + "/" + lineCh + " (DO)")
        else:
            print("Unknown Task type: " + taskType)

    def startAll(self):
        self.task.start()

    def stop(self):
        # self.task.stop()
        # self.task.close()
        self.task.__del__()

    def writeDO(self, tflist):
        self.task.write(tflist)

    def readDI(self):
        return self.task.read()

    def demoDIO(self):
        for i in range(5):
            self.writeDO([True, False, True, False])
            print(self.readDI())
            time.sleep(0.05)
            self.writeDO([False, True, False, True])
            print(self.readDI())
        self.writeDO([False, False, False, False])

    def __del__(self):
        print("Destructed: " + self.devID + "")
        self.task.stop()
        self.task.close()


class ao:
    def __init__(self, devID="Dev1", aoCh="ao0:1", type="sw", fs=1000):
        self.devID = devID
        self.aoCh = aoCh
        self.type = type
        self.task = nidaqmx.Task()

        self.task.ao_channels.add_ao_voltage_chan(
            "/" + devID + "/" + aoCh, max_val=5.0, min_val=0.0
        )
        self.nCh = self.task.number_of_channels

        if type == "sw":
            print("Configured: " + devID + ":" + aoCh + " (sw)")
            if self.nCh > 1:
                self.task.write(np.zeros((1, self.nCh)))
            elif self.nCh == 0:
                self.task.write(0)
            self.task.stop()
        elif type == "hw":
            self.task.timing.cfg_samp_clk_timing(fs)
            print("Configured: " + devID + ":" + aoCh + " (hw)")
            self.task.write(np.zeros((2, self.nCh)), auto_start=True)
            self.task.wait_until_done()
            self.task.stop()

    def writeAO(self, aomat):
        if self.type == "sw":
            self.task.write(aomat)
            self.task.stop()
        elif self.type == "hw":
            self.task.write(aomat, auto_start=True)
            self.task.wait_until_done()
            self.task.stop()

    def writeAO_pulse(self, aomat):
        pass

    def __del__(self):
        print("Destructed: " + self.devID + ":" + self.aoCh)
        self.task.stop()
        self.task.close()


class ai:
    def __init__(
        self,
        devID="Dev1",
        aiCh="ai0",
        trigger="PFI0",
        fs=1000,
        n_sample=10000,
        timeout=180,
    ):
        self.devID = devID
        self.aiCh = aiCh
        self.trigger = trigger
        self.fs = fs
        self.n_sample = n_sample
        self.timeout = timeout

    # Analog signal acquisition by start trigger
    def run_triggered_rec(self):
        with nidaqmx.Task() as aitask:
            # Configuration
            aitask.ai_channels.add_ai_voltage_chan(self.devID + "/" + self.aiCh)
            aitask.timing.cfg_samp_clk_timing(
                self.fs,
                "",
                Edge.RISING,
                AcquisitionType.FINITE,
                samps_per_chan=self.n_sample,
            )
            aitask.triggers.start_trigger.cfg_dig_edge_start_trig(
                "/" + self.devID + "/" + self.trigger, Edge.RISING
            )

            # Acquisition
            print(
                "Waiting Trigger for Capture-Start @" + self.devID + "/" + self.trigger
            )
            print("Acquisition from " + self.devID + "/" + self.aiCh)
            readout = aitask.read(
                number_of_samples_per_channel=self.n_sample, timeout=self.timeout
            )
            print("Acquisition Finished: ", np.shape(readout))

        return readout

    def __del__(self):
        pass


class pulse_counter:
    def __init__(self, devID="Dev1", counter="ctr0"):
        self.devID = devID
        self.ctrID = counter

    def wait_pulse_count(self, total_count=1):
        with nidaqmx.Task() as task:
            task.ci_channels.add_ci_count_edges_chan(self.devID + "/" + self.ctrID)
            task.start()
            last_count = 0
            while 1:
                data = task.read()
                if data > last_count:
                    print(data)
                    last_count = data
                    if last_count >= total_count:
                        break

    def __del__(self):
        pass


if __name__ == "__main__":
    task = dio("Dev2", "do", port="port0", lineCh="line0")
    # task2 = dio("Dev1", "di", port="port0", lineCh="line2:3")

    # task.writeDO([True, True])
    # time.sleep(.5)
    # task.writeDO([False,False])

    task.writeDO([True])
    time.sleep(0.5)
    task.writeDO([False])

    # time.sleep(.05)
    # print(task2.readDI())

    # task.writeDO( [False, False, False, False])
    # print(task2.readDI())
    # task.writeDO([False,False])

    task.stop()
    print("!!!")
    # # task2.stop()
    # sys.exit()
