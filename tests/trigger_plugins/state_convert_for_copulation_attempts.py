import time

import serial
import serial.tools.list_ports


class trigger_condition:
    def __init__(self):
        self.sig_state = 0
        self.no_sig_state = 0
        self.current_state = 0
        print("trigger_command")

    def trigger(self, tri_cl, in_cl, ser, results, now):
        # self: first argument (not used)
        # tri_cl: trigger class
        # in_cl: input class
        # ser: Serial

        # Copulation decision
        if tri_cl in in_cl:
            self.sig_state += 1
            self.no_sig_state = 0

            # If sig_state is over 30, change state to 1

            if self.sig_state > 10:
                self.current_state = 1
        else:
            self.no_sig_state += 1

            # If no_sig_state is over 10, reset both states and change state to 0

            if self.no_sig_state > 10:
                self.sig_state = 0
                self.no_sig_state = 0
                self.current_state = 0

        # print(self.sig_state)
        # print(self.no_sig_state)
        # print(self.current_state)
        self.trigger_to_arduino(ser)

    def trigger_to_arduino(self, ser):
        # Trigger to Arduino
        if self.current_state == 1:
            ser.write(b"1")
        elif self.current_state == 0:
            ser.write(b"0")
