import glob
import importlib
import os
import time

import serial
import serial.tools.list_ports


class yolo_trigger:
    def __init__(self, m_dict={}):
        print("== Trigger Start ==")
        self.m_dict = m_dict
        self.m_dict["Trigger"] = False

    def init_trigger(self):
        while not self.m_dict.get("quit", False):
            if not self.m_dict.get("Trigger", False):
                time.sleep(1)  # 1秒間スリープしてCPUの使用率を下げる
                continue

            print("trigger loading...")
            self.class_list = self.m_dict.get("class_list", [])

            try:
                self.arduino_tri = trigger_python(self.m_dict)
                print("read COM clear")
            except serial.serialutil.SerialException:
                # self.arduino_tri.serial_close
                print("No COM ....")

                time.sleep(1)
                # continue
            except Exception as e:  # 具体的なエラーメッセージを出力
                print(f"Error: {e}")
                continue

            self.process_triggers()

    def process_triggers(self):
        while not self.m_dict.get("quit", False) and self.m_dict.get("Trigger", False):
            try:
                # クラス名を取得
                self.class_ids = self.m_dict["yolo_results"][:, 5].astype(int)
                self.m_dict["yolo_class_names"] = [
                    self.class_list[i] for i in self.class_ids
                ]

                self.arduino_tri.trigger()
                # print(self.m_dict["yolo_results"])

                # 　trigger処理
            except serial.serialutil.SerialException:
                print("Trigger failure ....")
                time.sleep(1)
                break
            except TypeError:
                print("Not turning on YOLO....")
                time.sleep(1)
                break


class trigger_python:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict
        self.com = self.m_dict["arduino_com"]
        self.ser_baudrate = int(self.m_dict.get("baudrate", 9600))

        if self.com != "None":
            self.ser = serial.Serial(
                port=self.com, baudrate=self.ser_baudrate, timeout=1
            )
        else:
            self.ser = None

        self.tri_class = self.m_dict.get("trigger_class")
        self.m_dict["plugin_name"] = "trigger_plugins." + self.m_dict.get(
            "in_plugin_name", ""
        )
        self.trigger_instance = self._load_plugin()

        print("Open Port")

    def _load_plugin(self):
        import_path = self.m_dict.get("plugin_name")
        print(import_path)
        module = importlib.import_module(import_path)
        return module.trigger_condition()

    def trigger(self):
        # self.module.trigger_condition.trigger(self.tri_class, self.m_dict['yolo_class_names'], self.ser)
        self.trigger_instance.trigger(
            self.tri_class,
            self.m_dict.get("yolo_class_names", []),
            self.ser,
            self.m_dict["yolo_results"].astype(int),
            self.m_dict["now"],
        )
        # print("trigger_command")

    def serial_close(self):
        if self.ser:
            self.ser.close()


class read_condition:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def list_com_ports(self):
        comlist = serial.tools.list_ports.comports()
        self.m_dict["COM_list"] = [element.device for element in comlist]

    def list_plugins(self):
        file_paths = glob.glob("./trigger_plugins/*.py", recursive=True)
        self.m_dict["plugins"] = [
            os.path.splitext(os.path.basename(path))[0] for path in file_paths
        ]


if __name__ == "__main__":
    # Print list of connected COM ports
    m_dict = {}
    read = read_com(m_dict)
    read.list_com_ports()
