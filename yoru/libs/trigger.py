# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import importlib
import time

import serial
import serial.tools.list_ports

import yoru.libs.arduino as ard
from yoru.libs.paths import ensure_importable, list_trigger_plugins


class yolo_trigger:
    def __init__(self, m_dict={}):
        print("== Trigger Start ==")
        self.m_dict = m_dict
        self.m_dict["Trigger"] = False

    def init_trigger(self):
        while not self.m_dict.get("quit", False):
            # print("a")
            if not self.m_dict.get("Trigger", False):
                time.sleep(1)  # 1秒間スリープしてCPUの使用率を下げる
                continue

            print("trigger loading...")
            self.class_list = self.m_dict.get("class_list", [])

            try:
                self.arduino_tri = trigger_python(self.m_dict)
                print("read COM clear")
            # except serial.serialutil.SerialException:
            #     # self.arduino_tri.serial_close
            #     print("No COM ....")

            #     time.sleep(1)
            # continue
            except Exception as e:  # 具体的なエラーメッセージを出力
                print(f"Error: {e}")
                time.sleep(1)  # 失敗が続いてもCPUを占有しないようにする
                continue

            self.process_triggers()

    def process_triggers(self):
        try:
            while not self.m_dict.get("quit", False) and self.m_dict.get(
                "Trigger", False
            ):
                try:
                    self.arduino_tri.trigger()
                    # 　trigger処理
                except serial.serialutil.SerialException:
                    print("Trigger failure ....")
                    time.sleep(1)
                    break
                except TypeError:
                    print("Not turning on YOLO....")
                    time.sleep(1)
                    break
                except Exception as e:
                    # 想定外の例外でトリガープロセスごと落ちないようにする
                    print(f"Trigger error: {e}")
                    time.sleep(1)
                    break
        finally:
            if self.arduino_tri is not None:
                self.arduino_tri.close()
                self.arduino_tri = None


class trigger_python:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict
        self.com = self.m_dict["arduino_com"]
        # pyfirmata indexes board.digital[pin], so a str from the GUI would raise.
        try:
            self.pin = int(self.m_dict.get("pin", 13))
        except (TypeError, ValueError):
            print(f"Invalid trigger pin {self.m_dict.get('pin')!r}; falling back to 13")
            self.pin = 13
        self.myArduino = None  # 初期化
        # self.ser_baudrate = int(self.m_dict.get("baudrate", 9600))

        if self.com and self.com != "None":
            try:
                self.myArduino = ard.dio(comport=self.com, doCh_IDs=[self.pin])
            except PermissionError as e:
                print(f"Error: could not open port '{self.com}': {e}")
                if self.myArduino:
                    self.myArduino.close()        # ard.dio の close メソッド
                self.myArduino = None
            
        else:
            self.myArduino = None

        self.tri_class = self.m_dict.get("trigger_class")
        # config の trigger_threshold_configuration (= trigger_th_conf)
        try:
            self.tri_th_conf = float(self.m_dict.get("trigger_th_conf", 0.0))
        except (TypeError, ValueError):
            print(
                f"Invalid trigger threshold {self.m_dict.get('trigger_th_conf')!r}; "
                "falling back to 0.0"
            )
            self.tri_th_conf = 0.0
        self.m_dict["plugin_name"] = "trigger_plugins." + self.m_dict.get(
            "in_plugin_name", ""
        )
        self.trigger_instance = self._load_plugin()

        print("Open Port")

    def _load_plugin(self):
        import_path = self.m_dict.get("plugin_name")
        print(import_path)
        # trigger_plugins lives beside the yoru package, not inside it.
        ensure_importable()
        module = importlib.import_module(import_path)
        return module.trigger_condition(self.m_dict)

    def _detected_class_names(self):
        """Class names of detections at or above the trigger confidence threshold.

        ``yolo_results`` rows are
        ``[x1, y1, x2, y2, conf, class_id, class_name, total_time]``
        (see :mod:`yoru.libs.detection`).  Reading only this one key keeps the
        confidence and the class name consistent with each other.
        """
        results = self.m_dict.get("yolo_results")
        if results is None or len(results) == 0:
            return []
        names = []
        for row in results:
            try:
                if float(row[4]) >= self.tri_th_conf:
                    names.append(row[6])
            except (IndexError, TypeError, ValueError):
                continue
        return names

    def trigger(self):
        self.trigger_instance.trigger(
            self.tri_class,
            self._detected_class_names(),
            self.myArduino,
            self.m_dict.get("yolo_results", []),
            self.m_dict.get("now"),
        )
        # print("trigger_command")

    def close(self):
        if self.myArduino:
            try:
                self.myArduino.writeDO_all(0)
            finally:
                self.myArduino.close()
                print("Arduino connection closed.")
        self.trigger_instance = None
        print("Trigger instance set to None.")


class read_condition:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def list_com_ports(self):
        comlist = serial.tools.list_ports.comports()
        self.m_dict["COM_list"] = [element.device for element in comlist]

    def list_plugins(self):
        self.m_dict["plugins"] = list_trigger_plugins()


if __name__ == "__main__":
    # Print list of connected COM ports
    m_dict = {}
    read = read_condition(m_dict)
    read.list_com_ports()
