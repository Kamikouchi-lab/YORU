import os

import numpy as np
import yaml


class init_evaluater:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

        self.m_dict["model_path"] = ""
        self.m_dict["config_file_path"] = ""
        self.m_dict["project_dir"] = ""
        self.m_dict["result_dir"] = ""
        self.m_dict["datas_dir"] = ""

    def __del__(self):
        print("== Initialization finished ==.")


if __name__ == "__main__":
    mdict0 = init_movie_gui()
    print(mdict0.m_dict)
