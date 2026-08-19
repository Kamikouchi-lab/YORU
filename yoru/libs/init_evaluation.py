# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

class init_evaluater:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

        self.m_dict["model_path"] = ""
        self.m_dict["config_file_path"] = ""
        self.m_dict["project_dir"] = ""
        # Read by the GUI run loop on exit; without them Quit raised KeyError.
        self.m_dict["quit"] = False
        self.m_dict["back_to_home"] = False
        self.m_dict["result_dir"] = ""
        self.m_dict["data_dir"] = ""

    def __del__(self):
        print("== Initialization finished ==.")
