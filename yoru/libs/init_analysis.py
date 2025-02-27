import os

import yaml


class init_analysis:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

        self.m_dict["model_path"] = "."
        self.m_dict["input_path"] = "."
        self.m_dict["input_path_image"] = "."
        self.m_dict["output_path"] = "."

        self.m_dict["quit"] = False
        self.m_dict["back_to_home"] = False
        self.m_dict["no_movies"] = "Leaving movies: none"
        self.m_dict["create_video"] = True
        self.m_dict["tracking_state"] = False
        self.m_dict["estimate_time"] = "Estimated remaining time: none"
        self.m_dict["cr_estimate_time"] = "Estimated remaining time: none"

        self.m_dict["v_flip"] = False
        self.m_dict["h_flip"] = False

        self.m_dict["threshold"] = 0.3

    def __del__(self):
        print("== Initialization finished ==.")


if __name__ == "__main__":
    mdict0 = init_analysis()
    print(mdict0.m_dict)
