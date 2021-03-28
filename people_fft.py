import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os


class people_fft:

    def __init__(self):
        self.module_path = os.path.dirname(__file__)

    def people_fft(self):
        people_event = pd.read_csv(self.module_path + "/data/raw_data/event_log_20191114_180000.926.csv")
        people_signal = pd.read_csv(self.module_path + "/data/raw_data/raw_file_20191114_180000.926.csv")

        vehicle_event = pd.read_csv("/home/jry/MicroWave/Pole6m_20191119/event_log_20191119_220000.512.csv")

        vehicle_signal = pd.read_csv("/home/jry/MicroWave/Pole6m_20191119/raw_file_20191119_220000.512.csv")

        signal = people_signal[115540:115549:8]

        transformed = np.fft.fft(signal)
        # transformed = np.abs(transformed)
        #
        # plt.figure(1)
        #
        # #plt.plot(xs)
        # plt.title("original")
        # plt.show()

        plt.figure()
        plt.plot(transformed)
        plt.title("fft")
        plt.show()

    pass


if __name__ == '__main__':
    people_fft().people_fft()
