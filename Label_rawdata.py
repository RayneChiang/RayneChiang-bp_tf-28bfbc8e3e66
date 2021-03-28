# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation


class MicroWaveRawData:
    def __init__(self, stepsize=100):
        self.FFT_Moving_Step = stepsize
        self.FFT_Windows_Len = 256
        self.Sampling_Freq = 1000.0
        self.h_win = np.hanning(91)
        self.my_win = np.concatenate((self.h_win[:int(91 / 2)], np.ones(256 - 91 + 1), self.h_win[-int(91 / 2):]))

        self.Plotind = 0
        self.totalind = 0

        self.fig_object, self.axs = plt.subplots(2, 1, sharex=False)
        self.fig_object.canvas.mpl_connect('button_press_event', self.on_click)
        self.rawline = None
        self.pic = None

        self.rawdata = None
        self.myfft_result = None

    def get_fft_bin(self, data):
        fft_data = np.fft.fft(data)
        fft_data = abs(fft_data[:128]).tolist()
        return fft_data

    def pretreatment(self, filepath=None):
        # read the raw csv data
        self.rawdata = pd.read_csv(filepath, header=None)
        self.rawdata = self.rawdata.iloc[:, 0].tolist()

        # using the mean value as the midline.
        testdata = self.rawdata - np.mean(self.rawdata)

        calcnr = (len(testdata) - self.FFT_Windows_Len) // self.FFT_Moving_Step + 1

        fft_result = []
        for ical in range(calcnr):
            # window-ed data for fft
            thisdata = testdata[ical * self.FFT_Moving_Step:ical * self.FFT_Moving_Step + self.FFT_Windows_Len] * self.my_win
            fft_result.append(self.get_fft_bin(thisdata))

        fft_frame = np.array(fft_result)
        self.myfft_result = np.transpose(fft_frame)

        self.totalind = int((self.myfft_result.shape[1])) * 10

    def initfig(self):
        self.axs[0].set_ylim(-100, 4196)
        self.axs[0].set_xlim(0, 30000)
        self.axs[1].tick_params(axis="x", labelbottom=False)

        self.rawline, = self.axs[0].plot([], [], 'b-')
        self.pic = self.axs[1].imshow(np.zeros(128 * 300).reshape(128, 300), extent=[0, 300 * 10, 500, 0], vmin=0, vmax=500, cmap='jet')

        plt.legend()

        return self.rawline,

    def update(self, i):
        istart = self.Plotind * 200
        iend = self.Plotind * 200 + 300

        xzom = np.arange(istart * 100, iend * 100)
        self.axs[0].set_xlim(istart * 100, iend * 100)

        self.rawline.set_data(xzom, self.rawdata[istart * 100:iend * 100])
        self.pic.set_array(self.myfft_result[:, istart:iend])

        self.Plotind += 1

        if self.Plotind > self.totalind:
            exit(0)

        return self.rawline, self.pic

    def on_click(self, event):
        if event.inaxes is None:
            return

        print(int(event.xdata))






if __name__ == '__main__':
    thistest = MicroWaveRawData()
    thistest.pretreatment(filepath=r'/home/jry/MicroWave_2/Pole6m_20191116/raw_file_20191116_172214.289.csv')

    ani = animation.FuncAnimation(fig=thistest.fig_object,
                                  func=thistest.update,
                                  frames=thistest.totalind ,
                                  init_func=thistest.initfig,
                                  interval=2000,
                                  blit=False)

    plt.show()
