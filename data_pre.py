import os
import numpy as np
import pandas as pd
import datetime
import glob


class data_pre:
    def __init__(self):
        self.path = "/home/jry/MicroWave_right"
        self.module_path = os.path.dirname(__file__)

    def prepare_data(self):

        file_dir = os.walk(self.path)
        # data_dir = self.module_path + "/data/data_right.csv"
        # label_dir = self.module_path + "/data/label_right.csv"
        label_List = []
        data_List = []
        for path, dir_list, file_List in file_dir:
            event_logs_files = glob.glob(path + "/event*.csv")
            # raw_files = glob.glob(path + "/raw*.csv")

            # if file does not contain event, then continue
            if len(event_logs_files) == 0:
                continue

            # file_index = pd.read_csv(path + "/file_index.csv", index_col=0)

            for file in event_logs_files:

                raw_dir = os.path.dirname(file) + "/" + os.path.basename(file).replace("event_log", "raw_file")

                # try:
                #     real_index = int(file_index.real_index[
                #                          file_index['file_name'] == os.path.basename(file).replace("event_log",
                #                                                                                    "raw_file")].values)
                # except TypeError:
                #     continue

                try:
                    raw_file = pd.read_csv(raw_dir, header=None, names=["detect_signal"])
                except FileNotFoundError:
                    continue

                # if raw_file['detect_signal'].isnull().any():
                #     mean = raw_file.mean()
                #     #raw_file['detect_signal'].replace(to_replace='nan', value=mean)
                #     raw_file['detect_signal'].fillna(value=mean)
                event_log = pd.read_csv(file, index_col=None, header=None, names=["time", "Event", "frame", "label"])

                event_log["label"] = event_log["label"].replace(to_replace=" people Entered]", value="0")
                event_log["label"] = event_log["label"].replace(to_replace=" people Left]", value="1")
                event_log["label"] = event_log["label"].replace(to_replace=" vehicle Entered]", value="2")
                event_log["label"] = event_log["label"].replace(to_replace=" vehicle Left]", value="3")

                for index, row in event_log.iterrows():

                    var_label = row["label"]

                    if var_label == "2" or var_label == "3":

                        var_frame = row["frame"]
                        try:
                            signal_list = raw_file.loc[
                                          (var_frame * 100):(var_frame * 100 + 1999)]
                        except TypeError:
                            continue

                        if len(signal_list) < 2000:
                            continue

                        # signal_list = [signal_list[i] for i in range(2000) if i % 10 == 0]

                        signal_series = signal_list["detect_signal"].values

                        signal_series = [signal_series[i] for i in range(2000) if i % 5 == 0]

                        signal_series = data_pre().normalization(signal_series)
                        label_List.append(var_label)
                        data_List.append(signal_series)

        label_Frame = pd.DataFrame(data=label_List)
        # print(label_Frame.isna().sum())
        label_mean = label_Frame.mean()
        label_Frame.fillna(label_mean, inplace=True)
        # print("after")
        # print(label_Frame.isna().sum())
        label_Frame.to_csv(self.module_path + "/data/label_right_car_5.csv")

        data_Frame = pd.DataFrame(data=data_List)
        # print(data_Frame.isna().sum())
        data_mean = data_Frame.mean()
        data_Frame.fillna(data_mean, inplace=True)
        # print("after")
        # print(data_Frame.isna().sum())
        data_Frame.to_csv(self.module_path + "/data/data_right_car_5.csv")

    def normalization(self, x):
        min = np.min(x, axis=0)
        max = np.max(x, axis=0)
        mean = np.mean(x, axis=0)
        return (x - mean) / (max - min)

    def loadData(self):
        data_dir = self.module_path + "/data/data_right_car_5.csv"
        data = pd.read_csv(data_dir, index_col=0)

        label_dir = self.module_path + "/data/label_right_car_5.csv"
        label = pd.read_csv(label_dir, index_col=0)

        return data, label


    def load_npz_data(self, path):
        with np.load(path) as data:
            my_examples = data['array1']
            my_labels = data['array2']

        my_labels = my_labels[:322]
        my_examples = my_examples[:322]

        data = pd.DataFrame(data=my_examples)

        label = my_labels.T


        return data, label


    def loadLog(self):
        log_dir = self.module_path + "/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        return log_dir

    def loadModel(self):
        model_dir = self.module_path + "/my_model"

        return model_dir

    # write index to file_index.csv
    # solve the index discontinuous problem
    def resetIndex(self):

        raw_files = []
        file_dir = os.walk(self.path)
        for path, dir_list, file_List in file_dir:
            raw_files = glob.glob(path + "/raw*.csv")
            raw_files.sort()
            raw_file_index = []
            index_length = 0
            for file in raw_files:
                file_in_dir = os.path.dirname(file) + "/" + os.path.basename(file)
                try:
                    file_in = pd.read_csv(file_in_dir)
                except pd.errors.EmptyDataError:
                    print(file_in_dir)

                raw_file_index.append([os.path.basename(file), index_length])
                index_length += len(file_in)

            raw_file_csv = pd.DataFrame(columns=['file_name', 'real_index'], data=raw_file_index)
            raw_file_csv.to_csv(path + "/file_index.csv")

        # file_out_dir = os.path.dirname(raw_files[i]) + "/" + os.path.basename(raw_files[i])
        # file_out = pd.read_csv(file_out_dir, index_col=None)
        # file_out.index += len(file_in)
        # file_out.to_csv(file_out_dir, index_label=None)
        # i = i + 1



if __name__ == '__main__':
    data_pre().prepare_data()
    # data_pre().resetIndex()
    # data_pre().loadData()
