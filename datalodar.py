import csv
import glob
import random

import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os


class MyDataLoader(Dataset):
    def __init__(self, train: bool = True):
        super(MyDataLoader, self).__init__()
        self.train = train
        self.total_data, self.total_prediction_data = self.matrices_loader()

    def matrices_loader(self, filename="matrices_of_path.csv"):
        # data文件夹

        total_data_root = os.path.join(os.getcwd(), "datasets", "data")
        total_data, total_prediction_data = [], []
        prediction_number = 4
        for mobility in np.arange(0.5, 1.5, 0.001):
            # mobility文件夹
            print("mobility = {:.3f}".format(mobility))
            data_path = os.path.join(total_data_root, "mobility={:.3f}".format(mobility))

            matrices_path = []
            matrices_csv_path = os.path.join(data_path, "matrices")
            # print(matrices_csv_path)
            # 如果不存在路径汇总的csv文件 则创建
            if not os.path.exists(os.path.join(data_path, filename)):
                matrices_path += sorted(glob.glob(os.path.join(matrices_csv_path, "*.csv")))
                # print(matrices_path)
                with open(os.path.join(data_path, filename), mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for matrix_path in matrices_path:
                        # print(matrix_path)
                        writer.writerow([matrix_path])
                print("writen into csv file, length = " + str(len(matrices_path)) + "!")
            # 如果已经存在了汇总文件
            else:
                print("csv file has existed!")
                with open(os.path.join(data_path, filename)) as file:
                    reader = csv.reader(file)
                    for matrix_path in reader:
                        matrices_path += matrix_path

            data, prediction_data = [], []
            assert prediction_number % 2 == 0

            # 构建训练集输入和输出
            for i in range(len(matrices_path)):
                matrix_path = matrices_path[i]
                # print("matrix_path = ", matrix_path)
                matrix = pd.read_csv(matrix_path, header=None)
                matrix = matrix.values



                if i >= prediction_number:
                    prediction_data.append(matrix)
                if i < len(matrices_path) - prediction_number:
                    data.append(matrix)

            # print(len(data))
            # print(len(prediction_data))
            assert len(data) == len(prediction_data)

            total_data.append(data)
            total_prediction_data.append(prediction_data)

        total_data = np.array(total_data)
        total_prediction_data = np.array(total_prediction_data)

        print(len(total_data))

        if self.train:
            total_data = total_data[:int(0.8 * len(total_data))]
            total_prediction_data = total_prediction_data[:int(0.8 * len(total_prediction_data))]
        else:
            total_data = total_data[int(0.8 * len(total_data)):]
            total_prediction_data = total_prediction_data[int(0.8 * len(total_prediction_data)):]


        return total_data, total_prediction_data

    def __getitem__(self, item):
        return (torch.tensor(self.total_data[item])).to(torch.float32), \
               (torch.tensor(self.total_prediction_data[item])).to(torch.float32)

    def __len__(self):
        return len(self.total_data)


def main():
    db = MyDataLoader()
    data = db.total_data
    print(data.shape)

    # for data in db:
    #     x, y = data
    #
    #     print(x.shape)
    #     print(y.shape)


if __name__ == '__main__':
    main()
