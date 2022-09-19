import csv
import glob
import random
import sys
from sklearn.preprocessing import MinMaxScaler
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from joblib import Parallel, delayed
from config import parser_args


class MyDataLoader(Dataset):
    def __init__(self, prediction_number=parser_args().prediction_number):
        super(MyDataLoader, self).__init__()
        self.prediction_number = prediction_number
        self.data, self.data_prediction, self.data_mobility = self.matrices_loader()

    def function(self, mobility, total_data_root, filename):
        # if (mobility * 1000) % 10 == 0:
        print("mobility = {:.3f}".format(mobility))
        data_path = os.path.join(total_data_root, "mobility={:.3f}".format(mobility))

        matrices_path = []
        matrices_csv_path = os.path.join(data_path, "matrices")
        # print(matrices_csv_path)
        # 如果不存在路径汇总的csv文件 则创建
        # print("正在汇总数据集")
        if not os.path.exists(os.path.join(data_path, filename)):
            matrices_path += sorted(glob.glob(os.path.join(matrices_csv_path, "*.csv")))
            # print(matrices_path)
            with open(os.path.join(data_path, filename), mode='w', newline='') as file:
                writer = csv.writer(file)
                for matrix_path in matrices_path:
                    # print(matrix_path)
                    writer.writerow([matrix_path])
            # print("writen into csv file, length = " + str(len(matrices_path)) + "!")
        # 如果已经存在了汇总文件
        else:
            # print("csv file has existed!")
            with open(os.path.join(data_path, filename)) as file:
                reader = csv.reader(file)
                for matrix_path in reader:
                    matrices_path += matrix_path
        data, data_prediction, data_mobility = [], [], []

        # print("正在构建训练集输入和输出")
        for i in range(len(matrices_path)):
            matrix_path = matrices_path[i]
            # print("matrix_path = ", matrix_path)
            matrix = pd.read_csv(matrix_path, header=None)
            matrix = np.array(matrix.values)
            mm = MinMaxScaler()
            matrix = mm.fit_transform(matrix)

            # *********输入输出错开********
            if i >= int(self.prediction_number):
                data_prediction.append(matrix)
                data_mobility.append(mobility)
            if i < int(len(matrices_path) - self.prediction_number):
                data.append(matrix)
            # *****************
        data = np.array(data)
        data_prediction = np.array(data_prediction)
        data_mobility = np.array(data_mobility)
        return data, data_prediction, data_mobility

    def matrices_loader(self, filename=parser_args().filename):
        # data文件夹
        total_data_root = os.path.join(os.getcwd(), "datasets", "data")

        (data, data_prediction, data_mobility) = zip(*Parallel(n_jobs=24)(
            delayed(self.function)(mobility, total_data_root, filename,)
            for mobility in np.arange(0.55, 1.45, 0.001)))

        # print(type(data))
        data = torch.from_numpy(np.array(data).astype(np.float32))
        data_prediction = torch.from_numpy(np.array(data_prediction).astype(np.float32))
        data_mobility = torch.from_numpy(np.array(data_mobility).astype(np.float32))
        # data_prediction_mobility = torch.from_numpy(np.array(data_prediction_mobility).astype(np.float32))


        print("data.shape = ", data.shape)
        # print("--------------data=--------------")
        # print(data)

        # print(data_prediction.shape)
        # if(len(data) != 996):
        #     print("mobility = ", mobility)
        #     print(len(data))
        #
        # assert len(data) == 996
        # assert len(data_prediction) == 996
        return data, data_prediction, data_mobility

    def __getitem__(self, item):
        return self.data[item], self.data_prediction[item], self.data_mobility[item]

    def __len__(self):
        return len(self.data)


def main():
    data = MyDataLoader()

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    train_data = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_data = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

    # for i in train_data:
    #     print(i.shape)
    #     break
    # for data in db:
    #     print(data.shape)


if __name__ == '__main__':
    main()
