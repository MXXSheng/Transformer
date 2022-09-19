import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from config import parser_args
from torch.utils.tensorboard import SummaryWriter
from datalodar import MyDataLoader
from torch.utils.data import DataLoader
from model import TransformerEncoder, create_1d_absolute_sin_cos_embedding
import torch.nn as nn
import torch
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# /home/ml/python3.8.3/bin/tensorboard --logdir=/home/ml/xusheng/Transformer/logs
# writer.add_image()
if torch.cuda.is_available():
    print("gpu可用！")
    device = torch.device("cuda:1")
else:
    print("gpu不可用")
    device = torch.device("cpu")

data = MyDataLoader()
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

train_data = DataLoader(dataset=train_dataset, batch_size=parser_args().batch_size, shuffle=True, num_workers=24)
test_data = DataLoader(dataset=test_dataset, batch_size=parser_args().batch_size, shuffle=True, num_workers=24)


position_embedding = create_1d_absolute_sin_cos_embedding(pos_len=parser_args().pos_len,
                                                          dim=parser_args().dimension_per_length)
position_embedding = position_embedding.to(device)

model = TransformerEncoder(channel=parser_args().pos_len)
model.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=parser_args().learning_rate)

# total_train_step = 0
# total_test_step = 0

writer = SummaryWriter(os.path.join(os.getcwd(), "logs"))

min_test_loss = 9999999
if os.path.exists("TransformerEncoder_parameters_MSE_number=100.pth"):
    model.load_state_dict(torch.load("TransformerEncoder_parameters_MSE_number=100.pth"))
for i in range(parser_args().epoch):
    print("----------第{}轮训练开始----------".format(i+1))

    total_train_loss = 0
    for inputs, targets, inputs_mobility in train_data:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_mobility = inputs_mobility.to(device)
        # print(type(inputs))
        # print(type(position_embedding))
        outputs = model(inputs, position_embedding, inputs_mobility)

        loss = loss_fn(outputs, targets)
        total_train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("train_loss={}".format(total_train_loss.item()))
    writer.add_scalar("train_loss", total_train_loss, i)

    if i % 10 == 0:
        total_test_loss = 0
        with torch.no_grad():
            for inputs, targets, inputs_mobility in test_data:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs_mobility = inputs_mobility.to(device)
                outputs = model(inputs, position_embedding, inputs_mobility)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss
        print("test_loss：{}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss, i)

        if total_test_loss < min_test_loss:
            min_test_loss = total_test_loss
            torch.save(model.state_dict(), "TransformerEncoder_parameters_MSE_number=100.pth")
            print("保存的模型min_test_loss = ", min_test_loss)


writer.close()
#
# for i in range(100):
#     writer.add_scalar(tag="y=x", scalar_value=i, global_step=i)
# # writer.add_scalar()
#
# writer.close()