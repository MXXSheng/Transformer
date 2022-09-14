import os

from torch.utils.tensorboard import SummaryWriter
from datalodar import MyDataLoader
from torch.utils.data import DataLoader
from model import TransformerEncoder, create_1d_absolute_sin_cos_embedding
import torch.nn as nn
import torch

# /home/ml/python3.8.3/bin/tensorboard --logdir=/home/ml/xusheng/Transformer/logs
# writer.add_image()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

db_train = MyDataLoader(train=True)
db_test = MyDataLoader(train=False)
train_data = DataLoader(dataset=db_train, batch_size=16, shuffle=True)
test_data = DataLoader(dataset=db_test, batch_size=16, shuffle=True)

position_embedding = create_1d_absolute_sin_cos_embedding(pos_len=296, dim=144)
position_embedding = position_embedding.to(device)

model = TransformerEncoder(channel=296)
model.to(device)
loss_fn = nn.MSELoss()
loss_fn.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 100
# total_train_step = 0
# total_test_step = 0

writer = SummaryWriter(os.path.join(os.getcwd(), "logs"))


min_test_loss = 9999999
if os.path.exists("TransformerEncoder_parameters.pth"):
    model.load_state_dict(torch.load("TransformerEncoder_parameters.pth"))
for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))

    total_train_loss = 0
    for inputs, targets in train_data:
        inputs, targets = inputs.to(device), targets.to(device)
        # print(type(inputs))
        outputs = model(inputs, position_embedding)

        loss = loss_fn(outputs, targets)
        total_train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("train_loss={}".format(loss.item()))
    writer.add_scalar("train_loss", total_train_loss, i)

    total_test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, position_embedding)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
    print("test_loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, i)

    if total_test_loss < min_test_loss:
        min_test_loss = total_test_loss
        torch.save(model.state_dict(), "TransformerEncoder_parameters.pth")


writer.close()
#
# for i in range(100):
#     writer.add_scalar(tag="y=x", scalar_value=i, global_step=i)
# # writer.add_scalar()
#
# writer.close()