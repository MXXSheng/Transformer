import torch
import torch.nn as nn
from torchsummary import summary


# 1d绝对sin_cos编码

def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim // 2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, dimension_per_length=144, dimension_qkv=512, heads=8, bias=False):
        """

        :param input_x:   shape:[batch, length, dimension_per_length]
        :param dimension_qkv:
        :param heads:
        :param bias:
        :param dropout_prob:
        """
        super(MultiHeadAttention, self).__init__()
        self.bias = bias
        self.heads = heads
        self.dimension_qkv = dimension_qkv
        self.softmax = nn.Softmax(dim=-1)
        self.dimension_per_length = dimension_per_length
        self.linear_q = nn.Linear(self.dimension_per_length, self.heads * self.dimension_qkv, bias=self.bias)
        self.linear_k = nn.Linear(self.dimension_per_length, self.heads * self.dimension_qkv, bias=self.bias)
        self.linear_v = nn.Linear(self.dimension_per_length, self.heads * self.dimension_qkv, bias=self.bias)
        self.output_linear = nn.Linear(self.heads * self.dimension_qkv, self.dimension_per_length)

    def __call__(self, input_x):
        # print(input_x.shape)
        batch_size = input_x.shape[0]
        seq_length = input_x.shape[1]
        dimension_per_length = input_x.shape[2]
        q, k, v = self.linear_q(input_x), self.linear_k(input_x), self.linear_v(input_x)
        q = q.view(batch_size, seq_length, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_length, self.heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_length, self.heads, -1).permute(0, 2, 1, 3)

        attention = q @ k.transpose(2, 3) / (self.dimension_qkv ** 0.5)
        attention = self.softmax(attention)

        output = ((attention @ v).permute(0, 2, 1, 3)).reshape(batch_size, seq_length, -1)
        output = self.output_linear(output)
        return output


class FeedForwardBlock(nn.Module):
    def __init__(self, dimension_per_length=144, dimension_hidden=3, dropout_prob=0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(dimension_per_length, dimension_hidden)
        self.linear2 = nn.Linear(dimension_hidden, dimension_per_length)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def __call__(self, inputs):
        return self.linear2(self.dropout(self.relu(self.linear1(inputs))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dimension_per_length=144):
        super(TransformerEncoderBlock, self).__init__()
        self.mutihead_attention = MultiHeadAttention()
        self.feed_forward_block = FeedForwardBlock()
        self.layer_norm1 = nn.LayerNorm(dimension_per_length)
        self.layer_norm2 = nn.LayerNorm(dimension_per_length)

    def __call__(self, inputs):
        x = self.mutihead_attention(inputs) + inputs
        x = self.layer_norm1(x)
        x = self.feed_forward_block(x) + x
        x = self.layer_norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, channel, depth=2):
        super(TransformerEncoder, self).__init__()
        self.depth = depth
        self.con1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2)
        self.con2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2)
        self.con3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=1)

        self.contranspose1 = nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2)
        self.contranspose2 = nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2)
        self.contranspose3 = nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=1)

        self.transformer_encoder_block = TransformerEncoderBlock()

    def __call__(self, inputs, position_embedding):
        """

        :param inputs: [batch_size, seq_len, H, W]
        :return:
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        # print(seq_len)
        inputs = self.con1(inputs)
        inputs = self.con2(inputs)
        inputs = self.con3(inputs)
        w = inputs.shape[2]
        inputs = inputs.view(batch_size, seq_len, -1)
        # print(inputs.shape)

        # position embedding
        inputs += position_embedding

        for _ in range(self.depth):
            inputs = self.transformer_encoder_block(inputs)
        inputs = inputs.view(batch_size, seq_len, w, w)
        inputs = self.contranspose3(inputs)
        inputs = self.contranspose2(inputs)
        inputs = self.contranspose1(inputs)

        return inputs


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    inputs = torch.rand([16, 296, 64, 64])  # [batch_size, seq_len, W , W]
    position_embedding = create_1d_absolute_sin_cos_embedding(pos_len=296, dim=144)

    model = TransformerEncoder(channel=inputs.shape[1])
    print(model(inputs, position_embedding))

    # summary(TransformerEncoder(channel=inputs.shape[1]), input_size=[(296, 64, 64)], batch_size=8)
    # model = TransformerEncoder()
    #
    # img = Image.open('penguin.jpg')
    #
    # fig = plt.figure()
    # plt.imshow(img)
    # plt.show()
    #
    # print(model(input))

    # loss_fn = nn.MSELoss()
    #
    # optimizer = torch.optim.Adam(TransformerEncoder.parameters(), lr=0.01)
    #
    # # 记录训练的次数
    # total_train_step = 0
    # # 训练测试的次数
    # total_test_step = 0
    # #训练的次数
    # epoch = 10
    #
    # for i in range(epoch):
    #     print("------第{}轮训练开始------".format(i+1))
    #
    #     tudui.train()
    #     for data in train_dataloader:
    #         imgs, traget = data
    #         outputs = tudui(imgs)
    #         loss = loss_fn(outputs, targets)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_train_step = total_train_step + 1
    #         print("训练次数:{}， Loss:{}".format(total_train_step, loss.item()))
    #
    #     #测试部分
    #     tudui.eval()
    #     total_test_loss = 0
    #     with torch.no_grad():
    #         for data in test_dataloader:
    #             imgs, targets = data
    #             outputs = tudui(imgs)
    #             loss = loss_fn(outputs, targets)
    #             total_test_loss = total_test_loss + loss.item()
