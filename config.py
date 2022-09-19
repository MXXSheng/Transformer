import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="Parameters of Transformer")
    parser.add_argument('--pos_len', type=int, default=500, help="The number of images when training")
    parser.add_argument('--prediction_number', type=int, default=500, help="The prediction number")
    parser.add_argument('--dimension_per_length', type=int, default=144, help="The dimension of an image after three convs")
    parser.add_argument('--epoch', type=int, default=600, help="The epoch")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="The learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size")
    parser.add_argument('--depth', type=int, default=4, help="The depth of mutihead self-attention")
    parser.add_argument('--dimension_hidden', type=int, default=32, help="The linear dimension")
    parser.add_argument('--dropout_prob', type=float, default=0.4, help="The possibility of dropout")
    parser.add_argument('--filename', type=str, default="matrices_of_path.csv", help="The path summary file name")
    parser.add_argument('--head', type=int, default=8, help="The head of mutihead self-attention")
    parser.add_argument('--dimension_qkv', type=int, default=512, help="The dimension of q,k,v")
    return parser.parse_args()
