### CONFIGS ###
dataset = 'cora'
model = 'VGAE'

input_dim = 1433
hidden1_dim = 32
hidden2_dim = 16
use_feature = True

num_epoch = 200
learning_rate = 0.01


# import argparse
#
# parser = argparse.ArgumentParser(description="Variant Graph Auto Encoder")
# parser.add_argument(
#     "--learning_rate", type=float, default=0.01, help="Initial learning rate."
# )
# parser.add_argument(
#     "--epochs", "-e", type=int, default=200, help="Number of epochs to train."
# )
# parser.add_argument(
#     "--hidden1",
#     "-h1",
#     type=int,
#     default=32,
#     help="Number of units in hidden layer 1.",
# )
# parser.add_argument(
#     "--hidden2",
#     "-h2",
#     type=int,
#     default=16,
#     help="Number of units in hidden layer 2.",
# )
# parser.add_argument(
#     "--datasrc",
#     "-s",
#     type=str,
#     default="dgl",
#     help="Dataset download from dgl Dataset or website.",
# )
# parser.add_argument(
#     "--dataset", "-d", type=str, default="cora", help="Dataset string."
# )
# parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")
# args = parser.parse_args()