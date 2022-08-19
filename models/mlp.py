__author__ = "Jie Lei"

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hsz, n_layers, input_dropout_p=0.2, hidden_dropout_p=0.5):
        super(MLP, self).__init__()

        self.input_dropout = nn.Dropout(input_dropout_p)

        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hsz),
                    nn.ReLU(True),
                    nn.Dropout(hidden_dropout_p)
                ])
                prev_dim = hsz

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_dropout(x)
        return self.main(x)


if __name__ == '__main__':
    test_in = torch.randn(10, 300)

    mlp1 = MLP(300, 1, 100, 1)
    print("="*20)
    print(mlp1)
    print(mlp1(test_in).size())

    mlp2 = MLP(300, 10, 100, 2)
    print("="*20)
    print(mlp2)
    print(mlp2(test_in).size())

    mlp3 = MLP(300, 5, 100, 4)
    print("=" * 20)
    print(mlp3)
    print(mlp3(test_in).size())
