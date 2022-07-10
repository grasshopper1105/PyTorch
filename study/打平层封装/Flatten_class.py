from torch import nn


class Flatten(nn.Module):
    """ 打平数据，方便送入Linear层"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)
