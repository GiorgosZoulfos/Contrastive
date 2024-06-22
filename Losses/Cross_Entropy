import torch

class Cross_Entropy(torch.nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss_fn(output, target)