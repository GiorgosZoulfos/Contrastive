import torch

class Classifier(torch.nn.Module):
    
    def __init__(self, loss_function):
        super(Classifier, self).__init__()
        
        self.loss_fun = loss_function
        self.fully_connected_level =  torch.nn.Linear(10, 10)
        
    def forward(self, x):
        
        x = torch.nn.functional.relu(x)
        x = self.fully_connected_level(x)
        
        return x