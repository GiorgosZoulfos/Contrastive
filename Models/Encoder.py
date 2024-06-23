import torch

class Encoder(torch.nn.Module):
    
    def __init__(self, loss_function, visualization):
        super(Encoder, self).__init__()
        
        self.loss_function = loss_function
        self.visualization = visualization

        self.convolutional_level_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.convolutional_level_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        
        self.dropout = torch.nn.Dropout2d()
        
        self.fully_connected_level_1 = torch.nn.Linear(320, 50)
        if self.visualization == 'yes':
            self.fully_connected_level_2 = torch.nn.Linear(50, 2)
        else:
            self.fully_connected_level_2 = torch.nn.Linear(50, 10)
            
        
    def forward(self, x):
        
        # First Layer
        x = self.convolutional_level_1(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(x)
        
        # Second Layer
        x = self.convolutional_level_2(x)
        x = self.dropout(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(x)
        
        # Flatten the data before the fully connected layers
        x = x.view(-1, 320)
        
        # Third Layer
        x = self.fully_connected_level_1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x)
        
        # Fourth Layer
        x = self.fully_connected_level_2(x)
        
        # Normalize output data
        if self.loss_function in ['supcon', 'simclr']:
            x = torch.nn.functional.normalize(x, dim=1)
            
        return x
    
        
        
        
        
        