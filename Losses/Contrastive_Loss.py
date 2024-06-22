import torch

class Contrastive_Loss(torch.nn.Module):
    
    def __init__(self, margin=1.0):
        super(Contrastive_Loss, self).__init__()
        self.margin = margin
        
    def forward(self, output, target):
        
        if len(output) % 2 != 0:
            output = output[:-1]
            target = target[:-1]
        
        # Eucledean distance of each pair of data
        Distance_matrix = (1e-8 + (output.unsqueeze(1) - output.unsqueeze(0)).pow(2).sum(dim=2)).sqrt()

        # Label matrix has 1 in cell (i,j) if data i and data j are from the same class, else 0
        Label_matrix = (target.unsqueeze(1) == target.unsqueeze(0)).long()
        
        # Create upper triangular mask
        Mask = torch.triu(torch.ones_like(Label_matrix), diagonal=1)
        
        # Loss for similar and dissimilar data
        loss_for_similar_data = Label_matrix * torch.pow(Distance_matrix, 2)
        loss_for_dissimalar_data = (1-Label_matrix) * torch.pow(torch.clamp(self.margin - Distance_matrix, min=0.0), 2)
        
        # Total loss
        loss = Mask * (loss_for_similar_data + loss_for_dissimalar_data)
        loss = loss.sum() / Mask.sum()

        return loss