import torch

class Supervised_Contrastive_Loss(torch.nn.Module):
    
    def __init__(self, temperature=0.1):
        super(Supervised_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        
    def forward(self, output, target):
        
        # Compute cosine similarity for each pair of data and normalize with temperature
        cosine_sim_matrix = torch.matmul(output, output.T) / self.tau
        cosine_sim_matrix = torch.exp(cosine_sim_matrix)
        # Exclude the similarity of each data with itself
        cosine_sim_matrix.fill_diagonal_(0)
        # L1 norm of each row
        cosine_sim_matrix = torch.nn.functional.normalize(cosine_sim_matrix, p=1, dim=1)
        
        
        # Label matrix has 1 in cell (i,j) if data i and data j are from the same class, else 0 
        Label_matrix = (target.unsqueeze(1) == target.unsqueeze(0)).long()
        # Exclude the diagonal
        Label_matrix = Label_matrix - torch.eye(len(target)).to(target.device)
        
        
        # Compute loss for this batch
        loss = - torch.log((cosine_sim_matrix * Label_matrix).sum(dim=1)).sum() 
        return loss / len(target)