import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VGAE(nn.Module):
    def __init__(self, embedding, in_dim, hidden_dim, out_dim):
        super(VGAE, self).__init__()
        self.embedding = embedding
        
        # Encoder: Shared first layer
        self.conv1 = GCNConv(in_dim, hidden_dim)
        
        # Variational layers: Separate layers for mean and log variance
        self.conv_mu = GCNConv(hidden_dim, out_dim)    # Mean
        self.conv_logvar = GCNConv(hidden_dim, out_dim) # Log variance

    def encode(self, x, edge_index):
        x = self.embedding(x).squeeze()
        x = self.conv1(x, edge_index).relu()
        
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: Sample z ~ N(mu, sigma^2)
        if self.training:
            std = torch.exp(logvar * 0.5)  # Standard deviation
            eps = torch.randn_like(std)    # Random noise
            return mu + eps * std
        else:
            # During inference, use mean directly
            return mu

    def decode(self, z):
        # Inner product decoder with sigmoid activation
        return torch.sigmoid(z @ z.t())

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar  # Return decoded edges, mu, and logvar
    
class GAE(torch.nn.Module):
    def __init__(self, embedding, in_dim, hidden_dim, out_dim):
        super(GAE, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.embedding = embedding

    def encode(self, x, edge_index):
        x = self.embedding(x).squeeze()
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z):
        return torch.sigmoid(z @ z.t()) 
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z)