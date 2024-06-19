import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

class B3DBModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, dropout_rate=0.3):
        super(B3DBModel, self).__init__()

        self.conv1 = torch_geometric.nn.GINConv(
            nn.Sequential(
                nn.Linear(input_dim,  hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = torch_geometric.nn.GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim,     hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

        self.conv3 = torch_geometric.nn.GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim * 4),
            )
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)

        self.conv4 = torch_geometric.nn.GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )
        )
        self.bn4 = nn.BatchNorm1d(hidden_dim * 2)

        self.attn1 = torch_geometric.nn.GATConv(hidden_dim * 2, hidden_dim, heads=num_heads, concat=False)
        self.bn5 = nn.BatchNorm1d(hidden_dim)

        self.pool = torch_geometric.nn.GlobalAttention(gate_nn=torch.nn.Linear(hidden_dim, 1))
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.attn1(x, edge_index)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.pool(x, batch)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x