import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv, GCN2Conv


class MPNNs(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=2,
                 dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False,
                 ln=False, bn=False, jk=False, gnn='gcn', beta=0.005):
        super(MPNNs, self).__init__()

        self.dropout = dropout
        self.pre_ln = pre_ln
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.gnn = gnn
        self.hidden_channels = hidden_channels
        self.beta = beta
        self.local_layers = local_layers

        self.h_lins = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = nn.ModuleList()

        if gnn == 'sgc':
            self.local_convs.append(SGConv(in_channels, hidden_channels, K=local_layers, cached=True))
        elif gnn == 'gcnii':
            self.lin_in = nn.Linear(in_channels, hidden_channels)
            self.gcnii_alpha = 0.1
            self.gcnii_lambda = 0.5
            self.gcnii_convs = nn.ModuleList([
                GCN2Conv(hidden_channels, alpha=self.gcnii_alpha, theta=self.gcnii_lambda,
                         layer=i + 1, shared_weights=True)
                for i in range(local_layers)
            ])
        else:
            self.lin_in = nn.Linear(in_channels, hidden_channels)

            if not self.pre_linear:
                if gnn == 'gat':
                    self.local_convs.append(
                        GATConv(in_channels, hidden_channels, heads=heads, concat=True,
                                add_self_loops=False, bias=False)
                    )
                elif gnn == 'sage':
                    self.local_convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.local_convs.append(
                        GCNConv(in_channels, hidden_channels, cached=False, normalize=True)
                    )
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.lns.append(nn.LayerNorm(hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                if self.pre_ln:
                    self.pre_lns.append(nn.LayerNorm(in_channels))
                local_layers -= 1

            for _ in range(local_layers):
                if gnn == 'gat':
                    self.local_convs.append(
                        GATConv(hidden_channels, hidden_channels, heads=heads, concat=True,
                                add_self_loops=False, bias=False)
                    )
                elif gnn == 'sage':
                    self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
                else:
                    self.local_convs.append(
                        GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True)
                    )
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.lns.append(nn.LayerNorm(hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                if self.pre_ln:
                    self.pre_lns.append(nn.LayerNorm(hidden_channels))

        self.prototypes = nn.Linear(hidden_channels, out_channels, bias=False)

    def reset_parameters(self):
        if self.gnn == 'gcnii':
            self.lin_in.reset_parameters()
            for conv in self.gcnii_convs:
                conv.reset_parameters()
        else:
            for conv in self.local_convs:
                conv.reset_parameters()
            for lin in self.lins:
                lin.reset_parameters()
            for ln in self.lns:
                ln.reset_parameters()
            for bn in self.bns:
                bn.reset_parameters()
            if self.pre_ln:
                for pln in self.pre_lns:
                    pln.reset_parameters()
            if hasattr(self, 'lin_in'):
                self.lin_in.reset_parameters()
        self.prototypes.reset_parameters()

    def forward(self, x, edge_index):

        if self.gnn == 'gcnii':
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h0 = x
            for conv in self.gcnii_convs:
                x = conv(x, h0, edge_index)
                x_norm = F.normalize(x, p=2, dim=0)
                orthogonal_message = x_norm @ (x_norm.t() @ x_norm)
                x = x - self.beta * orthogonal_message
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            logits = self.prototypes(x)
            return logits, x

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0
        for i, conv in enumerate(self.local_convs):
            if self.res:
                x = conv(x, edge_index) + self.lins[i](x)
            else:
                x = conv(x, edge_index)

            x_norm = F.normalize(x, p=2, dim=0)
            orthogonal_message = x_norm @ (x_norm.t() @ x_norm)
            x = x - self.beta * orthogonal_message

            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jk:
                x_final += x
            else:
                x_final = x

        logits = self.prototypes(x_final)
        return logits, x_final
