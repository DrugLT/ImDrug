# The model implementation is adopted from the dgllife library
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, SumPooling, AvgPooling, MaxPooling, Set2Set, GlobalAttentionPooling
from dgllife.model import WeightedSumAndMax



class GCNLayer(nn.Module):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__
    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    gnn_norm : str
        The message passing normalizer, which can be `'right'`, `'both'` or `'none'`. The
        `'right'` normalizer divides the aggregated messages by each node's in-degree.
        The `'both'` normalizer corresponds to the symmetric adjacency normalization in
        the original GCN paper. The `'none'` normalizer simply sums the messages.
        Default to be 'none'.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """

    def __init__(self, in_feats, out_feats, gnn_norm='none', activation=None,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm=gnn_norm, activation=activation,
                                    allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        """
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats

class GCN(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__
    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    """

    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(gnn_norm), len(activation),
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i], gnn_norm[i], activation[i],
                                            residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

        gnn_out_feats = self.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    # def forward(self, g, feats):
    #     """Update node representations.
    #
    #     Parameters
    #     ----------
    #     g : DGLGraph
    #         DGLGraph for a batch of graphs
    #     feats : FloatTensor of shape (N, M1)
    #         * N is the total number of nodes in the batch of graphs
    #         * M1 is the input node feature size, which equals in_feats in initialization
    #
    #     Returns
    #     -------
    #     feats : FloatTensor of shape (N, M2)
    #         * N is the total number of nodes in the batch of graphs
    #         * M2 is the output node representation size, which equals
    #           hidden_sizes[-1] in initialization.
    #     """
    #     for gnn in self.gnn_layers:
    #         feats = gnn(g, feats)
    #     return feats

    def forward(self, input):
        input = move_to_device(input)
        # input = dgl.add_self_loop(input)
        node_feats = input.ndata["x"]
        for gnn in self.gnn_layers:
            node_feats = gnn(input, node_feats)

        # node_feats = super().forward(input, feats=node_feats)
        graph_feats = self.readout(input, node_feats)
        return graph_feats

class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


# pylint: disable=W0221, C0103
class GINLayer(nn.Module):
    r"""Single Layer GIN from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    Parameters
    ----------
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    emb_dim : int
        The size of each embedding vector.
    batch_norm : bool
        Whether to apply batch normalization to the output of message passing.
        Default to True.
    activation : None or callable
        Activation function to apply to the output node representations.
        Default to None.
    """

    def __init__(self, num_edge_emb_list, emb_dim, batch_norm=True, activation=None):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embeddings = nn.ModuleList()
        for num_emb in num_edge_emb_list:
            emb_module = MLP(input_dim=num_emb, hidden_dim=emb_dim, output_dim=emb_dim, num_layers=1)
            self.edge_embeddings.append(emb_module)

        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        # for emb_module in self.edge_embeddings:
        #     nn.init.xavier_uniform_(emb_module.weight.data)

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, categorical_edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : FloatTensor of shape (N, emb_dim)
            * Input node features
            * N is the total number of nodes in the batch of graphs
            * emb_dim is the input node feature size, which must match emb_dim in initialization
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as len(self.edge_embeddings)
            * E is the total number of edges in the batch of graphs
        Returns
        -------
        node_feats : float32 tensor of shape (N, emb_dim)
            Output node representations
        """
        edge_embeds = []
        for i, feats in enumerate(categorical_edge_feats):
            edge_embeds.append(self.edge_embeddings[i](feats))
        edge_embeds = torch.stack(edge_embeds, dim=0).sum(0)
        g = g.local_var()
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_embeds
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))

        node_feats = self.mlp(g.ndata.pop('feat'))
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats

class GIN(nn.Module):
    r"""Graph Isomorphism Network from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    This module is for updating node representations only.
    Parameters
    ----------
    num_node_emb_list : list of int
        num_node_emb_list[i] gives the number of items to embed for the
        i-th categorical node feature variables. E.g. num_node_emb_list[0] can be
        the number of atom types and num_node_emb_list[1] can be the number of
        atom chirality types.
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 300.
    JK : str
        JK for jumping knowledge as in `Representation Learning on Graphs with
        Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__. It decides
        how we are going to combine the all-layer node representations for the final output.
        There can be four options for this argument, ``concat``, ``last``, ``max`` and ``sum``.
        Default to 'last'.
        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.5
    """

    def __init__(self, num_node_emb_list, num_edge_emb_list,
                 num_layers=5, emb_dim=300, JK='last', dropout=0., readout="mean", ):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.node_embeddings = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = MLP(input_dim=num_emb, hidden_dim=emb_dim, output_dim=emb_dim, num_layers=2)
            self.node_embeddings.append(emb_module)

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayer(num_edge_emb_list, emb_dim))
            else:
                self.gnn_layers.append(GINLayer(num_edge_emb_list, emb_dim, activation=F.relu))

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if JK == 'concat':
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear((num_layers + 1) * emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear(emb_dim, 1))
        elif readout == 'set2set':
            self.readout = Set2Set()
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max', 'attention' or 'set2set', got {}".format(readout))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        # for emb_module in self.node_embeddings:
        #     nn.init.xavier_uniform_(emb_module.weight.data)

        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, input):
        """Update node representations
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(self.node_embeddings)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs
        Returns
        -------
        final_node_feats : float32 tensor of shape (N, M)
            Output node representations, N for the number of nodes and
            M for output size. In particular, M will be emb_dim * (num_layers + 1)
            if self.JK == 'concat' and emb_dim otherwise.
        """
        input = move_to_device(input)
        categorical_node_feats = [input.ndata["x"]]
        categorical_edge_feats = [input.edata["x"]]

        node_embeds = []
        for i, feats in enumerate(categorical_node_feats):
            node_embeds.append(self.node_embeddings[i](feats))
        node_embeds = torch.stack(node_embeds, dim=0).sum(0)

        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](input, all_layer_node_feats[layer],
                                                categorical_edge_feats)
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.max(torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.sum(torch.cat(all_layer_node_feats, dim=0), dim=0)
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', "
                              "'max' or 'sum', got {}".format(self.JK))

        graph_feats = self.readout(input, final_node_feats)

        return graph_feats

