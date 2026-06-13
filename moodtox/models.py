from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import GRUCell
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.utils import softmax

from .features import atom_feature_dim, bond_feature_dim, molecule_to_graph


class BondAwareGATEConv(MessagePassing):
    """Official AttentiveFP GATEConv extended to every atom layer."""

    def __init__(
        self,
        hidden_dim: int,
        bond_dim: int,
        dropout: float,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.dropout = dropout
        self.neighbor_projection = nn.Linear(
            hidden_dim + bond_dim,
            hidden_dim,
            bias=False,
        )
        self.message_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.neighbor_attention = nn.Parameter(
            torch.empty(1, hidden_dim)
        )
        self.center_attention = nn.Parameter(
            torch.empty(1, hidden_dim)
        )
        self.bias = nn.Parameter(torch.empty(hidden_dim))
        self.last_attention = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.neighbor_projection.weight)
        nn.init.xavier_uniform_(self.message_projection.weight)
        nn.init.xavier_uniform_(self.neighbor_attention)
        nn.init.xavier_uniform_(self.center_attention)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        attention = self.edge_updater(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        self.last_attention = attention
        output = self.propagate(
            edge_index,
            x=x,
            attention=attention,
        )
        return output + self.bias

    def edge_update(
        self,
        x_j,
        x_i,
        edge_attr,
        index,
        ptr,
        size_i,
    ):
        neighbor = F.leaky_relu(
            self.neighbor_projection(
                torch.cat([x_j, edge_attr], dim=-1)
            )
        )
        neighbor_score = (
            neighbor @ self.neighbor_attention.t()
        ).squeeze(-1)
        center_score = (
            x_i @ self.center_attention.t()
        ).squeeze(-1)
        attention = F.leaky_relu(neighbor_score + center_score)
        attention = softmax(attention, index, ptr, size_i)
        return F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )

    def message(self, x_j, attention):
        return (
            self.message_projection(x_j)
            * attention.unsqueeze(-1)
        )


class BackboneEncoder(nn.Module):
    """AttentiveFP with its official flow and bond-aware later atom layers."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be at least 1")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.embedding = nn.Linear(atom_feature_dim(), hidden_dim)
        self.gate_conv = BondAwareGATEConv(
            hidden_dim,
            bond_feature_dim(),
            dropout,
        )
        self.gru = GRUCell(hidden_dim, hidden_dim)
        self.atom_convs = nn.ModuleList(
            [
                BondAwareGATEConv(
                    hidden_dim,
                    bond_feature_dim(),
                    dropout,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.atom_grus = nn.ModuleList(
            [
                GRUCell(hidden_dim, hidden_dim)
                for _ in range(num_layers - 1)
            ]
        )

        self.molecule_conv = GATConv(
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.molecule_gru = GRUCell(hidden_dim, hidden_dim)
        self.last_bond_attentions: list[torch.Tensor] = []
        self.last_atom_attention = None

    def forward(self, data, return_nodes: bool = False):
        batch = getattr(
            data,
            "batch",
            torch.zeros(
                data.x.size(0),
                dtype=torch.long,
                device=data.x.device,
            ),
        )
        x = F.leaky_relu(self.embedding(data.x))
        message = F.elu(
            self.gate_conv(
                x,
                data.edge_index,
                data.edge_attr,
            )
        )
        message = F.dropout(
            message,
            p=self.dropout,
            training=self.training,
        )
        x = self.gru(message, x).relu()
        bond_attentions = [self.gate_conv.last_attention]

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            message = F.elu(
                conv(
                    x,
                    data.edge_index,
                    data.edge_attr,
                )
            )
            message = F.dropout(
                message,
                p=self.dropout,
                training=self.training,
            )
            x = gru(message, x).relu()
            bond_attentions.append(conv.last_attention)

        atom_rows = torch.arange(
            batch.size(0),
            device=batch.device,
        )
        atom_to_molecule = torch.stack([atom_rows, batch], dim=0)
        molecule = global_add_pool(x, batch).relu()
        atom_attention = None
        for _ in range(self.num_timesteps):
            message, (_, atom_attention) = self.molecule_conv(
                (x, molecule),
                atom_to_molecule,
                return_attention_weights=True,
            )
            message = F.elu(message)
            message = F.dropout(
                message,
                p=self.dropout,
                training=self.training,
            )
            molecule = self.molecule_gru(
                message,
                molecule,
            ).relu()

        molecule = F.dropout(
            molecule,
            p=self.dropout,
            training=self.training,
        )
        self.last_bond_attentions = bond_attentions
        self.last_atom_attention = atom_attention.mean(dim=-1)
        if return_nodes:
            return molecule, x, self.last_atom_attention
        return molecule


class EnvironmentClassifier(nn.Module):
    def __init__(
        self,
        graph_feat_size: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float,
        k: int,
    ):
        super().__init__()
        self.backbone = BackboneEncoder(
            graph_feat_size,
            num_layers,
            num_timesteps,
            dropout,
        )
        self.predictor = nn.Linear(graph_feat_size + 1, k)

    def forward(self, graph_batch):
        features = self.backbone(graph_batch)
        labels = graph_batch.y.view(-1, 1).float()
        return self.predictor(
            torch.cat([features, labels], dim=-1)
        )


class ConditionalEnvironmentPredictor(nn.Module):
    def __init__(
        self,
        graph_feat_size: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float,
        k: int,
    ):
        super().__init__()
        self.backbone = BackboneEncoder(
            graph_feat_size,
            num_layers,
            num_timesteps,
            dropout,
        )
        self.environment_embeddings = nn.Parameter(
            torch.zeros(k, graph_feat_size)
        )
        self.predictor = nn.Linear(graph_feat_size * 2, 1)

    def forward(self, graph_batch, environment_ids):
        features = self.backbone(graph_batch)
        environment_features = self.environment_embeddings[
            environment_ids
        ]
        return self.predictor(
            torch.cat([features, environment_features], dim=-1)
        ).squeeze(-1)

    def forward_all_environments(self, graph_batch):
        features = self.backbone(graph_batch)
        batch_size = features.size(0)
        environment_count = self.environment_embeddings.size(0)
        graph_features = features.unsqueeze(1).expand(
            batch_size,
            environment_count,
            -1,
        )
        environment_features = (
            self.environment_embeddings.unsqueeze(0).expand(
                batch_size,
                environment_count,
                -1,
            )
        )
        return self.predictor(
            torch.cat(
                [graph_features, environment_features],
                dim=-1,
            )
        ).squeeze(-1)


class MoodTOXModel(nn.Module):
    def __init__(
        self,
        graph_feat_size: int,
        auxiliary_feat_size: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float,
    ):
        super().__init__()
        self.molecular_backbone = BackboneEncoder(
            graph_feat_size,
            num_layers,
            num_timesteps,
            dropout,
        )
        self.fragment_backbone = BackboneEncoder(
            auxiliary_feat_size,
            num_layers,
            num_timesteps,
            dropout,
        )
        attention_dim = max(
            graph_feat_size,
            auxiliary_feat_size,
        )
        self.query = nn.Linear(graph_feat_size, attention_dim)
        self.key = nn.Linear(auxiliary_feat_size, attention_dim)
        predictor_layers = [
            nn.Linear(auxiliary_feat_size, auxiliary_feat_size)
        ]
        if 0.0 < dropout < 1.0:
            predictor_layers.append(nn.Dropout(dropout))
        predictor_layers.extend(
            [
                nn.ReLU(),
                nn.Linear(auxiliary_feat_size, 1),
            ]
        )
        self.predictor = nn.Sequential(*predictor_layers)
        self.last_substructure_attention: list[
            torch.Tensor
        ] = []
        self._fragment_graph_cache = {}

    def _encode_fragments(self, substructure_lists, device):
        flat_graphs = []
        lengths = []
        for fragments in substructure_lists:
            fragments = list(fragments)
            lengths.append(len(fragments))
            for smiles in fragments:
                graph = self._fragment_graph_cache.get(smiles)
                if graph is None:
                    graph = molecule_to_graph(smiles)
                    self._fragment_graph_cache[smiles] = graph
                flat_graphs.append(graph)
        batch = Batch.from_data_list(flat_graphs).to(device)
        encoded = self.fragment_backbone(batch)
        return encoded.split(lengths)

    def forward(self, graph_batch, substructure_lists):
        molecule_features = self.molecular_backbone(graph_batch)
        fragment_groups = self._encode_fragments(
            substructure_lists,
            graph_batch.x.device,
        )
        contexts = []
        attentions = []
        for molecule_feature, fragment_features in zip(
            molecule_features,
            fragment_groups,
        ):
            query = self.query(molecule_feature).unsqueeze(0)
            keys = self.key(fragment_features)
            scores = (
                query @ keys.t()
            ).squeeze(0) / math.sqrt(keys.size(-1))
            weights = torch.softmax(scores, dim=0)
            contexts.append(
                torch.sum(
                    fragment_features * weights.unsqueeze(-1),
                    dim=0,
                )
            )
            attentions.append(weights)
        self.last_substructure_attention = attentions
        context = torch.stack(contexts)
        return self.predictor(context).squeeze(-1)
