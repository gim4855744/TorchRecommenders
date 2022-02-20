import torch

from torch.nn import Module, ModuleList
from torch.nn import Parameter
from torch.nn import Embedding, Linear
from torch.nn.init import zeros_, uniform_


class LR(Module):

    def __init__(self, num_features):

        super(LR, self).__init__()

        self._output_layer = Linear(num_features, 1)

        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self._output_layer.weight, -1e-4, 1e-4)
        zeros_(self._output_layer.bias)

    def forward(self, features):
        predicts = self._output_layer(features)
        return predicts


class MF(Module):

    def __init__(self, num_embeddings1, num_embeddings2, embedding_size):

        super(MF, self).__init__()

        self._embeddings1 = Embedding(num_embeddings1, embedding_size)
        self._embeddings2 = Embedding(num_embeddings2, embedding_size)

        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self._embeddings1.weight, -1e-4, 1e-4)
        uniform_(self._embeddings2.weight, -1e-4, 1e-4)

    def forward(self, ids1, ids2):

        embeddings1 = self._embeddings1(ids1)
        embeddings2 = self._embeddings2(ids2)

        predicts = (embeddings1 * embeddings2).sum(dim=1, keepdim=True)

        return predicts


class TF(Module):

    def __init__(self, *num_embeddings, embedding_size):

        super(TF, self).__init__()

        self._embeddings = ModuleList(
            [Embedding(num_embeddings[i], embedding_size) for i in range(len(num_embeddings))]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self._embeddings:
            uniform_(embedding.weight, -1e-4, 1e-4)

    def forward(self, *ids):
        embeddings = [embedding(ids) for embedding, ids in zip(self._embeddings, ids)]

        predicts = embeddings[0]
        for embedding in embeddings[1:]:
            predicts = predicts * embedding
        predicts = predicts.sum(dim=1, keepdim=True)

        return predicts


class FM(Module):

    def __init__(self, num_features, emb_size):

        super(FM, self).__init__()

        self._embeddings = ModuleList([Linear(1, emb_size, bias=False) for _ in range(num_features)])
        self._bias = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self._embeddings:
            uniform_(embedding.weight, -1e-4, 1e-4)
        zeros_(self._bias)

    def forward(self, features):

        embeddings = []
        for i, embedding in enumerate(self._embeddings):
            embeddings.append(embedding(features[:, i].view(-1, 1)))
        embeddings = torch.stack(embeddings, dim=1)

        sum_square = embeddings.sum(dim=1).pow(2)
        square_sum = embeddings.pow(2).sum(dim=1)
        predicts = (sum_square - square_sum).sum(dim=1, keepdim=True)

        return predicts


class MLP(Module):

    def __init__(self, num_features):

        super(MLP, self).__init__()

        num_layers = 3
        units = 64

        self._input_layer = Linear(num_features, units)
        self._layers = ModuleList([Linear(units, units) for _ in range(num_layers - 1)])
        self._output_layer = Linear(units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self._input_layer.weight, -1e-4, 1e-4)
        zeros_(self._input_layer.bias)
        for layer in self._layers:
            uniform_(layer.weight, -1e-4, 1e-4)
            zeros_(layer.bias)
        uniform_(self._output_layer.weight, -1e-4, 1e-4)
        zeros_(self._output_layer.bias)

    def forward(self, features):

        predicts = self._input_layer(features)
        for layer in self._layers:
            predicts = layer(predicts)
        predicts = self._output_layer(predicts)

        return predicts
