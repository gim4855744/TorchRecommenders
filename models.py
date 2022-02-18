import torch

from torch.nn import Module, ModuleList
from torch.nn import Embedding
from torch.nn.init import uniform_


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
