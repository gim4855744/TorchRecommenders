import torch

from torch.nn import Module
from torch.nn import Embedding
from torch.nn.init import uniform_


class MF(Module):

    """
    Matrix Factorization (MF)
    MF is a collaborative filtering model that learns second-order feature interactions (e.g., user-item interactions).
    """

    def __init__(self, num_embeddings1, num_embeddings2, embedding_size):

        """
        initializer
        :param num_embeddings1:
        :param num_embeddings2:
        :param embedding_size:
        """

        super(MF, self).__init__()

        self._embeddings1 = Embedding(num_embeddings1, embedding_size)
        self._embeddings2 = Embedding(num_embeddings2, embedding_size)

    def reset_parameters(self):
        uniform_(self._embeddings1.weight, -1e-4, 1e-4)
        uniform_(self._embeddings2.weight, -1e-4, 1e-4)

    def forward(self, ids1, ids2):

        """
        forward propagation
        :param ids1:
        :param ids2:
        :return:
        """

        embeddings1 = self._embeddings1(ids1)
        embeddings2 = self._embeddings2(ids2)

        predicts = torch.sum(embeddings1 * embeddings2, dim=1)

        return predicts
