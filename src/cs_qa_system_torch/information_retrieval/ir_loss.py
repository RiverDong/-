from typing import Tuple
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import torch.tensor as T


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)

class BiEncoderBCELoss(object):
    def __call__(
        self,
        q_vectors: T, #10x768
        ctx_vectors: T, #30*768 => 10x30
        label_vector: T
    ):
        scores = torch.sum(q_vectors*ctx_vectors,dim=1)
        loss = BCEWithLogitsLoss()
        return loss(scores,label_vector)

class BiEncoderNllLoss(object):
    def __call__(
        self,
        q_vectors: T, #10x768
        ctx_vectors: T, #30*768 => 10x30
        positive_idx_per_question: T, # [2,4,7,..30] length = 10
        hard_negatice_idx_per_question: list = None,
        weight_factor = 0.8
    ):
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)
        if hard_negatice_idx_per_question is not None:
            weights = torch.ones([q_vectors.shape[0],ctx_vectors.shape[0]]).cuda()
            for i,l in enumerate(hard_negatice_idx_per_question):
                weights[i,l] = 1.0*(1+weight_factor)
            softmax_scores = softmax_scores*weights
        loss = F.nll_loss(
            softmax_scores,
            positive_idx_per_question,
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == positive_idx_per_question
        ).sum()
        return loss

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


import torch.tensor as T


class TripletLoss(object):
    def __call__(
            self,
            q_vectors: T,  # (batch_sizex768)
            ctx_vectors: T,  # (2*batch_sizex768)
            positive_idx_per_question: T,
            margin=0.3
    ):
        scores = self.get_euclidean_distance(q_vectors, ctx_vectors)
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)
        distance_matrix = torch.zeros((len(q_vectors), 2))
        for i in range(len(q_vectors)):
            distance_matrix[i, 0:2] = scores[i, 2 * i:2 * i + 2]
        loss = distance_matrix[:, 0] - distance_matrix[:, 1] + margin
        loss[loss<0] = 0
        loss = loss.mean()
        return loss

    def get_cosinesimilarity_distance(self, q_vector, ctx_vector):
        q_vector = F.normalize(q_vector, dim=1)
        ctx_vector = F.normalize(ctx_vector, dim=1)
        return torch.matmul(q_vector, torch.transpose(ctx_vector, 0, 1))

    def get_euclidean_distance(self, q_vector, ctx_vector):
        return torch.cdist(q_vector, ctx_vector)