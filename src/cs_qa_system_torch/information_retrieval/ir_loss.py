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