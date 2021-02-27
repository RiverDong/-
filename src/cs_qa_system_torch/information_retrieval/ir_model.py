import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
from transformers import DistilBertModel, BertModel, BertForSequenceClassification, ElectraModel


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


def get_enonding_vector(model, input_ids, attention_mask, segment_ids):
  if isinstance(model, DistilBertModel):
    vec = model(input_ids=input_ids, attention_mask=attention_mask)[-1]  # (batch_size, sequence_length, hidden_size)
    vec = vec[:, 0]
  elif isinstance(model, ElectraModel):
    vec = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[-1]  # (batch_size, sequence_length, hidden_size)
    vec = vec[:, 0]
  elif isinstance(model, BertModel):
    vec = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[-1]  # [bs,dim]
  elif isinstance(model, BertForSequenceClassification):
    raise ValueError("Wrong Input Model Given for Biencoder")
  else:
    raise ValueError("Wrong Input Model Given for Biencoder")
  return vec

class BiEncoderModel(nn.Module):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__()
    self.query_model = kwargs['query_model']
    self.context_model = kwargs['context_model']
    self.project_dim = kwargs['project_dim']
    self.encode_query_proj = (nn.Linear(config.hidden_size, self.project_dim) if self.project_dim != 0 else None)
    self.encode_document_proj = (nn.Linear(config.hidden_size, self.project_dim) if self.project_dim != 0 else None)

  def forward(self, query_input_ids, query_attention_mask, query_segment_ids,
              context_input_ids,  context_attention_mask, context_segment_ids):

    query_vec = get_enonding_vector(self.query_model, query_input_ids, query_attention_mask, query_segment_ids)
    if self.encode_query_proj:
      query_vec = self.encode_query_proj(query_vec)

    context_vec = get_enonding_vector(self.context_model, context_input_ids, context_attention_mask, context_segment_ids)
    if self.encode_document_proj:
      context_vec = self.encode_document_proj(context_vec)

    return query_vec, context_vec

class CrossEncoderModel(nn.Module):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__()
    self.model = kwargs['model']
    try:
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.fc1 = nn.Linear(config.hidden_size, 128)
      # self.fc11 = nn.Linear(128, 128)
      self.fc2 = nn.Linear(128,1)

    except:
      self.dropout = nn.Dropout(config.dropout)
      self.fc1 = nn.Linear(config.hidden_size, 128)
      # self.fc11 = nn.Linear(128, 128)
      self.fc2 = nn.Linear(128,1)

  def forward(self, input_ids,  attention_mask, segment_ids):

    if isinstance(self.model, DistilBertModel):
      bert_vec = self.model(input_ids, attention_mask)[-1]  # (batch_size, sequence_length, hidden_size)
      # Taking cls token vectors of dimension (batch_size,hidden_size)
      bert_vec = bert_vec[:, 0]
    elif isinstance(self.model, ElectraModel):
      bert_vec = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[-1] # (batch_size, sequence_length, hidden_size)
      # Taking cls token vectors of dimension (batch_size,hidden_size)
      bert_vec = bert_vec[:, 0]
    elif isinstance(self.model, BertModel):
      bert_vec = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[-1]  # pooled vector (batch_size,hidden_size)
    elif isinstance(self.model,BertForSequenceClassification):
      bert_vec = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
      bert_vec = torch.nn.functional.softmax(bert_vec, dim=1)
      return bert_vec[:,-1]
    else:
      raise ValueError("Wrong Input Model Given for crossencoder")
    bert_vec = torch.tanh(bert_vec)
    bert_vec = self.fc1(self.dropout(bert_vec))
    # bert_vec = self.fc11(self.dropout(bert_vec))
    # bert_vec = torch.tanh(bert_vec)
    bert_vec = self.fc2(bert_vec)
    bert_vec = torch.sigmoid(bert_vec)
    output = torch.squeeze(bert_vec)
    return output