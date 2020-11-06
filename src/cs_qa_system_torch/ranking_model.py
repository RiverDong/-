import numpy as np
import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, DistilBertModel

class BertBiEncoderModel(BertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.bert = kwargs['bert']
    try:
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.query_fc = nn.Linear(config.hidden_size, 128)
      self.context_fc = nn.Linear(config.hidden_size, 128)

    except:
      self.dropout = nn.Dropout(config.dropout)
      self.query_fc = nn.Linear(config.dim, 128)
      self.context_fc = nn.Linear(config.dim, 128)

  def forward(self, query_input_ids, query_attention_mask, query_segment_ids,
              context_input_ids,  context_attention_mask, context_segment_ids):

    if isinstance(self.bert, DistilBertModel):
      query_vec = self.bert(query_input_ids, query_attention_mask)[-1]  # [bs,dim]
      query_vec = query_vec[:, 0]
      context_vec = self.bert(context_input_ids, context_attention_mask)[-1]  # [bs,dim]
      context_vec = context_vec[:, 0]
    else:
      query_vec = self.bert(query_input_ids, query_attention_mask, query_segment_ids)[-1]  # [bs,dim]
      context_vec = self.bert(context_input_ids, context_attention_mask, context_segment_ids)[-1]  # [bs,dim]

    context_vec = self.context_fc(self.dropout(context_vec))
    context_vec = F.normalize(context_vec, 2, -1)

    query_vec = self.query_fc(self.dropout(query_vec))
    query_vec = F.normalize(query_vec, 2, -1)

    dot_product = torch.sum(context_vec*query_vec,dim=1)
    return dot_product

class BertCrossEncoderModel(BertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.bert = kwargs['bert']
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

    if isinstance(self.bert, DistilBertModel):
      bert_vec = self.bert(input_ids, attention_mask)[-1]  # [bs,dim]
      bert_vec = bert_vec[:, 0]
    else:
      bert_vec = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[-1]  # [bs,dim]
    bert_vec = torch.tanh(bert_vec)
    bert_vec = self.fc1(self.dropout(bert_vec))
    # bert_vec = self.fc11(self.dropout(bert_vec))
    # bert_vec = torch.tanh(bert_vec)
    bert_vec = self.fc2(bert_vec)
    bert_vec = torch.sigmoid(bert_vec)
    output = torch.squeeze(bert_vec)
    return output