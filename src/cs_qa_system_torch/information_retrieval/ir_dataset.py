import os
import csv
from tqdm import tqdm
import json
import numpy as np

import torch
from torch.utils.data import Dataset

import constants
from utils import pickle_load, pickle_dump


class InferenceDataset(Dataset):
  def __init__(self, predict_list, query_transform, context_transform):
      self.predict_list = predict_list
      self.query_transform = query_transform
      self.context_transform = context_transform

  def __len__(self):
    return len(self.predict_list)

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
      qid, pid, query, context, score = self.predict_list[index]
      transformed_query = self.query_transform(query)  # [token_ids],[seg_ids],[masks]
      transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
      return (*transformed_query, *transformed_context)

class CombinedInferenceDataset(Dataset):
  def __init__(self, predict_list, transform):
      self.predict_list = predict_list
      self.transform = transform

  def __len__(self):
    return len(self.predict_list)

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
      qid, pid, query, context, score = self.predict_list[index]
      transformed_query_context = self.transform(query, context)  # [token_ids],[seg_ids],[masks]
      return transformed_query_context

class RankingDataset(Dataset):
  def __init__(self, file_path, query_transform, context_transform, delimiter='\t', overwrite_cache:bool=False):
    self.query_transform = query_transform
    self.context_transform = context_transform

    self.data_source = []
    self.transformed_data = {}
    cache_path = '.'.join(file_path.split(".")[:-1]) + '.cache'
    if os.path.exists(cache_path) and not overwrite_cache:
      self.transformed_data = pickle_load(cache_path)
      self.data_source = [0] * len(self.transformed_data)
    else:
      f = open(file_path, 'r', encoding='utf-8')
      data = csv.DictReader(f, delimiter=delimiter)

      for row in data:
          self.data_source.append((row[constants.RANKING_INPUT_QUERY_NAME], row[constants.RANKING_INPUT_DOCUMENT_NAME],
                         int(row[constants.RANKING_INPUT_LABEL_NAME]), row[constants.RANKING_INPUT_QUERY_ID]))

      for idx in tqdm(range(len(self.data_source))):
          self.__get_single_item__(idx)

      pickle_dump(self.transformed_data, cache_path)
      self.data_source = [0] * len(self.transformed_data)

  def __len__(self):
    return len(self.data_source)

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    if index in self.transformed_data:
      key_data = self.transformed_data[index]
      return key_data
    else:
      query, context, label, query_id = self.data_source[index]
      transformed_query = self.query_transform(query)  # [token_ids],[seg_ids],[masks]
      transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
      key_data = transformed_query, transformed_context, label, query_id
      self.transformed_data[index] = key_data

      return key_data

  def batchify_join_str(self, batch):
    query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
    contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch = [], [], [], [], [], []
    qids_batch = []
    labels_batch = []
    for sample in batch:
      (query_token_ids_list, query_input_masks_list, query_segment_ids_list), \
      (contexts_token_ids_list, contexts_input_masks_list, contexts_segment_ids_list) = sample[:2]
      query_token_ids_list_batch.append(query_token_ids_list)
      query_segment_ids_list_batch.append(query_segment_ids_list)
      query_input_masks_list_batch.append(query_input_masks_list)

      contexts_token_ids_list_batch.append(contexts_token_ids_list)
      contexts_segment_ids_list_batch.append(contexts_segment_ids_list)
      contexts_input_masks_list_batch.append(contexts_input_masks_list)

      labels_batch.append(sample[-2])
      qids_batch.append(sample[-1])
    long_tensors = [query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,
                    contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch]

    query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,\
    contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch \
      = (torch.tensor(t, dtype=torch.long) for t in long_tensors)

    labels_batch = torch.tensor(labels_batch, dtype=torch.float32)
    return query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,\
           contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch, labels_batch, qids_batch

class CombinedRankingDataset(Dataset):
  def __init__(self, file_path, transform, delimiter='\t',overwrite_cache:bool=False):
    self.transform = transform

    self.data_source = []
    self.transformed_data = {}
    cache_path = '.'.join(file_path.split(".")[:-1]) + '_combined.cache'
    if os.path.exists(cache_path) and not overwrite_cache:
      self.transformed_data = pickle_load(cache_path)
      self.data_source = [0] * len(self.transformed_data)
    else:
      f = open(file_path, 'r', encoding='utf-8')
      data = csv.DictReader(f, delimiter=delimiter)

      for row in data:
          self.data_source.append((row[constants.RANKING_INPUT_QUERY_NAME], row[constants.RANKING_INPUT_DOCUMENT_NAME],
                         int(row[constants.RANKING_INPUT_LABEL_NAME]), row[constants.RANKING_INPUT_QUERY_ID]))

      for idx in tqdm(range(len(self.data_source))):
          self.__get_single_item__(idx)

      pickle_dump(self.transformed_data, cache_path)
      self.data_source = [0] * len(self.transformed_data)

  def __len__(self):
    return len(self.data_source)

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    if index in self.transformed_data:
      key_data = self.transformed_data[index]
      return key_data
    else:
      query, context, label, query_id = self.data_source[index]
      transformed_data = self.transform(query, context)  # [token_ids],[seg_ids],[masks]
      key_data = transformed_data, label, query_id
      self.transformed_data[index] = key_data

      return key_data

  def batchify_join_str(self, batch):
    token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch = [], [], []
    labels_batch = []
    qids_batch = []
    for sample in batch:
      token_ids_list, input_masks_list, segment_ids_list = sample[0]
      token_ids_list_batch.append(token_ids_list)
      segment_ids_list_batch.append(segment_ids_list)
      input_masks_list_batch.append(input_masks_list)
      labels_batch.append(sample[-2])
      qids_batch.append(sample[-1])

    long_tensors = [token_ids_list_batch, input_masks_list_batch, segment_ids_list_batch]

    token_ids_list_batch, input_masks_list_batch, segment_ids_list_batch \
      = (torch.tensor(t, dtype=torch.long) for t in long_tensors)

    labels_batch = torch.tensor(labels_batch, dtype=torch.float32)
    return token_ids_list_batch, input_masks_list_batch, segment_ids_list_batch, \
           labels_batch, qids_batch

class BiencoderRankingDataset(Dataset):
  def __init__(self, file_json, query_transform, context_transform, overwrite_cache:bool=False):
    with open(file_json,'r') as f:
      data = json.load(f)
      self.data_list = list(data.values())
      self.query_transform = query_transform
      self.context_transform = context_transform
  def __len__(self):
    return len(self.data_list)
  def __getitem__(self,index):
    datapoint = self.data_list[index]
    query  = self.data_list[index][constants.RANKING_INPUT_QUERY_NAME]
    pos_context = self.data_list[index][constants.RANKING_INPUT_DOCUMENT_NAME][0]
    hard_negs = self.data_list[index][constants.RANKING_INPUT_HARD_NEGATIVES]
    contexts = [pos_context] + hard_negs
    transformed_query = self.query_transform([query])  # [token_ids],[seg_ids],[masks]
    transformed_context = self.context_transform(contexts)  # [token_ids],[seg_ids],[masks]
    return transformed_query, transformed_context

  def batchify_join_str(self,  batch):
    query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
    contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch = [],[],[],[],[],[]
    #qids_batch = []
    labels_batch = []
    hard_neg_ctx_indices = []
    for sample in batch:
      (query_token_ids_list, query_input_masks_list, query_segment_ids_list), \
      (contexts_token_ids_list, contexts_input_masks_list, contexts_segment_ids_list) = sample
      query_token_ids_list_batch += query_token_ids_list
      query_segment_ids_list_batch += query_segment_ids_list
      query_input_masks_list_batch += query_input_masks_list

      hard_negatives_start_idx = 1
      hard_negatives_end_idx = 1 + len(contexts_token_ids_list)
      current_ctxs_len = len(contexts_token_ids_list_batch)

      labels_batch.append(current_ctxs_len)
      hard_neg_ctx_indices.append(
        [
          i
          for i in range(
          current_ctxs_len + hard_negatives_start_idx,
          current_ctxs_len + hard_negatives_end_idx,
        )
        ]
      )

      contexts_token_ids_list_batch += contexts_token_ids_list
      contexts_segment_ids_list_batch += contexts_segment_ids_list
      contexts_input_masks_list_batch += contexts_input_masks_list


      #qids_batch.append(sample[-1])
    long_tensors = [query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,
                    contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch]

    query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,\
    contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch \
      = (torch.tensor(t, dtype=torch.long) for t in long_tensors)

    labels_batch = torch.tensor(labels_batch, dtype=torch.long)
    return query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,\
           contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch, labels_batch, hard_neg_ctx_indices