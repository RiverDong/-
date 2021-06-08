import os
import csv
from collections import defaultdict
import logging

from tqdm import tqdm
import json
import pandas as pd

import torch
from torch.utils.data import Dataset

import constants
from utils import pickle_load, pickle_dump

logger = logging.getLogger(__name__)


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
    def __init__(self, file_path, query_transform, context_transform, delimiter='\t', overwrite_cache: bool = False):
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
                self.data_source.append(
                    (row[constants.RANKING_INPUT_QUERY_NAME], row[constants.RANKING_INPUT_DOCUMENT_NAME],
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

        query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
        contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch \
            = (torch.tensor(t, dtype=torch.long) for t in long_tensors)

        labels_batch = torch.tensor(labels_batch, dtype=torch.float32)
        return query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
               contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch, labels_batch, qids_batch


class CombinedRankingDataset(Dataset):
    def __init__(self, file_path, transform, delimiter='\t', overwrite_cache: bool = False):
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
                self.data_source.append(
                    (row[constants.RANKING_INPUT_QUERY_NAME], row[constants.RANKING_INPUT_DOCUMENT_NAME],
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
    def __init__(self, file_json, query_transform, context_transform, overwrite_cache: bool = False):
        with open(file_json, 'r') as f:
            data = json.load(f)
            self.data_list = list(data.values())
            self.query_transform = query_transform
            self.context_transform = context_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        datapoint = self.data_list[index]
        query = self.data_list[index][constants.RANKING_INPUT_QUERY_NAME]
        pos_context = self.data_list[index][constants.RANKING_INPUT_DOCUMENT_NAME][0]
        hard_negs = self.data_list[index][constants.RANKING_INPUT_HARD_NEGATIVES]
        contexts = [pos_context] + hard_negs
        transformed_query = self.query_transform([query])  # [token_ids],[seg_ids],[masks]
        transformed_context = self.context_transform(contexts)  # [token_ids],[seg_ids],[masks]
        return transformed_query, transformed_context

    def batchify_join_str(self, batch):
        query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
        contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch = [], [], [], [], [], []
        # qids_batch = []
        labels_batch = []
        hard_neg_ctx_indices = []
        for sample in batch:
            (query_token_ids_list, query_input_masks_list, query_segment_ids_list), \
            (contexts_token_ids_list, contexts_input_masks_list, contexts_segment_ids_list) = sample
            query_token_ids_list_batch += query_token_ids_list
            query_segment_ids_list_batch += query_segment_ids_list
            query_input_masks_list_batch += query_input_masks_list

            hard_negatives_start_idx = 0
            hard_negatives_end_idx = len(contexts_token_ids_list)
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

            # qids_batch.append(sample[-1])
        long_tensors = [query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,
                        contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch]

        query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
        contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch \
            = (torch.tensor(t, dtype=torch.long) for t in long_tensors)

        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
               contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch, labels_batch, hard_neg_ctx_indices


class SimpleDataset(Dataset):
    def __init__(self, text_list, transform):
        self.text_list = text_list
        self.transform = transform

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        id, text = self.text_list[index]
        return self.transform(text)


class MSMARCOTripletDataset(Dataset):
    corpus = {}
    queries = {}
    data_source = []

    def __init__(self, collection_filepath: str, queries_filepath: str, qidpidtriples_filepath: str, query_transform,
                 context_transform, sep='\t', overwrite_cache: bool = False):
        self.query_transform = query_transform
        self.context_transform = context_transform

        logger.info('****** Preparing the training data *******')
        with open(collection_filepath, encoding='utf8') as fIn:
            for line in fIn:
                pid, passage = line.strip().split(sep)
                self.corpus[pid] = passage

        with open(queries_filepath, encoding='utf8') as fIn:
            for line in fIn:
                qid, query = line.strip().split(sep)
                self.queries[qid] = query

        with tqdm(total=os.path.getsize(qidpidtriples_filepath)) as pbar:
            with open(qidpidtriples_filepath, encoding='utf8') as fIn:
                for line in fIn:
                    pbar.update(len(line))
                    list_row = line.strip().split(sep)
                    if len(list_row) == 3:
                        self.data_source.append(list_row)

    def __getitem__(self, index):
        transformed_query = self.query_transform([self.queries[self.data_source[index][0]]])
        transformed_context = self.context_transform(
            [self.corpus[self.data_source[index][1]], self.corpus[self.data_source[index][2]]])
        return transformed_query, transformed_context

    def __len__(self):
        return len(self.data_source)

    def batchify_join_str(self, batch):
        query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
        contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch = [], [], [], [], [], []
        labels_batch = []
        hard_neg_ctx_indices = []
        for sample in batch:
            (query_token_ids_list, query_input_masks_list, query_segment_ids_list), \
            (contexts_token_ids_list, contexts_input_masks_list, contexts_segment_ids_list) = sample
            query_token_ids_list_batch += query_token_ids_list
            query_segment_ids_list_batch += query_segment_ids_list
            query_input_masks_list_batch += query_input_masks_list

            hard_negatives_start_idx = 1
            hard_negatives_end_idx = len(contexts_token_ids_list)
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

            # qids_batch.append(sample[-1])
        long_tensors = [query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch,
                        contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch]

        query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
        contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch \
            = (torch.tensor(t, dtype=torch.long) for t in long_tensors)

        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return query_token_ids_list_batch, query_input_masks_list_batch, query_segment_ids_list_batch, \
               contexts_token_ids_list_batch, contexts_input_masks_list_batch, contexts_segment_ids_list_batch, labels_batch, hard_neg_ctx_indices

    @classmethod
    def msmarco_create_dev_data_from_triplets(cls, collection_filepath, queries_filepath, triplet_filepath,
                                              num_dev_queries,
                                              num_max_dev_negatives, sep='\t'):
        corpus = {}
        queries = {}

        logger.info('****** Creating the dev data *******\n Using dev file: {}'.format(triplet_filepath))
        with open(collection_filepath, encoding='utf8') as fIn:
            for line in fIn:
                pid, passage = line.strip().split(sep)
                corpus[pid] = passage

        with open(queries_filepath, encoding='utf8') as fIn:
            for line in fIn:
                qid, query = line.strip().split(sep)
                queries[qid] = query.strip()

        dev_corpus = {}
        dev_queries = set()
        num_negatives = defaultdict(int)
        qrels_columns = [constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME,
                         constants.RANKING_INPUT_DOCUMENT_ID]
        qrels_list = []
        with tqdm(total=os.path.getsize(triplet_filepath)) as pbar:
            with open(triplet_filepath, 'r') as fIn:
                for line in fIn:
                    pbar.update(len(line))
                    qid, pos_id, neg_id = line.strip().split(sep)
                    if len(dev_queries) < num_dev_queries or qid in dev_queries:
                        qrels_list.append(dict(zip(qrels_columns, [qid, queries[qid], pos_id])))
                        dev_queries.add(qid)

                        # Ensure the corpus has the positive
                        dev_corpus[pos_id] = ('', corpus[pos_id])

                        if num_negatives[qid] < num_max_dev_negatives:
                            dev_corpus[neg_id] = ('', corpus[neg_id])
                            num_negatives[qid] += 1

        qrels_dev_data = pd.DataFrame(qrels_list).to_json(
            os.path.join(os.path.dirname(triplet_filepath), 'dev-qrels.json'))
        json.dump(dev_corpus, open(os.path.join(os.path.dirname(triplet_filepath), 'dev-collection.json'), 'w'))

    @classmethod
    def msmarco_create_dev_data_from_qrels(cls, collection_filepath, queries_filepath, qrels_filepath, max_passages = -1, sep='\t'):
        dev_corpus = {}
        dev_queries = {}
        qrels_list = []
        dev_pos_ids = set()
        qrels_columns = [constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME,
                         constants.RANKING_INPUT_DOCUMENT_ID]
        logger.info('****** Creating the dev data *******\n Using qrels file: {}'.format(qrels_filepath))

        with open(queries_filepath, encoding='utf8') as fIn:
            for line in fIn:
                qid, query = line.strip().split(sep)
                dev_queries[qid] = query.strip()

        with open(qrels_filepath) as fIn:
            for line in fIn:
                qid, _, pos_id, _ = line.split(sep)

                if qid not in dev_queries:
                    continue

                qrels_list.append(dict(zip(qrels_columns, [qid, dev_queries[qid], pos_id])))
                dev_pos_ids.add(pos_id)

        with open(collection_filepath, encoding='utf8') as fIn:
            for line in fIn:
                pid, passage = line.strip().split(sep)

                if pid in dev_pos_ids or max_passages <= 0 or len(dev_corpus) <= max_passages:
                    dev_corpus[pid] = ('', passage)

        qrels_dev_data = pd.DataFrame(qrels_list).to_json(
            os.path.join(os.path.dirname(qrels_filepath), 'dev-qrels.json'))
        json.dump(dev_corpus, open(os.path.join(os.path.dirname(qrels_filepath), 'dev-collection.json'), 'w'))
