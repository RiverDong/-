import json
import logging
import os
from collections import defaultdict
import pandas as pd

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import constants

logger = logging.getLogger(__name__)

class TextSequenceDataset(Dataset):
    def __init__(self, text_list, transform):
        self.text_list = text_list
        self.transform = transform

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        _, text = self.text_list[index]
        return text

    def batchify(self, batch):
        text_sequences = []
        for sample in batch:
            text_sequences += [sample]
        return self.transform(text_sequences)

class MultipleNegativeDataset(Dataset):
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
        return query, contexts

    def batchify(self, batch):
        query_batch = []
        contexts_batch = []
        labels_batch = []
        hard_neg_ctx_indices = []
        for sample in batch:
            query, contexts =  sample
            query_batch += [query]
            hard_negatives_start_idx = 0
            hard_negatives_end_idx = len(contexts)
            current_ctxs_len = len(contexts_batch)
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
            contexts_batch += contexts
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return (*self.query_transform(query_batch), *self.context_transform(contexts_batch), labels_batch, hard_neg_ctx_indices)

#given triples of (query,pos_passage,neg_passage) use this
class TripletDataset(Dataset):
    corpus = {}
    queries = {}
    data_source = []

    def __init__(self, qidpidtriples_filepath: str, query_transform,
                 context_transform, sep='\t', overwrite_cache: bool = False):
        self.query_transform = query_transform
        self.context_transform = context_transform

        logger.info('****** Preparing the training data *******')

        with tqdm(total=os.path.getsize(qidpidtriples_filepath)) as pbar:
            with open(qidpidtriples_filepath, encoding='utf8') as fIn:
                for line in fIn:
                    pbar.update(len(line))
                    list_row = line.strip().split(sep)
                    if len(list_row) == 3:
                        self.data_source.append(list_row)

    def __getitem__(self, index):
        query = self.data_source[index][0]
        pos_neg_context = [self.data_source[index][1], self.data_source[index][2]]
        return query, pos_neg_context

    def __len__(self):
        return len(self.data_source)

    def batchify(self, batch):
        query_batch = []
        contexts_batch = []
        labels_batch = []
        hard_neg_ctx_indices = []
        for sample in batch:
            query, contexts =  sample
            query_batch += [query]
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = len(contexts)
            current_ctxs_len = len(contexts_batch)
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
            contexts_batch += contexts
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return (*self.query_transform(query_batch), *self.context_transform(contexts_batch), labels_batch, hard_neg_ctx_indices)


## Given triples of (qid,posid,negid) use this
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
        query = self.queries[self.data_source[index][0]]
        pos_neg_context = [self.corpus[self.data_source[index][1]], self.corpus[self.data_source[index][2]]]
        return query, pos_neg_context

    def __len__(self):
        return len(self.data_source)

    def batchify(self, batch):
        query_batch = []
        contexts_batch = []
        labels_batch = []
        hard_neg_ctx_indices = []
        for sample in batch:
            query, contexts =  sample
            query_batch += [query]
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = len(contexts)
            current_ctxs_len = len(contexts_batch)
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
            contexts_batch += contexts
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return (*self.query_transform(query_batch), *self.context_transform(contexts_batch), labels_batch, hard_neg_ctx_indices)

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