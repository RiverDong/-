# Licensed to the Amazon.com under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.
# Owners: CSML-QA team

import msmarco_config
import pandas as pd
import os
import numpy as np


class MSMARCOPassageData:

    @staticmethod
    def generate_train_input_for_ranking(save_dir: str, train_split_ratio: float = 0.8, num_pos_samples: int = 1,
                                         num_neg_samples: int = 3,
                                         queries_count: int = 2000):
        """
        We have maximum 7 positive samples for each query and 999 negative samples.
        In total we have 260 million data points containing 502940 queries.
        :param save_dir
        :param outfile:
        :param num_pos_samples: maximum number of positives for
        :param num_neg_samples:
        :param queries_count:
        :return:
        """
        passage_data = pd.read_csv(os.path.join(msmarco_config.ROOT_PATH, msmarco_config.PASSAGE_FILE),
                                   sep='\t', index_col=False, header=None, names=['pid', 'passage'])
        queries_train = pd.read_csv(os.path.join(msmarco_config.ROOT_PATH, msmarco_config.QUERIES_TRAIN),
                                    sep='\t', index_col=False, header=None, names=['qid', 'query'])
        qrels_train = pd.read_csv(os.path.join(msmarco_config.ROOT_PATH, msmarco_config.QRELS_TRAIN), sep='\t',
                                  index_col=False, header=None, names=['qid', '_', 'rel_pid', 'label'])
        rel_norel_train = pd.read_csv(os.path.join(msmarco_config.ROOT_PATH, msmarco_config.QID_PID_REL_NONREL_PAIRS),
                                      sep='\t',
                                      index_col=False, header=None, names=['qid', 'rel_pid', 'norel_pid'])

        rel_train = rel_norel_train[['qid', 'rel_pid']]
        qrels_train = qrels_train[['qid', 'rel_pid']]
        rel_train = pd.concat([rel_train, qrels_train]).drop_duplicates().reset_index(drop=True)
        norel_train = rel_norel_train[['qid', 'norel_pid']].drop_duplicates().reset_index(drop=True)

        tmp = rel_train[['qid']].drop_duplicates().reset_index(drop=True)
        norel_train = pd.merge(norel_train, tmp, on=['qid'], how='inner')

        ## If exactly equal to 2 or more negative samples
        norel_train = norel_train.groupby(['qid']).apply(
            lambda x: x.sample(num_neg_samples) if len(x) >= num_neg_samples else None).reset_index(drop=True)
        ## if lessthan equal to 1 or more negative samples
        # norel_train = norel_train.groupby(['qid']).apply(
        #     lambda x: x.sample(num_neg_samples) if len(x) > num_neg_samples else x).reset_index(drop=True)

        tmp = norel_train[['qid']].drop_duplicates().sample(queries_count).reset_index(drop=True)
        rel_train = pd.merge(rel_train, tmp, on=['qid'], how='inner').reset_index(drop=True)
        ## if lessthan equal to 1 or more positive samples
        rel_train = rel_train.groupby(['qid']).apply(
            lambda x: x.sample(num_pos_samples) if len(x) > num_pos_samples else x).reset_index(drop=True)

        rel_train = rel_train.sample(frac=1).reset_index(drop=True)
        rel_train.reset_index(level=0, inplace=True)
        norel_train = pd.merge(norel_train, rel_train[['index', 'qid']], on=['qid'], how='inner')

        rel_train['label'] = 1
        norel_train['label'] = 0

        rel_train = rel_train.rename(columns={'rel_pid': 'pid'})
        norel_train = norel_train.rename(columns={'norel_pid': 'pid'})
        cols = ['index', 'qid', 'pid', 'label']
        rel_train = rel_train[cols]
        norel_train = norel_train[cols]
        data_train = pd.concat([rel_train, norel_train])
        data_train = data_train.merge(queries_train, how='inner', on=['qid'])
        data_train = data_train.merge(passage_data, how='inner', on=['pid'])
        data_train = data_train.sort_values(by=['index', 'label'], ascending=False)
        # data_train = data_train.sample(frac=1).reset_index(drop=True)
        # cols = ['label', 'query', 'passage']
        # data_train[cols].to_csv(outfile_path, sep='\t', index=False, header=False)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_train = data_train.sample(frac=1).reset_index(drop=True)
        cols = ['qid', 'query', 'pid', 'passage', 'label']
        data_train = data_train[cols]
        msk = np.random.rand(len(data_train)) < train_split_ratio
        data_train[msk].to_csv(os.path.join(save_dir, msmarco_config.OUTPUT_TRAIN_FILE), sep='\t', index=False)
        data_train[~msk].to_csv(os.path.join(save_dir, msmarco_config.OUTPUT_TEST_FILE), sep='\t', index=False)

    @staticmethod
    def generate_dev_input_for_ranking(save_dir: str, queries_count: int = None):
        """
        We have 6980 unique queries in Top1000 dev. There could be queries here
        which may not have any positive document.
        :param save_dir
        :param outfile:
        :param queries_count: how many queries to use in dev set
        """
        qrels_dev = pd.read_csv(os.path.join(msmarco_config.ROOT_PATH, msmarco_config.QRELS_DEV), sep='\t',
                                index_col=False, header=None,
                                names=['qid', '_', 'pid', 'label'])
        top1000_dev = pd.read_csv(os.path.join(msmarco_config.ROOT_PATH, msmarco_config.TOP1000_DEV),
                                  sep='\t', index_col=False, names=['qid', 'pid', 'query', 'passage'])

        qrels_dev = qrels_dev[['qid', 'pid']].drop_duplicates().reset_index(drop=True)

        ## Change it to left join if you want to comply with MSMARCO dataset leaderboard devset
        ## where we also take queries that does not have even a single correct relevant passage
        data_dev = pd.merge(left=top1000_dev, right=qrels_dev, on=['qid', 'pid'],
                            how='right', validate='one_to_one', indicator=True)

        data_dev['_merge'] = (data_dev['_merge'] == 'both').astype(int)
        data_dev = data_dev.rename(columns={'_merge': 'label'})

        if queries_count:
            filtered_qid = data_dev.qid.drop_duplicates().sample(queries_count).reset_index(drop=True)
            data_dev = pd.merge(data_dev, filtered_qid, on=['qid'], how='inner')
            data_dev = data_dev.drop_duplicates().reset_index(drop=True)

        data_dev = data_dev.sort_values(by='qid')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        outfile_path = os.path.join(save_dir, msmarco_config.OUTPUT_VALID_FILE)
        cols = ['qid', 'query', 'pid', 'passage', 'label']
        data_dev[cols].to_csv(outfile_path, sep='\t', index=False)

    @staticmethod
    def create_inference_document_dict():
        passage_data = pd.read_csv(os.path.join(msmarco_config.ROOT_PATH, msmarco_config.PASSAGE_FILE),
                                   sep='\t', index_col=False, header=None, names=['pid', 'passage'])
        return passage_data.set_index('pid')['passage'].to_dict()
