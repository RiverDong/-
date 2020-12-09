import argparse
import os
import time
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.metrics import *
from sentence_transformers import SentenceTransformer
import json
from scipy.spatial.distance import cdist


parser = argparse.ArgumentParser(description='This module is used to do real time inference with the Question Similarity model')

parser.add_argument('--saved_model_path', type=str,
                    help='saved model path')

parser.add_argument('--test_data_path', type=str,
                    help='test data path')

parser.add_argument('--qa_repo_path', type=str,
                    help='qa repo path')


def get_repo(qa_repo_path, model):
    f_in = open(qa_repo_path, "r")
    data = json.load(f_in)
    qa_repo = []
    qa_id = []

    for i in range(len(data['qa_bank'])):
        qa_repo += data['qa_bank'][i]['questions']
        qa_id += [data['qa_bank'][i]['id']] * len(data['qa_bank'][i]['questions'])

    qa_embed = model.encode(qa_repo)
    return qa_embed, qa_repo, qa_id


def GetSimilarQ(query, model, question_embed, uttr_list, id_list, topn=1):
    query_embed = model.encode([query])
    cos_dist = np.squeeze(1 - cdist(question_embed, query_embed, 'cosine'), axis=1)
    idx_list = cos_dist.argsort()[-topn:][::-1]
    return uttr_list[idx_list[0]], cos_dist[idx_list[0]], id_list[idx_list[0]]


def replay(input_file, model, qa_embed, qa_repo, qa_id):
    with open(input_file,'r',errors='ignore') as f_in:
        out_list = []
        n_sample = 0 
        for idx, ln in enumerate(f_in):
            if idx !=0:
                n_sample += 1
                query, true_matched_q_idx = ln.split('\t')
                true_matched_q_idx = true_matched_q_idx.replace('\n','')
                matched_q, score, matched_q_idx = GetSimilarQ(query, model, qa_embed, qa_repo, qa_id)
                correct_match = 1 if true_matched_q_idx == matched_q_idx else 0
                out_list.append([query, true_matched_q_idx, matched_q, score, matched_q_idx, correct_match])
    df=pd.DataFrame(out_list)
    df.columns=['query','true_matched_q_idx','matched_q','score','matched_q_idx','correct_match']
    return df, n_sample



def QSM_inference(args):

    model = SentenceTransformer(args.saved_model_path)
    qa_embed, qa_repo, qa_id = get_repo(args.qa_repo_path, model)

    start = time.time()
    df, n_sample = replay(args.test_data_path, model, qa_embed, qa_repo, qa_id)
    end = time.time()
    print('{0:d} question pairs were scored in {1:.2f} seconds (in average {2:.2f} ms per query)'.format(n_sample, end - start, 1000 * (end - start) / n_sample))

    # Save the information retrieval result for future use
    test_infer_res_path = os.path.join(args.saved_model_path, 'test_infer_res.tsv')
    df.to_csv(test_infer_res_path, sep='\t')

    # Calculate the information retrieval metrics
    print('{} question pairs were scored by {}: \n'
          '    Number of correct match: {}\n'
          '    Percentage: {:.4f}'.format(n_sample, args.saved_model_path, df.correct_match.sum(), df.correct_match.sum()/len(df)))


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    QSM_inference(arg_parser)