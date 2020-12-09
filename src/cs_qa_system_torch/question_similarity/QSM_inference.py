import argparse
import os
import time
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.metrics import *
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser(description='This module is used to do inference with the Question Similarity model')

parser.add_argument('--saved_model_path', type=str,
                    help='saved model path')

parser.add_argument('--test_data_path', type=str,
                    help='test data path')


def getSimilarityScore(sentenceA,sentenceB, model):
    embeddingA = model.encode([sentenceA])[0]
    embeddingB = model.encode([sentenceB])[0]
    score = np.dot(embeddingA,embeddingB)/(norm(embeddingA)*norm(embeddingB))
    return score


def replay(input_file, model):
    with open(input_file,'r',errors='ignore') as f_in:
        y=[]
        ypred=[]
        uttr_list=[]
        for idx, ln in enumerate(f_in):
            if idx !=0:
                q1,q2,label = ln.split('\t')
                score=getSimilarityScore(q1,q2,model)
                y.append(int(label[0]))
                ypred.append(score)
                uttr_list.append([q1,q2])
    return y, ypred, uttr_list, idx-1


def metrics(thred, y, ypred):
    y=np.asarray(y)
    ypred=np.asarray(ypred)
    yhat= (ypred>thred).astype(int)
    recall = recall_score(y, yhat)
    precision = precision_score(y, yhat)
    pearson_cor = pearsonr(y,ypred)[0]
    fpr, tpr, threshold = roc_curve(y, ypred)
    roc_auc = auc(fpr, tpr)
    return roc_auc, precision, recall, pearson_cor


def QSM_inference(args):

    model = SentenceTransformer(args.saved_model_path)

    start = time.time()
    y, ypred, uttr_list, n_sample = replay(args.test_data_path,model)
    end = time.time()
    print('{0:d} question pairs were scored in {1:.2f} seconds (in average {2:.2f} ms per query)'.format(n_sample, end - start, 1000 * (end - start) / n_sample))

    # Save the information retrieval result for future use
    df = pd.DataFrame()
    df['label'] = y
    df['pred'] = ypred
    df['q1'] = [q[0] for q in uttr_list]
    df['q2'] = [q[1] for q in uttr_list]
    test_infer_res_path = os.path.join(args.saved_model_path, 'test_infer_res.tsv')
    df.to_csv(test_infer_res_path, sep='\t')

    # Calculate the information retrieval metrics
    roc_auc, precision, recall, pearson_cor = metrics(0.94, y, ypred)
    print('{} question pairs were scored by {}: \n'
          '    AUC: {:.4f}\n'
          '    Precision: {:.4f}\n'
          '    Recall: {:.4f}\n'
          '    pearson_correlation: {:.4f}'.format(n_sample, args.saved_model_path, roc_auc, precision, recall, pearson_cor))


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    QSM_inference(arg_parser)


