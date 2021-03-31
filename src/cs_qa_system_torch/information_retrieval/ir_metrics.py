import numpy as np
import pandas as pd
import constants
import os
import time
import psutil
from multiprocessing import Pool
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from utils import plot_distr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_mrr_metric(prediction, actual, query_ids, n=None):
    df = pd.DataFrame()
    df['prediction'] = prediction
    df['actual'] = actual
    df['query_ids'] = query_ids
    score = df.groupby('query_ids').apply(
        lambda x: _mrr(x['prediction'].tolist(), x['actual'].tolist(), n))
    return np.mean(score.values)


def _mrr(prediction, actual, n=None):
    """
    Computes the Reciprocal rank of a given query.

    :param prediction: list of prediction scores of documents for a given query
    :param actual: list of actual scores of documents for a given query
    :param n: consider only topN documents, i.e, for MRR@n
    :return: Reciprocal rank of a given query
    """
    out = 0
    length = len(prediction) if n is None else n
    sort_idx = np.argsort(prediction)[::-1][:length]
    for i, idx in enumerate(sort_idx):
        if actual[idx] > 0:
            out = 1.0 / (i + 1)
            break
    return out

def get_mrr(df, n):
    return df.groupby(constants.RANKING_INPUT_QUERY_ID).apply(
        lambda x: _mrr(x[constants.WEIGHTED_RANKING_SCORE].tolist(), x[constants.RANKING_INPUT_LABEL_NAME].tolist(), n)).to_frame()

def compute_acc_pr_rec_fs_metric(predictions, actuals, threshold):
    """
    :param actuals:
    :param predictions:
    :param threshold: cut-off score for positive
    :return: accuracy, precision, recall, f_score
    """

    np_preds = np.array(predictions)
    np_actuals = np.array(actuals)
    idx = np_preds >= threshold
    np_preds[idx] = 1
    np_preds[~idx] = 0
    precision, recall, f_score, _ = precision_recall_fscore_support(np_actuals, np_preds, pos_label=1,
                                                                    average='binary')
    accuracy = accuracy_score(np_actuals, np_preds)
    acc_pr_rc_fs = {'accuracy':accuracy,'precision':precision, 'recall':recall,'f-score':f_score}
    return acc_pr_rc_fs


def ap_rr_coverage(true_list, pred_list, k=None):
    # Given a true list (order doesn't matter), a predicted list (order does matter) and k, get AP@k, RR@k, Coverage@k
    assert (len(set(pred_list)) == len(pred_list)) and (len(set(true_list)) == len(true_list))

    if k is not None:
        pred_list = pred_list[:k]

    n_tp = 0
    ap_sum = 0
    rr = 0
    for i, j in enumerate(pred_list, start=1):
        if j in true_list:
            n_tp += 1
            ap_sum += n_tp / i
            if rr == 0:
                rr = 1 / i
    ap = ap_sum / max(1, len(true_list))
    coverage = len(set(pred_list) & set(true_list)) / max(1, len(true_list))

    return ap, rr, coverage


def append_passage(x, pred_dict, passage_collection, forcibly_keep_pos=False):
    assert (x[constants.RANKING_INPUT_QUERY_ID].nunique() == 1) and (x[constants.RANKING_INPUT_QUERY_NAME].nunique() == 1)

    # "in" operation of a series checks whether the value is in the index of the series,
    # so we make all_pos_pid a list
    # For example, "2 in pd.Series([2, 3])" is False
    all_pos_pid = sorted(list(set(x[constants.RANKING_INPUT_DOCUMENT_ID])))

    all_pid, all_score = pred_dict[x[constants.RANKING_INPUT_QUERY_ID].iloc[0]]
    if forcibly_keep_pos:
        pos_pid_to_add = [i for i in all_pos_pid if i not in all_pid]
        all_pid = pos_pid_to_add + all_pid
        all_score = [999] * len(pos_pid_to_add) + all_score
    all_passage = [passage_collection[i] for i in all_pid]
    all_label = [int(i in all_pos_pid) for i in all_pid]

    s = x.iloc[0, :]
    df = pd.DataFrame(columns=s.index)
    df = df.append([s] * len(all_pid), ignore_index=True)
    df[constants.RANKING_INPUT_DOCUMENT_ID] = all_pid
    df[constants.RANKING_INPUT_DOCUMENT_NAME] = all_passage
    df[constants.RANKING_INPUT_LABEL_NAME] = all_label
    df['bm25_predictions'] = all_score

    return df


def get_prediction_df(qrels, pred_dict, passage_collection, word_tokenizer, result_path, result_name, add_preprocessed):
    df = qrels.groupby(constants.RANKING_INPUT_QUERY_ID).apply(lambda x: append_passage(x, pred_dict, passage_collection)).reset_index(drop=True)

    if add_preprocessed != 0:
        n_cpu = psutil.cpu_count(logical=True)

        tic = time.time()
        with Pool(n_cpu) as pool:
            temp_query_preprocessed = pool.map(word_tokenizer, df[constants.RANKING_INPUT_QUERY_NAME].to_list())
        df[constants.RANKING_INPUT_QUERY_NAME + '_bm25_preprocessed'] = [' '.join(i) for i in temp_query_preprocessed]
        toc = time.time()
        print('{0:d} queries preprocessed using {1:d} processes in {2:.2f} second(s)'.format(df.shape[0], n_cpu, toc - tic))

        tic = time.time()
        with Pool(n_cpu) as pool:
            temp_passage_preprocessed = pool.map(word_tokenizer, df[constants.RANKING_INPUT_DOCUMENT_NAME].to_list())
        df[constants.RANKING_INPUT_DOCUMENT_NAME + '_bm25_preprocessed'] = [' '.join(i) for i in temp_passage_preprocessed]
        toc = time.time()
        print('{0:d} passages preprocessed using {1:d} processes in {2:.2f} second(s)'.format(df.shape[0], n_cpu, toc - tic))

    df.to_csv(os.path.join(result_path, result_name + '_prediction.tsv'), sep='\t', index=False)


def all_ap_rr_coverage(qrels, pred_dict, k, qid_col=constants.RANKING_INPUT_QUERY_ID, pid_col=constants.RANKING_INPUT_DOCUMENT_ID):
    """
    Given qrels, pred_dict and k, calculate MAP@k, MRR@k, MCoverage@k
    :param qrels: pandas dataframe with at least two columns 'qid' and 'pid', where 'pid' is the passage id of the
                  true relevant passage of the query whose query id is 'qid'
    :param pred_dict: dictionary {qid1: [pid11, pid12, ...], qid2: [pid21, pid22, ...], ...}, where [pid11, pid12, ...]
                      are passage ids of the predicted relevant passages of the query whose query id is 'qid1'
    :param k: AP, RR, Coverage are calculated up to the kth entry of the predicted list
    :param qid_col: column of qid
    :param pid_col: column of pid
    :return: all AP, RR, Coverage in three lists
    """
    qrels_grouped = qrels.groupby(qid_col).apply(lambda x: list(x[pid_col].unique())).to_frame(name=pid_col).reset_index(drop=False)
    true_dict = dict(zip(qrels_grouped[qid_col], qrels_grouped[pid_col]))

    assert set(pred_dict.keys()) == set(true_dict.keys())

    all_ap = list()
    all_rr = list()
    all_coverage = list()
    for qid in pred_dict:
        ap, rr, coverage = ap_rr_coverage(true_dict[qid], pred_dict[qid], k)
        all_ap.append(ap)
        all_rr.append(rr)
        all_coverage.append(coverage)

    return np.array(all_ap), np.array(all_rr), np.array(all_coverage)


def eval_ir_result(qrels, pred_dict_pid_only, plot_dir, all_k, all_rank_thres_interested,
                   qid_col=constants.RANKING_INPUT_QUERY_ID, pid_col=constants.RANKING_INPUT_DOCUMENT_ID):
    metrics = {'map': [], 'mrr': [], 'mcoverage': []}
    for k in all_k:
        all_ap, all_rr, all_coverage = all_ap_rr_coverage(qrels, pred_dict_pid_only, k, qid_col, pid_col)

        metrics['map'].append(np.mean(all_ap))
        metrics['mrr'].append(np.mean(all_rr))
        metrics['mcoverage'].append(np.mean(all_coverage))
        for rank_thres in all_rank_thres_interested:
            metric_name = 'proportion_rank_<={:d}'.format(rank_thres)
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(np.mean(all_rr >= 1 / rank_thres))

        if plot_dir is not None:
            plot_distr(pd.Series(all_rr), os.path.join(plot_dir, 'Distribution_of_RR@{:d}.pdf'.format(k)),
                      'Distribution of RR@{:d}'.format(k), (0.1, 0.2, 0.33, 0.5, 1))

    if plot_dir is not None:
        with PdfPages(os.path.join(plot_dir, 'Metrics_for_all_k.pdf')) as pdf:
            fig, ax = plt.subplots(2, 2, figsize=(24, 12))

            for i, title in enumerate(['map', 'mrr', 'mcoverage', 'proportions']):
                if title != 'proportions':
                    ax.flat[i].plot(all_k, metrics[title], 'b-o', linewidth=1, markersize=2)
                else:
                    all_proportion_name = [metric_name for metric_name in metrics if metric_name not in ['map', 'mrr', 'mcoverage']]
                    for proportion_name in all_proportion_name:
                        k_thres_idx = np.where(np.array(all_k) >= max(all_rank_thres_interested))[0][0]
                        ax.flat[i].plot(all_k[:(k_thres_idx + 1)], metrics[proportion_name][:(k_thres_idx + 1)], '-o',
                                        label=proportion_name, linewidth=1, markersize=2)
                        ax.flat[i].legend(loc='upper left', frameon=True, ncol=3, mode=None, fontsize=8)
                        ax.flat[i].set_xticks(all_k[:(k_thres_idx + 1)])

                ax.flat[i].xaxis.grid(True, which="major", linestyle="-", color="w", lw=0.5)
                ax.flat[i].yaxis.grid(True, which="major", linestyle="-", color="w", lw=0.5)
                ax.flat[i].set_facecolor("#F5F5F5")
                ax.flat[i].set_title(title, fontsize=10)
                ax.flat[i].set_xlabel('k')
                for tick in ax.flat[i].xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)

            pdf.savefig(bbox_inches='tight')
            plt.close()

    return metrics



