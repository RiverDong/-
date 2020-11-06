from json import JSONDecodeError
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from ranking_metrics import get_prediction_df, eval_ir_result
import utils
import constants
import argparse
import os
import time
import json
import pandas as pd

from ranking_utils import load_collection

parser = argparse.ArgumentParser(description='This module is used to do inference with the information retrieval model')

parser.add_argument('--passage_collection_path', type=str,
                    help='path of the passage collection')
parser.add_argument('--qrels_path', type=str,
                    help='path of the qrels file')
parser.add_argument('--result_dir', type=str,
                    help='directory to save the information retrieval result')
parser.add_argument('--ir_model_name', type=str, default='BM25Okapi',
                    help='name of the information retrieval model to use')
parser.add_argument('--word_tokenizer_name', type=str, default='simple_word_tokenizer',
                    help='name of the word tokenizer for the information retrieval model to use')
parser.add_argument('--n', type=int, default=400,
                    help='number of relevant passages we want the IR model to return')
parser.add_argument('--add_preprocessed', type=int, default=0,
                    help='if it is not 0, add preprocessed queries and passages to prediction.tsv; otherwise, do not add')
parser.add_argument('--all_k', type=str, default='1,2,3,4,5,6,7,8,9,10',
                    help='List of k values')
parser.add_argument('--all_rank_thres_interested', type=str, default='1,2,3,4,5,6,7,8,9,10',
                    help='List of rank thresholds we are interested in')
parser.add_argument('--k1', type=float, default=None)
parser.add_argument('--b', type=float, default=None)
parser.add_argument('--load_existing_model', type=int, default=0)




def top_n_passage_ids(query, ir_model, n):
    # For a given query, get pids of the sorted top n relevant passages and their scores using the IR model
    try:
        return ir_model.get_top_n_single(query, n)
    except (AttributeError, TypeError) as error:
        print('Error: please make sure the IR model has a "get_top_n_single" method with desired signature')
        raise error

def ir_inference(args):

    if not args.load_existing_model:
        os.makedirs(args.result_dir, exist_ok=False)
    else:
        assert os.path.exists(args.result_dir)

    # Create the IR model
    passage_collection, pid_passage_tuple = load_collection(args.passage_collection_path)
    word_tokenizer = WordTokenizerFactory.create_word_tokenizer(args.word_tokenizer_name)
    ir_model = IRModelFactory.create_ir_model(ir_model_name=args.ir_model_name, ir_model_path=args.result_dir,
                                              corpus=pid_passage_tuple, tokenizer=word_tokenizer, k1=args.k1, b=args.b)

    # Retrieve top n relevant passages for each query in qrels
    assert args.qrels_path[-5:] == '.json'
    qrels = pd.read_json(args.qrels_path, orient='columns', typ='frame')

    pred_dict = dict()
    tic = time.time()
    for i in qrels.index:
        qid = qrels.loc[i, constants.RANKING_INPUT_QUERY_ID]
        query = qrels.loc[i, constants.RANKING_INPUT_QUERY_NAME]
        if qid not in pred_dict:
            pred_dict[qid] = top_n_passage_ids(query, ir_model, args.n)
    toc = time.time()
    print('Top-{0:d} relevant passages of {1:d} queries retrieved in {2:.2f} seconds (in average {3:.2f} ms '
          'per query)'.format(args.n, len(pred_dict), toc - tic, 1000 * (toc - tic) / len(pred_dict)))

    # Get and plot the metrics
    all_rank_thres_interested = sorted(list(set(int(i) for i in args.all_rank_thres_interested.split(',')) | {args.n}), reverse=True)
    assert min(all_rank_thres_interested) >= 1
    all_k = sorted(list(set(int(i) for i in args.all_k.split(',')) | {args.n}))
    assert min(all_k) > 0 and max(all_k) <= args.n
    pred_dict_pid_only = {k: v[0] for k, v in pred_dict.items()}
    metrics = eval_ir_result(qrels, pred_dict_pid_only, args.result_dir, all_k, all_rank_thres_interested)

    print('Retrieved top-{0:d} passages (from a pool of {1:d} passages) for {2:d} queries by {3:} (k1={4:.5f}, b={5:.5f}, delta={6:.5f}):\n'.format(args.n,
          len(pid_passage_tuple), len(pred_dict), args.ir_model_name, ir_model.k1, ir_model.b, ir_model.delta) + '\n'.join(['{0:}@{1:d}: {2:.4f}'.format(
           metric_name, all_k[-1], metric_value[-1]) for metric_name, metric_value in metrics.items()]))

    # Save all results
    if not args.load_existing_model:
        ir_model.save()
        utils.save_list_dict(pred_dict, 'top{0:d}_pred_dict_of_{1:}_on_{2:}'.format(args.n, args.ir_model_name, os.path.basename(args.qrels_path)[:-5]), args.result_dir)
        utils.save_list_dict(metrics, 'metrics_of_{0:}_on_{1:}'.format(args.ir_model_name, os.path.basename(args.qrels_path)[:-5]), args.result_dir)
        get_prediction_df(qrels, pred_dict, passage_collection, word_tokenizer, args.result_dir, args.ir_model_name, args.add_preprocessed)


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    ir_inference(arg_parser)


