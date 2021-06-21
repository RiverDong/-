import argparse
import os
import logging
import pandas as pd

import torch

from information_retrieval.ir_utils import get_ranking_evaluation
from information_retrieval.ir_metrics import get_mrr
from utils import logging_config

logger = logging.getLogger(__name__)


def display_result(df, n_mrr=10, proportion_rank_index=None):
    mrr, proportion_ranks = get_mrr(df, n_mrr, proportion_rank_index)
    logger.info('\n' * 3 + '=' * 40 + 'Evaluation Metrics' + '=' * 40 + '\n')
    logger.info('mrr@{}: {}'.format(n_mrr, mrr))
    for i, rank in enumerate(proportion_rank_index):
        logger.info('proportion rank@{}: {}'.format(rank, proportion_ranks[i]))
    logger.info('\n' + '=' * 80 + '\n' * 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to predict the relevant documents for a '
                                                 'given query in Question Answering system')
    parser.add_argument('--gpu', type=str, default='', help='gpus to run model on')
    parser.add_argument('--output_dir', type=str, default='inference_ir', help='path to inference output')
    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')

    parser.add_argument('--passage_collection_path', type=str, help='path of the passage collection')
    parser.add_argument('--qrels_path', type=str,
                        help='path of the qrels file')
    parser.add_argument('--context_embeddings_path', type=str, default=None,
                        help='path to context embeddings')
    parser.add_argument('--overwrite_context_embeddings', action="store_true",
                        help='Overwrite context embeddings if set to true')

    ## Initial Ranking Arguments
    parser.add_argument('--rank_model_name_or_path', type=str, default='BM25Okapi',
                        help='name of the information retrieval model to use')
    parser.add_argument('--rank_model_architecture', type=str, default=None,
                        help='when using external architectures to load use this as either [bi,cross or single]')
    parser.add_argument("--rank_batch_size", default=128, type=int,
                        help="The batch size to use for each ranking inference")
    parser.add_argument('--rank_top_n', type=int, default=50,
                        help='number of relevant passages we want the IR model to return')
    parser.add_argument('--rank_threshold_score', type=float, default=0.0,
                        help='The threshold score to be used to filter documents')
    parser.add_argument('--rerank_score_weight', type=float, default=1.0,
                        help='The weight factor between ranking score and reranking score [wt*reranking_score + (1-wt)*ranking_score]')

    ##reranking arguments
    parser.add_argument(
        "--do_rerank", action="store_true", help="Set this flag if you are reranking the model."
    )
    parser.add_argument('--rerank_model_name_or_path', type=str, default='BM25Okapi',
                        help='name of the information retrieval model to use')
    parser.add_argument('--rerank_model_architecture', type=str, default=None,
                        help='when using external architectures to load use this as either [bi,cross or single]')
    parser.add_argument("--rerank_batch_size", default=128, type=int,
                        help="The batch size to use for each ranking inference")
    parser.add_argument('--rerank_top_n', type=int, default=5,
                        help='number of relevant passages we want the IR model to return')
    parser.add_argument('--rerank_threshold_score', type=float, default=0.0,
                        help='The threshold score to be used to filter documents')

    parser.add_argument(
        "--only_print_results", action="store_true", help="Set this flag if you are reranking the model."
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != '' else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging_config(args.output_dir, name='inference')
    logger.info("The input args are: \n{}".format(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("The inference files will be saved to: {}".format(args.output_dir))

    rank_results_df = get_ranking_evaluation(args.qrels_path, args.passage_collection_path,
                                             args.rank_model_name_or_path, args.rank_batch_size, args.rank_top_n,
                                             args.rank_threshold_score, device,
                                             model_architecture=args.rank_model_architecture,
                                             context_embeddings_path=args.context_embeddings_path if args.context_embeddings_path is not None else args.output_dir,
                                             overwrite_context_embeddings = args.overwrite_context_embeddings)

    rank_results_df.to_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
    print('Rank results:\n')
    display_result(rank_results_df, 10, [5,3,2])

    if args.do_rerank:
        rerank_results_df = get_ranking_evaluation(rank_results_df, args.passage_collection_path,
                                                   args.rerank_model_name_or_path, args.rerank_batch_size,
                                                   args.rerank_top_n, args.rerank_threshold_score,
                                                   device, rerank=True, rerank_score_weight=args.rerank_score_weight,
                                                   model_architecture=args.rerank_model_architecture)
        rerank_results_df.to_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
        print('ReRank results:\n')
        display_result(rerank_results_df, 10, [5,3,2])

    if args.only_print_results:
        df = pd.read_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
        display_result(df, 10, [5,3,2])
