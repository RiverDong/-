######## This file is still under construction #############################

import os
import argparse
import logging
import csv
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import constants
from ranking_dataset import CombinedRankingDataset, RankingDataset
from ranking_model import BertBiEncoderModel, BertCrossEncoderModel
from ranking_transform import CombinedRankingTransform, RankingTransform
from ranking_utils import MODEL_CLASSES, get_prediction, store_prediction_results
from utils import logging_config

logger = logging.getLogger(__name__)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to predict the relevant documents for a '
                                                 'given query in Question Answering system')

    parser.add_argument('--save_dir', type=str, default='inference', help='path to inference output')
    parser.add_argument('--params_dir', type=str, default='ranking_model_output.params',
                        help='model artifacts file')
    parser.add_argument("--model_type", default='bert', type=str, help='[bert, distilbert]')
    parser.add_argument("--model_type_name", default='bert-base-uncased', type=str)
    parser.add_argument("--architecture", required=True, type=str, help='[bi, cross]')
    parser.add_argument("--max_passage_length", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument('--gpu', type=str, help='gpus to run model on')
    parser.add_argument("--test_data_path", default='data', type=str)
    parser.add_argument("--test_batch_size", default=2, type=int, help="Total batch size for eval.")
    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')


    parser.add_argument('--document_file', type=str,
                        help='predictions file')
    parser.add_argument('--mrr_n', type=int, default=2,
                        help='value of N for Mean reciprocal Rank')
    parser.add_argument('--pos_threshold_score', type=float, default=0.5,
                        help='value of threshold for positive')



    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging_config(args.save_dir, name='inference')

    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]
    tokenizer = TokenizerClass.from_pretrained(args.params_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    transform = None
    query_transform = None
    context_transform = None
    if args.architecture == 'cross':
        transform = CombinedRankingTransform(tokenizer=tokenizer,
                                             max_len=(args.max_query_length + args.max_passage_length))
    else:
        query_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_query_length)
        context_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_passage_length)

    if args.architecture == 'cross':
        dataset = CombinedRankingDataset(args.test_data_path, transform)
    else:
        dataset = RankingDataset(args.test_data_path,
                                       query_transform, context_transform)
    dataloader = DataLoader(dataset,
                                batch_size=args.test_batch_size, shuffle=False,
                                collate_fn=dataset.batchify_join_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_config = ConfigClass.from_pretrained(args.model_type_name)
    bert = BertModelClass(bert_config)

    if args.architecture == 'poly':
        model = BertBiEncoderModel(bert_config, bert=bert)
    elif args.architecture == 'bi':
        model = BertBiEncoderModel(bert_config, bert=bert)
    elif args.architecture == 'cross':
        model = BertCrossEncoderModel(bert_config, bert=bert)
    else:
        raise Exception('Unknown architecture.')

    model.load_state_dict(torch.load(os.path.join(args.params_dir, 'pytorch_model.pt')))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    predictions, actuals, qids = get_prediction(dataloader, model, device)
    store_prediction_results(args.test_data_path, predictions, actuals, qids,
                             os.path.join(args.save_dir, args.prediction_file))



