import argparse
import os
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import constants
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from ir_inference import load_collection, top_n_passage_ids
from ranking_dataset import CombinedInferenceDataset, CombinedRankingDataset, InferenceDataset
from ranking_metrics import _mrr
from ranking_model import BertBiEncoderModel, BertCrossEncoderModel
from ranking_transform import CombinedRankingTransform, RankingTransform
from ranking_utils import MODEL_CLASSES, get_prediction, store_prediction_results, predict
from utils import logging_config, plot_distr


class ScorePrediction   :
    passage_collection = None
    ir_model = None
    ml_model = None
    transform = None
    query_transform = None
    context_transform = None
    ir_n = 5
    passage_collection_path = '/data/QAData/MSAmazonFinalData_large/collection-test.json'
    word_tokenizer_name = 'simple_word_tokenizer_no_stopwords_stem'
    ir_model_name = 'BM25Okapi'
    params_dir = 'output_torch/cross/artifacts/'
    model_type = 'bert'
    model_type_name = 'bert-base-uncased'
    architecture = 'cross'
    max_query_length = 64
    max_passage_length = 256
    @classmethod
    def load_ir_model(cls):
        result_dir = ''
        cls.passage_collection, pid_passage_tuple = load_collection(cls.passage_collection_path)
        word_tokenizer = WordTokenizerFactory.create_word_tokenizer(cls.word_tokenizer_name)
        cls.ir_model = IRModelFactory.create_ir_model(ir_model_name=cls.ir_model_name, ir_model_path=result_dir,
                                              corpus=pid_passage_tuple, tokenizer=word_tokenizer)

        ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[cls.model_type]
        tokenizer = TokenizerClass.from_pretrained(cls.params_dir)
        if cls.architecture == 'cross':
            cls.transform = CombinedRankingTransform(tokenizer=tokenizer,
                                             max_len=(cls.max_query_length + cls.max_passage_length),
                                             bool_np_array=True)
        else:
            cls.query_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_query_length, bool_np_array=True)
            cls.context_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_passage_length, bool_np_array=True)

        bert_config = ConfigClass.from_pretrained(cls.model_type_name)
        bert = BertModelClass(bert_config)

        if cls.architecture == 'poly':
            cls.ml_model = BertBiEncoderModel(bert_config, bert=bert)
        elif cls.architecture == 'bi':
            cls.ml_model = BertBiEncoderModel(bert_config, bert=bert)
        elif cls.architecture == 'cross':
            cls.ml_model = BertCrossEncoderModel(bert_config, bert=bert)
        else:
            raise Exception('Unknown architecture.')

        cls.ml_model.load_state_dict(torch.load(os.path.join(cls.params_dir, 'pytorch_model.pt')))


    @classmethod
    def get_documents(cls,query):
        if cls.ml_model == None:
            print('yo!!!')
            cls.load_ir_model()
            print('done yo!!!')
        pred_list = [(1, doc, query, cls.passage_collection[doc], score) for doc,score in zip(*top_n_passage_ids(query, cls.ir_model, cls.ir_n))]
        if cls.architecture == 'cross':
            dataset = CombinedInferenceDataset(pred_list, cls.transform)
        else:
            dataset = InferenceDataset(pred_list, cls.query_transform, cls.context_transform)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        predictions = predict(dataloader, cls.ml_model, "cpu")
        df = pd.DataFrame(pred_list, columns=['qid', 'pid', 'query', 'passage', 'ir_score'])
        df['ml_score'] = predictions

        return df

def get_rr(df, n, qid_col='qid', score_col='model_ir_score', label_col='label'):
    return df.groupby(qid_col).apply(lambda x: _mrr(x[score_col].tolist(), x[label_col].tolist(), n))

def model_ir_score(x, model_score_col, ir_score_col, wt):
    ir_score_sum = x[ir_score_col].sum()
    if ir_score_sum == 0:
        x['model_ir_score'] = wt * x[model_score_col]
    else:
        x['model_ir_score'] = wt * x[model_score_col] + (1 - wt) * (x[ir_score_col] / ir_score_sum)
    return x

def rr_for_wt(prediction_df, wt, n=10, return_df=False):
    prediction_df = prediction_df.groupby('qid').apply(lambda x: model_ir_score(x, 'model_score', 'ir_score', wt)).reset_index(drop=True)
    if return_df:
        return get_rr(prediction_df, n, 'qid', 'model_ir_score', 'label').to_frame(name='wt_{:.2f}'.format(wt)), prediction_df
    else:
        return get_rr(prediction_df, n, 'qid', 'model_ir_score', 'label').to_frame(name='wt_{:.2f}'.format(wt))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This module is used to predict the relevant documents for a '
                                                 'given query in Question Answering system')
    parser.add_argument('--passage_collection_path', type=str,
                        help='path of the passage collection')
    parser.add_argument('--ir_model_name', type=str, default='BM25Okapi',
                        help='name of the information retrieval model to use')
    parser.add_argument('--word_tokenizer_name', type=str, default='simple_word_tokenizer',
                        help='name of the word tokenizer for the information retrieval model to use')
    parser.add_argument('--qrels_path', type=str,
                        help='path of the qrels file')
    parser.add_argument('--n', type=int, default=100,
                        help='number of relevant passages we want the IR model to return')
    parser.add_argument('--params_dir', type=str, default='ranking_model_output.params',
                        help='model artifacts file')
    parser.add_argument("--model_type", default='bert', type=str, help='[bert, distilbert]')
    parser.add_argument("--model_type_name", default='bert-base-uncased', type=str)
    parser.add_argument("--architecture", required=True, type=str, help='[bi, cross]')
    parser.add_argument("--max_passage_length", default=256, type=int)
    parser.add_argument("--max_query_length", default=32, type=int)
    parser.add_argument('--gpu', type=str, help='gpus to run model on')
    parser.add_argument('--save_dir', type=str, default='inference', help='path to inference output')
    parser.add_argument("--test_batch_size", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')


    args = parser.parse_args()
    result_dir = ''
    passage_collection, pid_passage_tuple = load_collection(args.passage_collection_path)
    word_tokenizer = WordTokenizerFactory.create_word_tokenizer(args.word_tokenizer_name)
    ir_model = IRModelFactory.create_ir_model(ir_model_name=args.ir_model_name, ir_model_path=result_dir, corpus=pid_passage_tuple, tokenizer=word_tokenizer)

    # Retrieve top n relevant passages for each query in qrels
    assert args.qrels_path[-5:] == '.json'
    qrels = pd.read_json(args.qrels_path, orient='columns', typ='frame')
    qrels = qrels.groupby([constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]).apply(lambda x: (list(set(x[constants.RANKING_INPUT_DOCUMENT_ID])))).to_frame(name=constants.RANKING_INPUT_DOCUMENT_ID).reset_index()
    pred_list = []
    label_list = []
    tic = time.time()
    for i in qrels.index:
        qid = qrels.loc[i, constants.RANKING_INPUT_QUERY_ID]
        query = qrels.loc[i, constants.RANKING_INPUT_QUERY_NAME]
        all_pos_pid = qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]
        all_pid, all_score = top_n_passage_ids(query, ir_model, args.n)
        all_label = [int(i in all_pos_pid) for i in all_pid]
        pred_list.extend([(qid, doc, query,passage_collection[doc], score) for doc,score in zip(all_pid,all_score)])
        label_list.extend(all_label)

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
                                             max_len=(args.max_query_length + args.max_passage_length), bool_np_array=True)
    else:
        query_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_query_length, bool_np_array = True)
        context_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_passage_length, bool_np_array = True)

    if args.architecture == 'cross':
        dataset = CombinedInferenceDataset(pred_list, transform)
    else:
        dataset = InferenceDataset(pred_list, query_transform, context_transform)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size, shuffle=False)

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
    predictions = predict(dataloader, model, device)
    #predictions = [0]*len(label_list)
    df = pd.DataFrame(pred_list, columns = ['qid','pid','query','passage','ir_score'])
    df['model_score'] = predictions
    df['label'] = label_list
    df.to_csv(os.path.join(args.save_dir, args.prediction_file), sep='\t')
    toc = time.time()
    df = pd.read_csv(os.path.join(args.save_dir, args.prediction_file), sep='\t')
    all_rr, prediction_final = rr_for_wt(df, wt=1.0, n=50, return_df=True)
    print('MRR on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), all_rr.mean().iloc[0]))
    print('proportion_rank_<=5 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), (all_rr >= 1 / 5).mean().iloc[0]))
    print('proportion_rank_<=3 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), (all_rr >= 1 / 3).mean().iloc[0]))
    print('proportion_rank_<=2 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), (all_rr >= 1 / 2).mean().iloc[0]))

    plot_distr(all_rr.iloc[:, 0], os.path.join(args.save_dir,'prediction_final.pdf'), 'Distribution of RR@{:d}'.format(10), (0.1, 0.2, 0.33, 0.5, 1))