import argparse
import os
import time
import logging
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import constants
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from information_retrieval.ir_dataset import CombinedInferenceDataset, InferenceDataset
from information_retrieval.ir_model import BiEncoderModel, CrossEncoderModel
from information_retrieval.ir_transform import CombinedRankingTransform, RankingTransform
from information_retrieval.ir_utils import predict, load_passage_collection
from utils import logging_config, plot_distr

logger = logging.getLogger(__name__)


class ScorePrediction:
    passage_collection = None
    ir_model = None
    ml_model = None
    transform = None
    query_transform = None
    context_transform = None
    index_top_n = 10
    passage_collection_path = '/data/QAData/InformationRetrievalData/amazon/production_collection.json'
    word_tokenizer_name = 'simple_word_tokenizer_no_stopwords_stem'
    ir_model_name = 'BM25Okapi'

    params_dir = '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase_finetuned/'
    do_lower_case = True
    architecture = 'cross'

    max_query_length = 64
    max_passage_length = 256

    @classmethod
    def load_ir_model(cls):
        result_dir = ''
        cls.passage_collection, pid_passage_tuple, cls.passage_collection_ae = load_passage_collection(
            cls.passage_collection_path)
        word_tokenizer = WordTokenizerFactory.create_word_tokenizer(cls.word_tokenizer_name)
        cls.ir_model = IRModelFactory.create_ir_model(ir_model_name=cls.ir_model_name, ir_model_path=result_dir,
                                                      corpus=pid_passage_tuple, tokenizer=word_tokenizer)

        config = AutoConfig.from_pretrained(cls.params_dir)
        tokenizer = AutoTokenizer.from_pretrained(cls.params_dir, do_lower_case=cls.do_lower_case)
        if cls.architecture in ['poly', 'bi', 'cross']:
            pre_model = AutoModel.from_pretrained(
                cls.params_dir,
                from_tf=bool(".ckpt" in cls.params_dir),
                config=config)
        else:
            pre_model = AutoModelForSequenceClassification.from_pretrained(
                cls.params_dir,
                from_tf=bool(".ckpt" in cls.params_dir),
                config=config)

        if cls.architecture == 'poly':
            cls.ml_model = BiEncoderModel(config, model=pre_model)
        elif cls.architecture == 'bi':
            cls.ml_model = BiEncoderModel(config, model=pre_model)
        elif cls.architecture == 'cross':
            cls.ml_model = CrossEncoderModel(config, model=pre_model)
        elif cls.architecture == 'cross-default':
            cls.ml_model = CrossEncoderModel(config, model=pre_model)
        else:
            raise ValueError("Wrong architecture name")

        if 'cross' in cls.architecture:
            cls.transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=512, bool_np_array=True)
        else:
            cls.query_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_query_length, bool_np_array=True)
            cls.context_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_passage_length, bool_np_array=True)

        cls.ml_model.load_state_dict(torch.load(os.path.join(cls.params_dir, 'pytorch_model.pt')))

    @classmethod
    def get_documents(cls, query):
        if cls.ml_model == None:
            cls.load_ir_model()
        pred_list = [(1, doc, query, cls.passage_collection_ae[doc], score) for doc, score in
                     zip(*top_n_passage_ids(query, cls.ir_model, cls.index_top_n))]

        if 'cross' in cls.architecture:
            dataset = CombinedInferenceDataset(pred_list, cls.transform)
        else:
            dataset = InferenceDataset(pred_list, cls.query_transform, cls.context_transform)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        predictions = predict(dataloader, cls.ml_model, "cpu")
        df = pd.DataFrame(pred_list, columns=['qid', 'pid', 'query', 'passage', 'ir_score'])
        df['ml_score'] = predictions

        return df


def top_n_passage_ids(query, ir_model, n):
    # For a given query, get pids of the sorted top n relevant passages and their scores using the IR model
    try:
        return ir_model.get_top_n_single(query, n)
    except (AttributeError, TypeError) as error:
        print('Error: please make sure the IR model has a "get_top_n_single" method with desired signature')
        raise error


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
    prediction_df = prediction_df.groupby('qid').apply(
        lambda x: model_ir_score(x, 'model_score', 'ir_score', wt)).reset_index(drop=True)
    if return_df:
        return get_rr(prediction_df, n, 'qid', 'model_ir_score', 'label').to_frame(
            name='wt_{:.2f}'.format(wt)), prediction_df
    else:
        return get_rr(prediction_df, n, 'qid', 'model_ir_score', 'label').to_frame(name='wt_{:.2f}'.format(wt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to predict the relevant documents for a '
                                                 'given query in Question Answering system')
    parser.add_argument('--gpu', type=str, default='', help='gpus to run model on')
    parser.add_argument('--output_dir', type=str, default='inference_ir', help='path to inference output')
    parser.add_argument('--index_top_n', type=int, default=50,
                        help='number of relevant passages we want the IR model to return')
    parser.add_argument('--qrels_path', type=str,
                        help='path of the qrels file')
    parser.add_argument('--test_data_path', type=str,
                        help='path of the test file')
    parser.add_argument('--passage_collection_path', type=str,
                        help='path of the passage collection')
    parser.add_argument('--ir_model_name', type=str, default='BM25Okapi',
                        help='name of the information retrieval model to use')
    parser.add_argument('--word_tokenizer_name', type=str, default='simple_word_tokenizer',
                        help='name of the word tokenizer for the information retrieval model to use')
    parser.add_argument('--model_name_or_path', type=str, default='model artifacts folder',
                        help='model artifacts file')
    parser.add_argument("--architecture", required=True, type=str, help='[bi, cross]')
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--max_length_cross_architecture", default=384, type=int)
    parser.add_argument("--max_passage_length", default=256, type=int)
    parser.add_argument("--max_query_length", default=32, type=int)

    parser.add_argument("--test_batch_size", default=128, type=int, help="Total batch size for eval.")
    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging_config(args.output_dir, name='inference')
    logger.info("The input args are: \n{}".format(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("The inference files will be saved to: {}".format(args.output_dir))

    result_dir = ''
    passage_collection, pid_passage_tuple, _ = load_passage_collection(args.passage_collection_path)
    word_tokenizer = WordTokenizerFactory.create_word_tokenizer(args.word_tokenizer_name)
    ir_model = IRModelFactory.create_ir_model(ir_model_name=args.ir_model_name, ir_model_path=result_dir,
                                              corpus=pid_passage_tuple, tokenizer=word_tokenizer)

    # Retrieve top n relevant passages for each query in qrels
    assert args.qrels_path[-5:] == '.json'
    qrels = pd.read_json(args.qrels_path, orient='columns', typ='frame')
    qrels = qrels.groupby([constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]).apply(
        lambda x: (list(set(x[constants.RANKING_INPUT_DOCUMENT_ID])))).to_frame(
        name=constants.RANKING_INPUT_DOCUMENT_ID).reset_index()
    pred_list = []
    label_list = []
    tic = time.time()
    for i in qrels.index:
        qid = qrels.loc[i, constants.RANKING_INPUT_QUERY_ID]
        query = qrels.loc[i, constants.RANKING_INPUT_QUERY_NAME]
        all_pos_pid = qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]
        all_pid, all_score = top_n_passage_ids(query, ir_model, args.index_top_n)
        all_label = [int(i in all_pos_pid) for i in all_pid]
        pred_list.extend([(qid, doc, query, passage_collection[doc], score) for doc, score in zip(all_pid, all_score)])
        label_list.extend(all_label)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.architecture in ['poly', 'bi', 'cross']:
        pre_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)
    else:
        pre_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)

    if args.architecture == 'poly':
        model = BiEncoderModel(config, model=pre_model)
    elif args.architecture == 'bi':
        model = BiEncoderModel(config, model=pre_model)
    elif args.architecture == 'cross':
        model = CrossEncoderModel(config, model=pre_model)
    elif args.architecture == 'cross-default':
        model = CrossEncoderModel(config, model=pre_model)
    else:
        raise ValueError("Wrong architecture name")

    transform = None
    query_transform = None
    context_transform = None
    if args.architecture == 'cross':
        transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=args.max_length_cross_architecture, bool_np_array=True)
    else:
        query_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_query_length, bool_np_array = True)
        context_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_passage_length, bool_np_array = True)

    if 'cross' in args.architecture:
        dataset = CombinedInferenceDataset(pred_list, transform)
    else:
        dataset = InferenceDataset(pred_list, query_transform, context_transform)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.pt')))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)


    predictions = predict(dataloader, model, device)
    # predictions = [0]*len(label_list)
    df = pd.DataFrame(pred_list, columns=['qid', 'pid', 'query', 'passage', 'ir_score'])
    df['model_score'] = predictions
    df['label'] = label_list
    df.to_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
    toc = time.time()
    df = pd.read_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
    all_rr, prediction_final = rr_for_wt(df, wt=1.0, n=50, return_df=True)
    print('MRR on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), all_rr.mean().iloc[0]))
    print('proportion_rank_<=5 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(),
                                                                                 (all_rr >= 1 / 5).mean().iloc[0]))
    print('proportion_rank_<=3 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(),
                                                                                 (all_rr >= 1 / 3).mean().iloc[0]))
    print('proportion_rank_<=2 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(),
                                                                                 (all_rr >= 1 / 2).mean().iloc[0]))

    plot_distr(all_rr.iloc[:, 0], os.path.join(args.output_dir, 'prediction_final.pdf'),
               'Distribution of RR@{:d}'.format(10), (0.1, 0.2, 0.33, 0.5, 1))
