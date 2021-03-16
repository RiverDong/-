import argparse
import os
import time
import logging
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import constants
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from information_retrieval.ir_dataset import CombinedInferenceDataset, InferenceDataset, SimpleDataset
from information_retrieval.ir_model import BiEncoderModel, CrossEncoderModel
from information_retrieval.ir_transform import CombinedRankingTransform, RankingTransform
from information_retrieval.ir_utils import predict, load_passage_collection, get_ir_model_attributes
from ir_loss import dot_product_scores
from utils import logging_config, plot_distr

logger = logging.getLogger(__name__)


class ScorePrediction:
    passage_collection = None
    passage_collection_ae = None
    ir_model = None
    ml_model = None
    transform = None
    query_transform = None
    context_transform = None
    index_top_n = 10
    passage_collection_path = '/data/QAData/InformationRetrievalData/amazon/production_collection.json'
    word_tokenizer_name = 'simple_word_tokenizer_no_stopwords_stem'
    ir_model_name = 'BM25Okapi'

    ir_model_weight = 0.50
    ir_top_n = 10
    params_dir = '/data/qyouran/QABot/output_torch_train/ir_finetuned_large'
    do_lower_case = True
    architecture = 'cross'

    max_length_cross_architecture = 384
    max_query_length = 64
    max_passage_length = 256

    ir_ml_model_score_threshold = 0.80

    @classmethod
    def top_n_passage_ids(cls, query, ir_model, n):
        # For a given query, get pids of the sorted top n relevant passages and their scores using the IR model
        try:
            return ir_model.get_top_n_single(query, n)
        except (AttributeError, TypeError) as error:
            print('Error: please make sure the IR model has a "get_top_n_single" method with desired signature')
            raise error

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
            cls.transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=cls.max_length_cross_architecture,
                                                     bool_np_array=True)
        else:
            cls.query_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_query_length,
                                                   bool_np_array=True)
            cls.context_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_passage_length,
                                                     bool_np_array=True)

        cls.ml_model.load_state_dict(torch.load(os.path.join(cls.params_dir, 'pytorch_model.pt')))

    @classmethod
    def model_ir_score(cls, x, model_score_col, ir_score_col, wt):
        ir_score_sum = x[ir_score_col].sum()
        if ir_score_sum == 0:
            x['model_ir_score'] = wt * x[model_score_col]
        else:
            x['model_ir_score'] = wt * x[model_score_col] + (1 - wt) * (x[ir_score_col] / ir_score_sum)
        return x

    @classmethod
    def get_documents(cls, query):
        if cls.ml_model == None:
            cls.load_ir_model()
        pred_list = [(1, doc, query, cls.passage_collection[doc], score) for doc, score in
                     zip(*cls.top_n_passage_ids(query, cls.ir_model, cls.index_top_n))]

        if 'cross' in cls.architecture:
            dataset = CombinedInferenceDataset(pred_list, cls.transform)
        else:
            dataset = InferenceDataset(pred_list, cls.query_transform, cls.context_transform)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        predictions = predict(dataloader, cls.ml_model, "cpu")
        df = pd.DataFrame(pred_list, columns=['qid', 'pid', 'query', 'passage', 'ir_score'])
        df['ml_score'] = predictions
        df = df[df.ml_score >= cls.ir_ml_model_score_threshold].reset_index(drop=True)
        cls.model_ir_score(df, 'ml_score', 'ir_score', cls.ir_model_weight)
        df.sort_values('model_ir_score', ascending=False)
        df['passage'] = df['pid'].apply(lambda x: cls.passage_collection_ae[x])
        return df.head(cls.ir_top_n)


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


def get_rr(df, n):
    return df.groupby(constants.RANKING_INPUT_QUERY_ID).apply(lambda x: _mrr(x[constants.WEIGHTED_RANKING_SCORE].tolist(), x[constants.RANKING_INPUT_LABEL_NAME].tolist(), n))


def model_ir_score(x, wt):
    ir_score_sum = x[constants.RANKING_SCORE].sum()
    if ir_score_sum == 0:
        x[constants.WEIGHTED_RANKING_SCORE] = wt * x[constants.RERANKING_SCORE]
    else:
        x[constants.WEIGHTED_RANKING_SCORE] = wt * x[constants.RERANKING_SCORE] + (1 - wt) * (x[constants.RANKING_SCORE] / ir_score_sum)
    return x


def rr_for_wt(prediction_df, wt, n=10, return_df=False):
    if not constants.RERANKING_SCORE in prediction_df.columns:
        prediction_df[constants.RERANKING_SCORE] = 0.0
        wt = 0.0
    prediction_df = prediction_df.groupby(constants.RANKING_INPUT_QUERY_ID).apply(
        lambda x: model_ir_score(x, wt)).reset_index(drop=True)
    if return_df:
        return get_rr(prediction_df, n).to_frame(
            name='wt_{:.2f}'.format(wt)), prediction_df
    else:
        return get_rr(prediction_df, n).to_frame(name='wt_{:.2f}'.format(wt))


def rank_bulk_prediction(qrels_path_or_data_frame, document_collection_tuple, model_type, model, top_n, query_transform, context_transform, context_embeddings, rank_batch_size, device, rerank = False):

    document_collection_map, idx_to_doc, doc_to_idx = document_collection_tuple

    qrels = None
    queries = None

    if rerank and isinstance(qrels_path_or_data_frame, pd.DataFrame):
        query_passage_data = list(qrels_path_or_data_frame[[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                                                       constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME, constants.RANKING_SCORE]].to_records(index=False))
    else:
        qrels = pd.read_json(qrels_path_or_data_frame, orient='columns', typ='frame')
        qrels = qrels.groupby([constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]).apply(
            lambda x: (list(set(x[constants.RANKING_INPUT_DOCUMENT_ID])))).to_frame(
            name=constants.RANKING_INPUT_DOCUMENT_ID).reset_index()
        queries = list(
            qrels[[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]].to_records(index=False))
        query_passage_data = [(qid, pid, query, passage, 0.0) for (qid,query) in queries for (pid,passage) in idx_to_doc]


    if model_type == 'cross':
        dataset = CombinedInferenceDataset(query_passage_data, context_transform)
        dataloader = DataLoader(dataset, batch_size=args.rank_batch_size, shuffle=False)
        prediction_score = predict(dataloader, model, device)
        if not rerank:
            label_list = []
            for i in qrels.index:
                all_pos_pid = qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]
                all_label = [0] * len(idx_to_doc)
                for i in all_pos_pid:
                    all_label[doc_to_idx[i]] = 1
                label_list.extend(all_label)
            df = pd.DataFrame(query_passage_data, columns=[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                                                           constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME, constants.RANKING_SCORE])
            df[constants.RANKING_SCORE] = prediction_score
            df[constants.RANKING_INPUT_LABEL_NAME] = label_list
            df = df.groupby([constants.RANKING_INPUT_QUERY_ID]).apply(lambda x:x.sort_values([constants.RANKING_SCORE],  ascending=False)).reset_index(drop=True)
            df = df.groupby([constants.RANKING_INPUT_QUERY_ID]).head(top_n)
        else:
            df = qrels_path_or_data_frame
            df[constants.RERANKING_SCORE] = prediction_score
        return df

    if model_type != 'cross' and rerank:
        raise NotImplementedError

    if model_type == 'BM25':
        prediction = [list(zip(*top_n_passage_ids(query, model, top_n))) for id, query in queries]
    elif model_type == 'bi':
        query_dataset = SimpleDataset(queries, query_transform)
        query_dataloader = DataLoader(query_dataset, batch_size=rank_batch_size, shuffle=False)
        query_embeddings = []
        model.eval()
        for step, batch in enumerate(query_dataloader, start=1):
            input_batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                query_embeddings.append(model.module.query_model(*input_batch)[-1])
        query_embeddings = torch.cat(query_embeddings, dim=0)
        scores = dot_product_scores(query_embeddings, context_embeddings)
        sotmax_scores = F.softmax(scores, dim=1)
        output = torch.sort(sotmax_scores, dim=1, descending=True)
        values = output[0][:, 0:top_n].tolist()
        indices = output[1][:, 0:top_n].tolist()
        prediction = [list(zip(i[0], i[1])) for i in zip(indices, values)]
    else:
        raise NotImplementedError

    pred_list = []
    label_list = []
    for i in qrels.index:
        qid = qrels.loc[i, constants.RANKING_INPUT_QUERY_ID]
        query = qrels.loc[i, constants.RANKING_INPUT_QUERY_NAME]
        all_pos_pid = qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]
        if model_type == 'BM25':
            all_label = [int(i in all_pos_pid) for (i,score) in prediction[i]]
            pred_list.extend([(qid, doc, query, document_collection_map[doc], score) for (doc, score) in prediction[i]])
        else:
            all_label = [int(idx_to_doc[index][0] in all_pos_pid) for (index, score) in prediction[i]]
            pred_list.extend(
                [(qid, idx_to_doc[index][0], query, document_collection_map[idx_to_doc[index][0]], score) for (index, score)
                 in prediction[i]])
        label_list.extend(all_label)
    df = pd.DataFrame(pred_list, columns=[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                                                           constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME, constants.RANKING_SCORE])
    df[constants.RANKING_INPUT_LABEL_NAME] = label_list
    return df


def rank_prediction_biencoder(query, model, top_n, query_transform, context_embeddings, device):
    query_input = (torch.tensor(t.reshape(1, -1), dtype=torch.long).to(device) for t in query_transform(query))
    model.eval()
    with torch.no_grad():
        query_embedding = model.query_model(*query_input)[-1]

    scores = dot_product_scores(query_embedding, context_embeddings)
    sotmax_scores = F.softmax(scores, dim=1)
    output = torch.sort(sotmax_scores, dim=1, descending=True)
    values = output[0][:, 0:top_n].tolist()
    indices = output[1][:, 0:top_n].tolist()
    result = [list(zip(i[0], i[1])) for i in zip(indices, values)]
    return result

def getRankPrediction(qrels_path, passage_collection_path, model_name_or_path, inference_batch_size, top_n, device, rerank = False):

    ## Reading collection file
    passage_collection, pid_passage_tuple, _ = load_passage_collection(passage_collection_path)
    doc_to_idx = {val[0]:index for index, val in enumerate(pid_passage_tuple)}
    collection_tuple = (passage_collection, pid_passage_tuple, doc_to_idx)

    query_transform = None
    context_transform = None
    context_embeddings = None

    result_dir = ''

    if model_name_or_path == constants.IR_BM25OKAPI:
        rank_type = 'BM25'
        word_tokenizer = WordTokenizerFactory.create_word_tokenizer(constants.IR_MODELS[constants.IR_BM25OKAPI]['word_tokenizer_name'])
        model = IRModelFactory.create_ir_model(ir_model_name=model_name_or_path,
                                                    ir_model_path=result_dir,
                                                    corpus=pid_passage_tuple,
                                                    tokenizer=word_tokenizer)
    else:
        params = torch.load(os.path.join(model_name_or_path, 'training_args.bin'))
        rank_type = params.architecture
        model, tokenizer, config, query_tokenizer, query_config = get_ir_model_attributes(model_name_or_path=model_name_or_path,
                         architecture=params.architecture,
                         projection_dim=params.projection_dim,
                         do_lower_case=params.do_lower_case)

        if os.path.exists(model_name_or_path):
            model.load_state_dict(torch.load(os.path.join(model_name_or_path, 'pytorch_model.pt')))

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        if params.architecture == 'cross':
            context_transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=params.max_query_passage_length,
                                                 bool_np_array=True)
        elif params.architecture == 'bi':
            query_transform = RankingTransform(tokenizer=query_tokenizer, max_len=params.max_query_length,
                                               bool_np_array=True)
            context_transform = RankingTransform(tokenizer=tokenizer, max_len=params.max_passage_length,
                                                 bool_np_array=True)

            context_dataset = SimpleDataset(pid_passage_tuple, context_transform)
            context_dataloader = DataLoader(context_dataset, batch_size=inference_batch_size)

            context_embeddings = []
            model.eval()
            for step,batch in enumerate(context_dataloader,start=1):
                input_batch = tuple(t.to(device) for t in batch)
                with torch.no_grad():
                    context_embeddings.append(model.module.context_model(*input_batch)[-1])
            context_embeddings = torch.cat(context_embeddings, dim=0)

    rank_results_df = rank_bulk_prediction(qrels_path, collection_tuple, rank_type, model, top_n, query_transform,
                         context_transform, context_embeddings, inference_batch_size, device, rerank = rerank)
    return rank_results_df

def displayResult(df):
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



    ## Initial Ranking Arguments
    parser.add_argument('--rank_model_name_or_path', type=str, default='BM25Okapi',
                        help='name of the information retrieval model to use')
    parser.add_argument("--rank_batch_size", default=128, type=int, help="The batch size to use for each ranking inference")
    parser.add_argument('--rank_top_n', type=int, default=50,
                        help='number of relevant passages we want the IR model to return')


    ##reranking arguments
    parser.add_argument(
        "--do_rerank", action="store_true", help="Set this flag if you are reranking the model."
    )
    parser.add_argument('--rerank_model_name_or_path', type=str, default='BM25Okapi',
                        help='name of the information retrieval model to use')
    parser.add_argument("--rerank_batch_size", default=128, type=int, help="The batch size to use for each ranking inference")
    parser.add_argument('--rerank_top_n', type=int, default=5,
                        help='number of relevant passages we want the IR model to return')

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

    result_dir = ''


    rank_results_df = getRankPrediction(args.qrels_path, args.passage_collection_path, args.rank_model_name_or_path, args.rank_batch_size, args.rank_top_n, device)




    rank_results_df.to_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
    print('Rank results:\n')
    displayResult(rank_results_df)

    if args.do_rerank:
        rerank_results_df = getRankPrediction(rank_results_df, args.passage_collection_path, args.rerank_model_name_or_path, args.rerank_batch_size,
                                              args.rerank_top_n, device, rerank=True)
        rerank_results_df.to_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
        print('ReRank results:\n')
        displayResult(rerank_results_df)

    if args.only_print_results:
        df = pd.read_csv(os.path.join(args.output_dir, args.prediction_file), sep='\t')
        displayResult(df)


