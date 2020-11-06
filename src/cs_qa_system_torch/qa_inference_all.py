import argparse
import os
import time
import timeit

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import SquadExample, squad_convert_examples_to_features, AutoModelForQuestionAnswering, AutoTokenizer
from transformers.data.processors.squad import SquadResult, squad_convert_example_to_features_init, \
    squad_convert_example_to_features

import constants
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from ir_inference import load_collection, top_n_passage_ids
from ranking_dataset import CombinedInferenceDataset, CombinedRankingDataset, InferenceDataset
from ranking_metrics import _mrr
from ranking_model import BertBiEncoderModel, BertCrossEncoderModel
from ranking_transform import CombinedRankingTransform, RankingTransform
from ranking_utils import MODEL_CLASSES, get_prediction, store_prediction_results, predict, load_passage_collection
from utils import logging_config, plot_distr
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)


class ScorePrediction   :
    passage_collection = None
    ir_model = None
    ml_model = None
    transform = None
    query_transform = None
    context_transform = None
    ir_n = 5
    passage_collection_path = '/data/QAData/AnswerExtractionData/amazon/production_collection.json'
    word_tokenizer_name = 'simple_word_tokenizer_no_stopwords_stem'
    ir_model_name = 'BM25Okapi'
    params_dir = '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/artifacts/'
    model_type = 'bert'
    model_type_name = 'bert-base-uncased'
    architecture = 'cross'
    max_query_length = 64
    max_passage_length = 256

    params_dir_ae = '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output'
    do_lower_case = True
    ae_model = None
    ae_tokenizer = None
    wt = 0.46
    final_n = 2
    max_seq_length = 384
    doc_stride = 128
    model_type_ae = 'bert'
    n_best_size = 10
    max_answer_length = 150

    @classmethod
    def load_ir_model(cls):
        result_dir = ''
        cls.passage_collection, pid_passage_tuple, cls.passage_collection_ae = load_passage_collection(cls.passage_collection_path)
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
        cls.ae_model = AutoModelForQuestionAnswering.from_pretrained(cls.params_dir_ae)  # , force_download=True)
        cls.ae_tokenizer = AutoTokenizer.from_pretrained(cls.params_dir_ae, do_lower_case=cls.do_lower_case)
        cls.device = "cpu"
        cls.ae_model.to(cls.device)

    @classmethod
    def get_documents(cls,query):
        if cls.ml_model == None:
            cls.load_ir_model()
        pred_list = [(1, doc, query, cls.passage_collection[doc], score) for doc,score in zip(*top_n_passage_ids(query, cls.ir_model, cls.ir_n))]
        if cls.architecture == 'cross':
            dataset = CombinedInferenceDataset(pred_list, cls.transform)
        else:
            dataset = InferenceDataset(pred_list, cls.query_transform, cls.context_transform)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        predictions = predict(dataloader, cls.ml_model, "cpu")
        df = pd.DataFrame(pred_list, columns=['qid', 'pid', 'query', 'passage', 'ir_score'])
        df['ml_score'] = predictions
        x = df.groupby(['qid', 'query']).apply(
            lambda x: model_ir_score(x, 'ml_score', 'ir_score', cls.wt)).reset_index(drop=True)
        y = x.groupby(['qid', 'query']).apply(
            lambda x: x.sort_values('model_ir_score', ascending=False).head(cls.final_n)).reset_index(drop=True)
        prediction_df = y.groupby(['qid', 'query']).apply(
            lambda x: get_original_passage(x['pid'], cls.passage_collection_ae)).reset_index()
        prediction_df.columns = ['qid', 'query', 'passage']


        example = SquadExample(
            qas_id=0,
            question_text=query,
            context_text=prediction_df.iloc[0]['passage'],
            answer_text=None,
            start_position_character=None,
            title=None,
            answers=None,
        )
        squad_convert_example_to_features_init(cls.ae_tokenizer)
        features = squad_convert_example_to_features(example, cls.max_seq_length, cls.doc_stride, cls.max_query_length, False)

        new_features = []
        unique_id = 1000000000
        for feature in features:
            feature.example_index = 0
            feature.unique_id = unique_id
            new_features.append(feature)
            unique_id += 1
        features = new_features
        del new_features

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index,
                                all_cls_index, all_p_mask)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=len(dataset))

        all_results = []
        for batch in dataloader:
            cls.ae_model.eval()
            batch = tuple(t.to(cls.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if cls.model_type_ae in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                    del inputs["token_type_ids"]

                feature_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if cls.model_type_ae in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(cls.ae_model, "config") and hasattr(cls.ae_model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                        )
                outputs = cls.ae_model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

        predictions_realtime = compute_predictions_logits(
            [example],
            features,
            all_results,
            cls.n_best_size,
            cls.max_answer_length,
            cls.do_lower_case,
            None,
            None,
            None,
            True,
            True,
            0.0,
            cls.ae_tokenizer,
        )

        return predictions_realtime[0]

def get_rr(df, n, qid_col='qid', score_col='model_ir_score', label_col='label'):
    return df.groupby(qid_col).apply(lambda x: _mrr(x[score_col].tolist(), x[label_col].tolist(), n))

def get_original_passage(x,passage_collection_ae):
    output = ''
    sentinel = ''
    for i in x.values:
        if i in passage_collection_ae:
            output = output + sentinel + passage_collection_ae[i]
            sentinel = '.'
    return output

def get_passages_for_answerextraction(x, passage_collection_ae, model_score_col, ir_score_col, wt, final_n):
    ir_score_sum = x[ir_score_col].sum()
    if ir_score_sum == 0:
        x['model_ir_score'] = wt * x[model_score_col]
    else:
        x['model_ir_score'] = wt * x[model_score_col] + (1 - wt) * (x[ir_score_col] / ir_score_sum)
    return x.sort_values('model_ir_score', ascending=False).head(final_n).groupby(['qid', 'query']).apply(lambda x: get_original_passage(x['pid'],passage_collection_ae)).reset_index()


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


def get_squad_example(x,answer_labels):
    qid = x['qid']
    query = x['query']
    passage = x['passage']
    answers = answer_labels[qid] if qid in answer_labels else []
    answer = ''
    index = -1
    if len(answers) > 0:
        for i in answers:
            index = x['passage'].find(i)
            if index != -1:
                answer = i
                break
    if index == -1:
        is_impossible = True
    else:
        is_impossible = False
    return SquadExample(
        qas_id=qid,
        question_text=query,
        context_text=passage,
        answer_text= answer,
        start_position_character=index,
        title=None,
        is_impossible=is_impossible,
        answers=[{"text":answer}] if not is_impossible else [],
    )

def to_list(tensor):
    return tensor.detach().cpu().tolist()

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
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument('--gpu', type=str, help='gpus to run model on')
    parser.add_argument('--save_dir', type=str, default='inference', help='path to inference output')
    parser.add_argument("--test_batch_size", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')
    parser.add_argument('--wt', type=float, default=.46,
                        help='weightage to give ti IR score relative to ml model score')
    parser.add_argument('--final_n', type=int, default=1,
                        help='number of documents to consider for final result of IR component')
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument('--ae_params_dir', type=str,
                        help='path of the params directory of answer extraction')
    parser.add_argument('--do_lower_case', action="store_true", help="true of convert data to lower case")
    parser.add_argument("--model_ae_type", default='bert', type=str, help='[bert, distilbert]')
    parser.add_argument(
        "--n_best_size",
        default=10,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=100,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )



    args = parser.parse_args()
    result_dir = ''
    passage_collection_ir, pid_passage_tuple, passage_collection_ae = load_passage_collection(args.passage_collection_path)
    word_tokenizer = WordTokenizerFactory.create_word_tokenizer(args.word_tokenizer_name)
    ir_model = IRModelFactory.create_ir_model(ir_model_name=args.ir_model_name, ir_model_path=result_dir, corpus=pid_passage_tuple, tokenizer=word_tokenizer)

    # Retrieve top n relevant passages for each query in qrels
    assert args.qrels_path[-5:] == '.json'
    qrels_all = pd.read_json(args.qrels_path, orient='columns', typ='frame')
    answer_labels = qrels_all.groupby([constants.RANKING_INPUT_QUERY_ID]).apply(lambda x: list(x[constants.ANSWER_EXTRACTION_ANSWER])).to_dict()
    qrels = qrels_all.groupby([constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]).apply(lambda x: (list(set(x[constants.RANKING_INPUT_DOCUMENT_ID])))).to_frame(name=constants.RANKING_INPUT_DOCUMENT_ID).reset_index()
    pred_list = []
    label_list = []
    tic = time.time()
    for i in qrels.index:
        qid = qrels.loc[i, constants.RANKING_INPUT_QUERY_ID]
        query = qrels.loc[i, constants.RANKING_INPUT_QUERY_NAME]
        all_pos_pid = qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]
        all_pid, all_score = top_n_passage_ids(query, ir_model, args.n)
        all_label = [int(i in all_pos_pid) for i in all_pid]
        pred_list.extend([(qid, doc, query,passage_collection_ir[doc], score) for doc,score in zip(all_pid,all_score)])
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
    all_rr, prediction_final = rr_for_wt(df, wt=args.wt, n=50, return_df=True)
    print('MRR on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), all_rr.mean().iloc[0]))
    print('proportion_rank_<=5 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), (all_rr >= 1 / 5).mean().iloc[0]))
    print('proportion_rank_<=3 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), (all_rr >= 1 / 3).mean().iloc[0]))
    print('proportion_rank_<=2 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), (all_rr >= 1 / 2).mean().iloc[0]))

    plot_distr(all_rr.iloc[:, 0], os.path.join(args.save_dir,'prediction_final.pdf'), 'Distribution of RR@{:d}'.format(10), (0.1, 0.2, 0.33, 0.5, 1))

    ##################### Answer Extraction Component ##############

    model_ae = AutoModelForQuestionAnswering.from_pretrained(args.ae_params_dir)  # , force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ae_params_dir, do_lower_case=args.do_lower_case)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    model_ae.to(device)

    x = df.groupby(['qid', 'query']).apply(lambda x: model_ir_score(x, 'model_score', 'ir_score',args.wt)).reset_index(drop=True)
    y = x.groupby(['qid', 'query']).apply(lambda x: x.sort_values('model_ir_score', ascending=False).head(args.final_n)).reset_index(drop=True)
    prediction_df = y.groupby(['qid', 'query']).apply(lambda x: get_original_passage(x['pid'], passage_collection_ae)).reset_index()
    prediction_df.columns = ['qid', 'query', 'passage']
    examples = []
    for index, row in prediction_df.iterrows():
        examples.append(get_squad_example(row,answer_labels))

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model_ae, torch.nn.DataParallel):
        model_ae = torch.nn.DataParallel(model_ae)

    print("***** Running evaluation {} *****".format(''))
    print("  Num examples = ", len(dataset))
    print("  Batch size = ", args.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model_ae.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_ae_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_ae_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model_ae, "config") and hasattr(model_ae.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model_ae(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    print("  Evaluation done in total {} secs ({} sec per example)".format(evalTime, evalTime / len(dataset)))

    prefix = ''
    # Compute predictions
    output_prediction_file = os.path.join(args.ae_params_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.ae_params_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.ae_params_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_ae_type in ["xlnet", "xlm"]:
        start_n_top = model_ae.config.start_n_top if hasattr(model_ae, "config") else model_ae.module.config.start_n_top
        end_n_top = model_ae.config.end_n_top if hasattr(model_ae, "config") else model_ae.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)