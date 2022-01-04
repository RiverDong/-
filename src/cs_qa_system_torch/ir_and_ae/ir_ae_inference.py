import argparse
import os
import time
import timeit
import logging
import pandas as pd
import numpy as np
import re
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import SquadExample, squad_convert_examples_to_features, AutoModelForQuestionAnswering, AutoTokenizer, \
    AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.data.processors.squad import SquadResult, squad_convert_example_to_features_init, \
    squad_convert_example_to_features
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

import constants
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from information_retrieval.ir_dataset import CombinedInferenceDataset, InferenceDataset
from information_retrieval.ir_model import BiEncoderModel, CrossEncoderModel
from information_retrieval.ir_transform import CombinedRankingTransform, RankingTransform
from ir_and_ae.answer_formatter import reformat_answer
from ir_utils import load_document_collection, load_ranking_model, predict_ranking_model
from utils import logging_config, plot_distr

logger = logging.getLogger(__name__)

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


def get_original_passage(x, passage_collection_ae):
    output = ''
    sentinel = ''
    for i in x.values:
        if i in passage_collection_ae:
            output = output + sentinel + passage_collection_ae[i]
            sentinel = '.'
    return output


def get_passages_for_answerextraction(x, passage_collection_ae, model_score_col, ir_score_col, wt, ir_top_n):
    ir_score_sum = x[ir_score_col].sum()
    if ir_score_sum == 0:
        x['model_ir_score'] = wt * x[model_score_col]
    else:
        x['model_ir_score'] = wt * x[model_score_col] + (1 - wt) * (x[ir_score_col] / ir_score_sum)
    return x.sort_values('model_ir_score', ascending=False).head(ir_top_n).groupby(['qid', 'query']).apply(
        lambda x: get_original_passage(x['pid'], passage_collection_ae)).reset_index()


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


def get_squad_example(x, answer_labels):
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
        answer_text=answer,
        start_position_character=index,
        title=None,
        is_impossible=is_impossible,
        answers=[{"text": answer}] if not is_impossible else [],
    )


def to_list(tensor):
    return tensor.detach().cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to predict the relevant documents for a '
                                                 'given query in Question Answering system')
    parser.add_argument('--gpu', type=str, default='', help='gpus to run model on')
    parser.add_argument('--output_dir', type=str, default='inference_ir_ae', help='path to inference output')
    parser.add_argument('--passage_collection_path', type=str,
                        help='path of the passage collection')
    parser.add_argument('--ir_model_name', type=str, default='BM25Okapi',
                        help='name of the information retrieval model to use')
    parser.add_argument('--word_tokenizer_name', type=str, default='simple_word_tokenizer',
                        help='name of the word tokenizer for the information retrieval model to use')
    parser.add_argument('--qrels_path', type=str,
                        help='path of the qrels file')
    parser.add_argument('--index_top_n', type=int, default=50,
                        help='number of relevant passages we want the IR index model BM25 to return')
    parser.add_argument('--model_name_or_path', type=str, default='',
                        help='model artifacts folder for IR')
    parser.add_argument("--architecture", required=True, type=str, help='[bi, cross]')
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--max_passage_length", default=256, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')

    parser.add_argument('--ir_model_weight', type=float, default=.46,
                        help='weightage to give to ml ir model score relative to index model (like bm25)')
    parser.add_argument('--ir_top_n', type=int, default=1,
                        help='number of documents to consider for final result of IR component')
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument('--ae_params_dir', type=str,
                        help='path of the params directory of answer extraction')
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging_config(args.output_dir, name='inference')
    logger.info("The input args are: \n{}".format(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("The inference files for (ir+ae) will be saved to: {}".format(args.output_dir))

    result_dir = ''
    passage_collection_ir, pid_passage_tuple, passage_collection_ae = load_passage_collection(
        args.passage_collection_path)
    word_tokenizer = WordTokenizerFactory.create_word_tokenizer(args.word_tokenizer_name)
    ir_model = IRModelFactory.create_ir_model(ir_model_name=args.ir_model_name, ir_model_path=result_dir,
                                              corpus=pid_passage_tuple, tokenizer=word_tokenizer)

    # Retrieve top n relevant passages for each query in qrels
    assert args.qrels_path[-5:] == '.json'
    qrels_all = pd.read_json(args.qrels_path, orient='columns', typ='frame')
    answer_labels = qrels_all.groupby([constants.RANKING_INPUT_QUERY_ID]).apply(
        lambda x: list(x[constants.ANSWER_EXTRACTION_ANSWER])).to_dict()
    qrels = qrels_all.groupby([constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]).apply(
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
        pred_list.extend(
            [(qid, doc, query, passage_collection_ir[doc], score) for doc, score in zip(all_pid, all_score)])
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
        transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=512, bool_np_array=True)
    else:
        query_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_query_length, bool_np_array=True)
        context_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_passage_length, bool_np_array=True)

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
    all_rr, prediction_final = rr_for_wt(df, wt=args.ir_model_weight, n=50, return_df=True)
    print('MRR on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(), all_rr.mean().iloc[0]))
    print('proportion_rank_<=5 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(),
                                                                                 (all_rr >= 1 / 5).mean().iloc[0]))
    print('proportion_rank_<=3 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(),
                                                                                 (all_rr >= 1 / 3).mean().iloc[0]))
    print('proportion_rank_<=2 on test set with {0:d} queries is {1:.4f}'.format(df['qid'].nunique(),
                                                                                 (all_rr >= 1 / 2).mean().iloc[0]))

    plot_distr(all_rr.iloc[:, 0], os.path.join(args.output_dir, 'prediction_final.pdf'),
               'Distribution of RR@{:d}'.format(10), (0.1, 0.2, 0.33, 0.5, 1))

    ##################### Answer Extraction Component ##############

    model_ae = AutoModelForQuestionAnswering.from_pretrained(args.ae_params_dir)  # , force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ae_params_dir, do_lower_case=args.do_lower_case)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    model_ae.to(device)

    x = df.groupby(['qid', 'query']).apply(
        lambda x: model_ir_score(x, 'model_score', 'ir_score', args.ir_model_weight)).reset_index(drop=True)
    y = x.groupby(['qid', 'query']).apply(
        lambda x: x.sort_values('model_ir_score', ascending=False).head(args.ir_top_n)).reset_index(drop=True)
    prediction_df = y.groupby(['qid', 'query']).apply(
        lambda x: get_original_passage(x['pid'], passage_collection_ae)).reset_index()
    prediction_df.columns = ['qid', 'query', 'passage']
    examples = []
    for index, row in prediction_df.iterrows():
        examples.append(get_squad_example(row, answer_labels))

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
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * 1).to(args.device)}
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
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
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
