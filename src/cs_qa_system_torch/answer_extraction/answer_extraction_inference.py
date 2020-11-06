
import argparse
import json
import os
import logging

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering

from answer_extraction.answer_extraction_utils import get_features, evaluate

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This module is used to predict the answer '
                                                 'given query and document in Question Answering system')

    parser.add_argument('--save_dir', type=str, default='inference', required=True, help='path to inference output')
    parser.add_argument('--params_dir', type=str, default=None, required=True, help="path of the artifacts directory")
    parser.add_argument("--test_file", type=str, default=None, required=True, help="file to do evaluation")
    parser.add_argument('--gpu', type=str, default="")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, "
                             "how much stride to take between chunks.")
    parser.add_argument("--test_batch_size", default=32, type=int, help="Total batch size for test data")

    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. "
                             "This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')


    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')




    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.params_dir)
    with open(args.test_file) as f:
        dataset_json = json.load(f)
        test_dataset = dataset_json['data']
        test_examples, test_features = get_features(args.test_file, tokenizer, args.max_seq_length, args.doc_stride,
                                     args.max_query_length, False, None, None)
        logger.info("***** Test *****")
        logger.info("  Num orig examples = %d", len(test_examples))
        logger.info("  Num split examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.test_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size)
        #model_eval = BertForQuestionAnswering.from_pretrained(args.params_dir)
        model_eval = BertForQuestionAnswering.from_pretrained('bert-base-cased')
        checkpoint = torch.load(os.path.join(args.params_dir, 'pytorch_model.bin'))
        model_eval.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.device_count() > 1:
            model_eval = nn.DataParallel(model_eval)
        model_eval.to(device)
        result, _, _ = evaluate(args, model_eval, device, test_dataset,
                                test_dataloader, test_examples, test_features)
    #
    #     na_prob_thresh = 1.0
    #     if args.version_2_with_negative:
    #         eval_result_file = os.path.join(args.save_dir, "eval_results.txt")
    #         if os.path.isfile(eval_result_file):
    #             with open(eval_result_file) as f:
    #                 for line in f.readlines():
    #                     if line.startswith('best_f1_thresh'):
    #                         na_prob_thresh = float(line.strip().split()[-1])
    #                         logger.info("na_prob_thresh = %.6f" % na_prob_thresh)
    #
    #     result, preds, _ = \
    #         evaluate(args, model, device, eval_dataset,
    #                  eval_dataloader, eval_examples, eval_features,
    #                  na_prob_thresh=na_prob_thresh,
    #                  pred_only=args.eval_test)
    #     with open(os.path.join(args.save_dir, "predictions.json"), "w") as writer:
    #         writer.write(json.dumps(preds, indent=4) + "\n")