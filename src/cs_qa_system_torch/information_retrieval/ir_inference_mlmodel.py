######## This file is still under construction #############################

import os
import argparse
import logging


import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer, \
    AutoConfig


from information_retrieval.ir_dataset import CombinedRankingDataset, RankingDataset
from information_retrieval.ir_model import BiEncoderModel, CrossEncoderModel
from information_retrieval.ir_transform import CombinedRankingTransform, RankingTransform
from information_retrieval.ir_utils import get_prediction, store_prediction_results
from utils import logging_config

logger = logging.getLogger(__name__)

class ScorePrediction:
    ml_model = None
    transform = None
    query_transform = None
    context_transform = None
    params_dir = '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase_finetuned/'
    do_lower_case = True
    architecture = 'cross'

    max_length_cross_architecture = 384
    max_query_length = 64
    max_passage_length = 256

    @classmethod
    def load_ir_model(cls):
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
            cls.transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=cls.max_length_cross_architecture, bool_np_array=True)
        else:
            cls.query_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_query_length, bool_np_array=True)
            cls.context_transform = RankingTransform(tokenizer=tokenizer, max_len=cls.max_passage_length, bool_np_array=True)
        cls.ml_model.load_state_dict(torch.load(os.path.join(cls.params_dir, 'pytorch_model.pt')))

    @classmethod
    def get_document_score(cls, query, passage):
        if cls.ml_model == None:
            cls.load_ir_model()
        if 'cross' in cls.architecture:
            input = cls.transform(query,passage)
        else:
            input = (*cls.query_transform(query), *cls.context_transform(passage))
        input = (torch.tensor(i.reshape(1,-1)) for i in input)
        return cls.ml_model(*input).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to predict the relevant documents for a '
                                                 'given query in Question Answering system')

    parser.add_argument('--output_dir', type=str, default='inference', help='path to inference output')

    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)
    parser.add_argument("--architecture", required=True, type=str, help='[bi, cross]')
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--max_length_cross_architecture", default=384, type=int)
    parser.add_argument("--max_passage_length", default=128, type=int, help='Not Required for CrossEncoder architecture.It uses 512 for query+passage')
    parser.add_argument("--max_query_length", default=64, type=int, help='Not Required for CrossEncoder architecture.It uses 512 for query+passage')
    parser.add_argument('--gpu', type=str, help='gpus to run model on')
    parser.add_argument("--test_data_path", default='data', type=str)
    parser.add_argument("--test_batch_size", default=2, type=int, help="Total batch size for eval.")
    parser.add_argument('--prediction_file', type=str, default='prediction.csv',
                        help='predictions file')



    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging_config(args.output_dir, name='inference')
    logger.info("The input args are: \n{}".format(args))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("The inference files will be saved to: {}".format(args.output_dir))


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
    if 'cross' in args.architecture:
        transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=args.max_length_cross_architecture)
    else:
        query_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_query_length)
        context_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_passage_length)

    if 'cross' in args.architecture:
        dataset = CombinedRankingDataset(args.test_data_path, transform)
    else:
        dataset = RankingDataset(args.test_data_path,
                                       query_transform, context_transform)
    dataloader = DataLoader(dataset,
                                batch_size=args.test_batch_size, shuffle=False,
                                collate_fn=dataset.batchify_join_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.pt')))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    predictions, actuals, qids = get_prediction(dataloader, model, device)
    store_prediction_results(args.test_data_path, predictions, actuals, qids,
                             os.path.join(args.output_dir, args.prediction_file))



