import json
import os
from json import JSONDecodeError
import pandas as pd

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import constants
from information_retrieval.ir_model import BiEncoderModel, CrossEncoderModel

def load_collection(collection_path):
    # Load the queries/passages collection
    try:
        with open(collection_path, 'r') as f:
            collection = json.load(f)
    except (JSONDecodeError, FileNotFoundError) as error:
        print('Error: please make sure the path is correct and the file is a json file')
        raise error
    collection = {int(k): v for k, v in collection.items()}
    collection_tuple = list(collection.items())
    return collection, collection_tuple

def load_passage_collection(collection_path):
    # Load the queries/passages collection
    try:
        with open(collection_path, 'r') as f:
            collection = json.load(f)
    except (JSONDecodeError, FileNotFoundError) as error:
        print('Error: please make sure the path is correct and the file is a json file')
        raise error
    collection_ir = {k: v2 for k, (v1, v2) in collection.items()}
    collection_ae = {k: v1 for k, (v1, v2) in collection.items()}
    collection_tuple = list(collection_ir.items())
    return collection_ir, collection_tuple, collection_ae

def predict(dataloader, model, device):
    prediction_list = []
    model.eval()
    for step, batch in enumerate(dataloader, start=1):
        input_batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            out = model(*input_batch)
            prediction_list.extend(out[0].squeeze().tolist())
    return prediction_list

def get_prediction(dataloader, model, device):
    prediction_list = []
    label_list = []
    qid_list = []
    model.eval()
    for step, batch in enumerate(dataloader, start=1):
        input_batch = tuple(t.to(device) for t in batch[:-2])
        labels_batch = batch[-2]
        qid_batch = batch[-1]
        with torch.no_grad():
            out = model(*input_batch)
            # out = torch.sigmoid(out)
            prediction_list.extend(out.squeeze().tolist())
            label_list.extend(labels_batch.squeeze().tolist())
            qid_list.extend(qid_batch)
    return prediction_list, label_list, qid_list

def store_prediction_results(input_file, predictions, actuals, qids, output_file, input_delimiter='\t'):
    data = pd.read_csv(input_file, sep=input_delimiter)
    data[constants.INFERENCE_PREDICTIONS] = predictions
    data[constants.INFERENCE_ACTUALS] = actuals
    data[constants.INFERECE_QUERY_IDS] = qids
    data.to_csv(output_file, sep=input_delimiter, index=False)


def get_ir_model_attributes(model_name_or_path,
                            query_model_name_or_path=None,
                            architecture='cross',
                            projection_dim=0,
                            do_lower_case=True):
    if architecture == 'bi':
        if os.path.exists(os.path.dirname(model_name_or_path)):
            query_config = AutoConfig.from_pretrained(
                os.path.join(model_name_or_path, constants.QUERY_PRE_MODEL_FOLDER))
            query_tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_name_or_path, constants.QUERY_PRE_MODEL_FOLDER),
                do_lower_case=do_lower_case
            )
            context_config = AutoConfig.from_pretrained(
                os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            context_tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER),
                do_lower_case=do_lower_case
            )
            query_pre_model = AutoModel.from_config(query_config)
            context_pre_model = AutoModel.from_config(context_config)
        else:
            query_config = AutoConfig.from_pretrained(
                query_model_name_or_path if query_model_name_or_path else model_name_or_path,
            )

            query_tokenizer = AutoTokenizer.from_pretrained(
                query_model_name_or_path if query_model_name_or_path else model_name_or_path,
                do_lower_case=do_lower_case
            )

            query_pre_model = AutoModel.from_pretrained(
                query_model_name_or_path if query_model_name_or_path else model_name_or_path,
                config=query_config,
            )

            context_config = AutoConfig.from_pretrained(model_name_or_path)
            context_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
            context_pre_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=context_config,
            )
        model = BiEncoderModel(query_config=query_config,
                               context_config=context_config,
                               query_model=query_pre_model,
                               context_model=context_pre_model,
                               projection_dim=projection_dim)
        return model, context_tokenizer, context_config, query_tokenizer, query_config
    elif architecture == 'cross':
        if os.path.exists(os.path.dirname(model_name_or_path)):
            config = AutoConfig.from_pretrained(os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER),
                do_lower_case=do_lower_case
            )
            pre_model = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
            pre_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
            )
        model = CrossEncoderModel(config, model=pre_model)
        return model, tokenizer, config, None, None
    elif architecture == 'cross-default':
        if os.path.exists(os.path.dirname(model_name_or_path)):
            config = AutoConfig.from_pretrained(os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER),
                do_lower_case=do_lower_case
            )
            pre_model = AutoModelForSequenceClassification.from_config(config)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
            pre_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=config,
            )
        model = CrossEncoderModel(config, model=pre_model)
        return model, tokenizer, config, None, None
    else:
        raise ValueError("Wrong architecture name")
