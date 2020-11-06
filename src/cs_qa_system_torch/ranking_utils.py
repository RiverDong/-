import json
from json import JSONDecodeError
import pandas as pd

import torch
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

import constants

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
    collection_ir = {int(k): v2 for k, (v1,v2) in collection.items()}
    collection_ae = {int(k): v1 for k, (v1,v2) in collection.items()}
    collection_tuple = list(collection_ir.items())
    return collection_ir, collection_tuple,collection_ae

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer, BertModel),
    'distilbert': (DistilBertConfig, DistilBertTokenizer, DistilBertModel)
}

def predict(dataloader, model, device):
    prediction_list = []
    model.eval()
    for step, batch in enumerate(dataloader, start=1):
        input_batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            out = model(*input_batch)
            #out = torch.sigmoid(out)
            prediction_list.extend(out.squeeze().tolist())
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
            #out = torch.sigmoid(out)
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