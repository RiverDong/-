import json
import os
from json import JSONDecodeError
from typing import List, Tuple

import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import constants
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from information_retrieval.ir_model import BiEncoderModel, CrossEncoderModel, get_encoding_vector
from information_retrieval.ir_loss import dot_product_scores
from information_retrieval.ir_dataset import SimpleDataset, CombinedInferenceDataset
from information_retrieval.ir_transform import CombinedRankingTransform, RankingTransform


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


def load_document_collection(collection_path: str):
    """
    :param collection_path: a json file containing dictionary of docid as key and value as tuple of
                            (preprocessed document for ir, preprocessed document for ae)
    :return: a tuple containing dictionary of ir documents, list of tuples containing (docid, document),
            dictionary of docid with value as index, dictionary of ae documents
    """
    try:
        with open(collection_path, 'r') as f:
            collection = json.load(f)
    except (JSONDecodeError, FileNotFoundError) as error:
        print('Error: please make sure the path is correct and the file is a json file')
        raise error
    collection_ir = {k: v2 for k, (v1, v2) in collection.items()}
    collection_ae = {k: v1 for k, (v1, v2) in collection.items()}
    idx_to_doc = list(collection_ir.items())
    doc_to_idx = {val[0]: index for index, val in enumerate(idx_to_doc)}
    return collection_ir, idx_to_doc, doc_to_idx, collection_ae


def load_ranking_model(model_name_or_path: str, idx_to_doc: List[Tuple[str, str]], device: torch,
                       inference_batch_size: int = 512):
    """
    :param model_name_or_path: path to trained ranking model. NOTE: This path should contain training_args.bin file containing all training arguments
    :param idx_to_doc: list of documents in the collection
    :param device: whether to load model to cpu or gpu based on cuda availability
    :param inference_batch_size: batch size to use to do prediction if number of documents is really large
    :return a tuple (<trained model object>, <rank type such as BM25, bi or coss>, <query transform function>,
                     <context transform function>, <context embeddings for all documents if biencoder>)
    """
    model = None
    rank_type = None
    query_transform = None
    context_transform = None
    context_embeddings = None

    result_dir = ''

    if model_name_or_path == constants.IR_BM25OKAPI:
        rank_type = 'BM25'
        word_tokenizer = WordTokenizerFactory.create_word_tokenizer(
            constants.IR_MODELS[constants.IR_BM25OKAPI]['word_tokenizer_name'])
        model = IRModelFactory.create_ir_model(ir_model_name=model_name_or_path,
                                               ir_model_path=result_dir,
                                               corpus=idx_to_doc,
                                               tokenizer=word_tokenizer)
    else:
        params = torch.load(os.path.join(model_name_or_path, 'training_args.bin'))
        rank_type = params.architecture
        model, tokenizer, config, query_tokenizer, query_config = get_ir_model_attributes(
            model_name_or_path=model_name_or_path,
            architecture=params.architecture,
            projection_dim=params.projection_dim if 'projection_dim' in vars(params).keys() else None,
            do_lower_case=params.do_lower_case)

        if os.path.exists(model_name_or_path):
            model.load_state_dict(torch.load(os.path.join(model_name_or_path, 'pytorch_model.pt')))

            model = nn.DataParallel(model)
        model.to(device)

        if params.architecture == 'cross':
            context_transform = CombinedRankingTransform(tokenizer=tokenizer,
                                                         max_len=params.max_query_passage_length if 'max_query_passage_length' in vars(
                                                             params).keys() else 384,
                                                         bool_np_array=True)
        elif params.architecture == 'bi':
            query_transform = RankingTransform(tokenizer=query_tokenizer, max_len=params.max_query_length,
                                               bool_np_array=True)
            context_transform = RankingTransform(tokenizer=tokenizer, max_len=params.max_passage_length,
                                                 bool_np_array=True)

            context_dataset = SimpleDataset(idx_to_doc, context_transform)
            context_dataloader = DataLoader(context_dataset, batch_size=inference_batch_size)

            context_embeddings = []
            model.eval()
            for step, batch in enumerate(context_dataloader, start=1):
                input_batch = tuple(t.to(device) for t in batch)
                with torch.no_grad():
                    #context_embeddings.append(model.module.context_model(*input_batch)[-1])
                    context_vec = get_encoding_vector(model.module.context_model,*input_batch)
                    if model.module.encode_document_proj is not None:
                        context_vec = model.module.encode_document_proj(context_vec)
                    context_embeddings.append(context_vec)
            context_embeddings = torch.cat(context_embeddings, dim=0)
    return model, rank_type, query_transform, context_transform, context_embeddings


def get_ranking_evaluation(qrels_path_or_data_frame: str, passage_collection_path: str, model_name_or_path: str,
                           inference_batch_size: int, top_n: int, rank_threshold_score: float, device: torch,
                           rerank: bool = False,
                           rerank_score_weight: float = 0.5):
    """
    :param qrels_path_or_data_frame: This could be a path to qrels files containing (queryid, query, relevant documentid) as json which can be read by pandas read json
                                     This could also be a dataframe containing (queryid, query, docid, document, score) that is used for predicting
                                     the ranking for the given documents in the dataframe (Usually used for reranking)
    :param passage_collection_path: path to collections json file containing dictionary of docid as key and value as tuple of
                            (preprocessed document for ir, preprocessed document for ae)
    :param model_name_or_path: path to the trained ranking model
    :param inference_batch_size: batch size to use to do prediction if number of documents is really large
    :param top_n: number of documents to return for each query
    :param device: whether to load model to cpu or gpu based on cuda availability
    :return a dataframe containing top n documents for each query and their respective ranking scores
    """
    collection_ir, idx_to_doc, doc_to_idx, _ = load_document_collection(passage_collection_path)
    collection_tuple = (collection_ir, idx_to_doc, doc_to_idx)

    model, rank_type, query_transform, context_transform, context_embeddings = load_ranking_model(model_name_or_path,
                                                                                                  idx_to_doc, device,
                                                                                                  inference_batch_size)

    rank_results_df = evaluate_ranking_model(qrels_path_or_data_frame, collection_tuple, rank_type, model, top_n,
                                        query_transform,
                                        context_transform, context_embeddings, inference_batch_size, device,
                                        rank_threshold_score, rerank=rerank, rerank_score_weight = rerank_score_weight)
    return rank_results_df


def evaluate_ranking_model(qrels_path_or_data_frame, document_collection_tuple, model_type, model, top_n,
                           query_transform,
                           context_transform, context_embeddings, inference_batch_size, device,
                           rank_threshold_score=0.0,
                           rerank=False, rerank_score_weight=0.5):
    """
    :param qrels_path_or_data_frame: This could be a path to qrels files containing (queryid, query, relevant documentid) as json which can be read by pandas read json
                                     This could also be a dataframe containing (queryid, query, docid, document, score) that is used for predicting
                                     the ranking for the given documents in the dataframe (Usually used for reranking)
    :param document_collection_tuple: A tuple containing (<document collection dictionary with key as docid and value as processed ir document>,
                                                          <list of documents as index to document>, <dictionary of document to index>)
    :param model_type: has values 'BM25', 'bi', 'cross'
    :param model: ranking model object
    :params top_n: top n documents to be returned for each query
    :param query_transform: query transform funcntion to use if its a biencoder architecture and needed to transform each query
    :param context_transform: context transform function to use to transform both query and document if its a cross encoder architecture
    :param context_embeddings: contextembeddings is a torch tensor matrix of size (number of documents x embedding size of each document)
    :param rank_batch_size: batch size to use to do prediction if number of documents is really large
    :param device: whether to load model to cpu or gpu based on cuda availability
    :param rerank: boolean parameter to set to true if the prediction is being done for reranking the ranking results
    :return a dataframe containing top n documents for each query and their respective ranking scores
    """
    document_collection_map, idx_to_doc, doc_to_idx = document_collection_tuple

    qrels = None
    label_list = None

    if rerank:
        if not isinstance(qrels_path_or_data_frame, pd.DataFrame):
            raise TypeError("Input should be a dataframe for reranking")
        list_query_tuple = list(qrels_path_or_data_frame[[constants.RANKING_INPUT_QUERY_ID,
                                                          constants.RANKING_INPUT_QUERY_NAME]].itertuples(index=False, name=None))
        list_query_doc_tuple = list(
            qrels_path_or_data_frame[[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                                      constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME,
                                      constants.RANKING_SCORE]].itertuples(index=False, name=None))
        label_list = qrels_path_or_data_frame[constants.RANKING_INPUT_LABEL_NAME].tolist()
    else:
        qrels = pd.read_json(qrels_path_or_data_frame, orient='columns', typ='frame')
        qrels = qrels.groupby([constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]).apply(
            lambda x: (list(set(x[constants.RANKING_INPUT_DOCUMENT_ID].astype(str))))).to_frame(
            name=constants.RANKING_INPUT_DOCUMENT_ID).reset_index()
        list_query_tuple = list(
            qrels[[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_QUERY_NAME]].to_records(index=False))
        list_query_doc_tuple = [(qid, pid, query, passage, 0.0) for (qid, query) in list_query_tuple for
                                (pid, passage) in idx_to_doc]

    if model_type == 'cross':
        df = result_crossencoder(list_query_doc_tuple, document_collection_tuple, model, context_transform,
                                 inference_batch_size,
                                 top_n, rank_threshold_score, device, qrels=qrels, rerank=rerank,
                                 rerank_score_weight=rerank_score_weight,
                                 label_list=label_list)

    elif model_type == 'BM25':
        df = result_bm25(list_query_tuple, document_collection_tuple, model, top_n, rank_threshold_score, qrels=qrels,
                         rerank=rerank)
    elif model_type == 'bi':
        df = result_biencoder(list_query_tuple, document_collection_tuple, model, query_transform, context_embeddings,
                              inference_batch_size, top_n, rank_threshold_score, device, qrels=qrels, rerank=rerank)
    else:
        raise NotImplementedError

    return df


def predict_ranking_model(list_or_data_frame, document_collection_tuple, model_type, model, top_n, query_transform,
                 context_transform, context_embeddings, inference_batch_size, rank_threshold_score, device,
                 rerank=False, rerank_score_weight=0.5):
    """
    :param qrels_path_or_data_frame: This could be a path to qrels files containing (queryid, query, relevant documentid) as json which can be read by pandas read json
                                     This could also be a dataframe containing (queryid, query, docid, document, score) that is used for predicting
                                     the ranking for the given documents in the dataframe (Usually used for reranking)
    :param document_collection_tuple: A tuple containing (<document collection dictionary with key as docid and value as processed ir document>,
                                                          <list of documents as index to document>, <dictionary of document to index>)
    :param model_type: has values 'BM25', 'bi', 'cross'
    :param model: ranking model object
    :params top_n: top n documents to be returned for each query
    :param query_transform: query transform funcntion to use if its a biencoder architecture and needed to transform each query
    :param context_transform: context transform function to use to transform both query and document if its a cross encoder architecture
    :param context_embeddings: contextembeddings is a torch tensor matrix of size (number of documents x embedding size of each document)
    :param rank_batch_size: batch size to use to do prediction if number of documents is really large
    :param device: whether to load model to cpu or gpu based on cuda availability
    :param rerank: boolean parameter to set to true if the prediction is being done for reranking the ranking results
    :return a dataframe containing top n documents for each query and their respective ranking scores
    """
    document_collection_map, idx_to_doc, doc_to_idx = document_collection_tuple
    if rerank:
        if not isinstance(list_or_data_frame, pd.DataFrame):
            raise TypeError("Input should be a dataframe for reranking")
        list_query_tuple = list(list_or_data_frame[[constants.RANKING_INPUT_QUERY_ID,
                                                    constants.RANKING_INPUT_QUERY_NAME]].itertuples(index=False, name=None))
        list_query_doc_tuple = list(
            list_or_data_frame[[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                                constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME,
                                constants.RANKING_SCORE]].itertuples(index=False, name=None))
    else:
        if isinstance(list_or_data_frame, str):
            list_query_tuple = list_or_data_frame
            list_query_doc_tuple = [(1, pid, list_or_data_frame, passage, 0.0) for (pid, passage) in idx_to_doc]
        elif isinstance(list_or_data_frame, list):
            list_query_tuple = list_or_data_frame
            list_query_doc_tuple = [(qid, pid, query, passage, 0.0) for (qid, query) in list_query_tuple for
                                    (pid, passage) in idx_to_doc]
        else:
            raise TypeError("input has to be a query string or a list of (queryid,query) tuples")

    if model_type == 'cross':
        df = result_crossencoder(list_query_doc_tuple, document_collection_tuple, model, context_transform,
                                 inference_batch_size,
                                 top_n, rank_threshold_score, device, qrels=None, rerank=rerank,
                                 rerank_score_weight=rerank_score_weight,
                                 label_list=None)

    elif model_type == 'BM25':
        df = result_bm25(list_query_tuple, document_collection_tuple, model, top_n, rank_threshold_score, qrels=None,
                         rerank=rerank)
    elif model_type == 'bi':
        df = result_biencoder(list_query_tuple, document_collection_tuple, model, query_transform, context_embeddings,
                              inference_batch_size, top_n, rank_threshold_score, device, qrels=None, rerank=rerank)
    else:
        raise NotImplementedError

    return df


def result_crossencoder(list_query_doc_tuple, document_collection_tuple, model, context_transform, inference_batch_size,
                        top_n, rank_threshold_score, device, qrels=None, rerank=False, rerank_score_weight=0.5,
                        label_list=None):
    collection_ir, idx_to_doc, doc_to_idx = document_collection_tuple

    prediction_score = predict_crossencoder(list_query_doc_tuple, model, context_transform, inference_batch_size,
                                            device)

    df = pd.DataFrame(list_query_doc_tuple,
                      columns=[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                               constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME,
                               constants.RANKING_SCORE])

    if rerank:
        df[constants.RERANKING_SCORE] = prediction_score
        if label_list is not None:
            df[constants.RANKING_INPUT_LABEL_NAME] = label_list
        df = df[df[constants.RERANKING_SCORE] > rank_threshold_score].reset_index(drop=True)
        df = df.groupby([constants.RANKING_INPUT_QUERY_ID]).apply(
            lambda x: compute_ranking_score(x, top_n, rerank_score_weight)).reset_index(drop=True)
        df = df.groupby([constants.RANKING_INPUT_QUERY_ID]).apply(lambda x:x.sort_values(by=[constants.WEIGHTED_RANKING_SCORE], ascending=False).head(top_n)).reset_index(drop=True)
    else:
        df[constants.RANKING_SCORE] = prediction_score
        if qrels is not None:
            label_list = []
            for i in qrels.index:
                all_pos_pid = qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]
                all_label = [0] * len(idx_to_doc)
                for i in all_pos_pid:
                    all_label[doc_to_idx[i]] = 1
                label_list.extend(all_label)
            df[constants.RANKING_INPUT_LABEL_NAME] = label_list
        df = df.groupby([constants.RANKING_INPUT_QUERY_ID]).apply(
            lambda x: x.sort_values([constants.RANKING_SCORE], ascending=False)).reset_index(drop=True)
        df = df[df[constants.RANKING_SCORE] > rank_threshold_score].reset_index(drop=True)
        df[constants.WEIGHTED_RANKING_SCORE] = df[constants.RANKING_SCORE]
        df = df.groupby([constants.RANKING_INPUT_QUERY_ID]).head(top_n).reset_index(drop=True)
    return df


def result_bm25(list_query_tuple, document_collection_tuple, model, top_n, rank_threshold_score, qrels=None,
                rerank=False):
    if rerank:
        raise NotImplementedError

    collection_ir, idx_to_doc, doc_to_idx = document_collection_tuple
    prediction = predict_bm25(list_query_tuple, model, top_n)
    if isinstance(list_query_tuple, str):
        # if the input is a single query string then convert to tuple to avoid additional conditions on the following logic
        list_query_tuple = [(1, list_query_tuple)]
    assert len(prediction) == len(list_query_tuple)

    pred_list = [(list_query_tuple[i][0], docid, list_query_tuple[i][1], collection_ir[docid], score) for
                 i, predict_query in enumerate(prediction) for
                 (docid, score) in predict_query]
    df = pd.DataFrame(pred_list, columns=[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                                          constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME,
                                          constants.RANKING_SCORE])

    if qrels is not None:
        assert len(prediction) == len(qrels)
        label_list = [int(docid in qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]) for i, predict_query in enumerate(prediction)
                      for (docid, score) in predict_query]
        df[constants.RANKING_INPUT_LABEL_NAME] = label_list

    df = df[df[constants.RANKING_SCORE] > rank_threshold_score].reset_index(drop=True)
    df[constants.WEIGHTED_RANKING_SCORE] = df[constants.RANKING_SCORE]
    return df


def result_biencoder(list_query_tuple, document_collection_tuple, model, query_transform, context_embeddings,
                     rank_batch_size, top_n, rank_threshold_score, device, qrels=None, rerank=False):
    if rerank:
        raise NotImplementedError

    collection_ir, idx_to_doc, doc_to_idx = document_collection_tuple
    prediction = predict_biencoder(list_query_tuple, model, query_transform, context_embeddings, rank_batch_size, top_n,
                                   device)

    if isinstance(list_query_tuple, str):
        # if the input is a single query string then convert to tuple to avoid
        # additional conditions on the following logic
        list_query_tuple = [(1, list_query_tuple)]
    assert len(prediction) == len(list_query_tuple)
    pred_list = [(list_query_tuple[i][0], idx_to_doc[index][0], list_query_tuple[i][1],
                  collection_ir[idx_to_doc[index][0]], score) for i, predict_query in enumerate(prediction) for
                 (index, score) in predict_query]
    df = pd.DataFrame(pred_list, columns=[constants.RANKING_INPUT_QUERY_ID, constants.RANKING_INPUT_DOCUMENT_ID,
                                          constants.RANKING_INPUT_QUERY_NAME, constants.RANKING_INPUT_DOCUMENT_NAME,
                                          constants.RANKING_SCORE])

    if qrels is not None:
        assert len(prediction) == len(qrels)
        label_list = [int(idx_to_doc[index][0] in qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]) for
                      i, predict_query in enumerate(prediction) for (index, score) in predict_query]
        df[constants.RANKING_INPUT_LABEL_NAME] = label_list

    df = df[df[constants.RANKING_SCORE] > rank_threshold_score].reset_index(drop=True)
    df[constants.WEIGHTED_RANKING_SCORE] = df[constants.RANKING_SCORE]
    return df


def predict_bm25(list_query_tuple, model, top_n):
    # For a given query, get pids of the sorted top n relevant passages and their scores using the BM25 model
    try:
        if isinstance(list_query_tuple, str):
            # if the input is a single query with string
            query = list_query_tuple
            return [list(zip(*model.get_top_n_single(query, top_n)))]
        else:
            return [list(zip(*model.get_top_n_single(query, top_n))) for (id, query) in list_query_tuple]
    except (AttributeError, TypeError) as error:
        print('Error: please make sure the IR model has a "get_top_n_single" method with desired signature')
        raise error


def predict_crossencoder(list_query_doc_tuple, model, context_transform, inference_batch_size, device):
    dataset = CombinedInferenceDataset(list_query_doc_tuple, context_transform)
    dataloader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False)
    prediction_score_list = []
    model.eval()
    for step, batch in enumerate(dataloader, start=1):
        input_batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            out = model(*input_batch)
            prediction_score_list.extend(out[0].squeeze().tolist())
    return prediction_score_list


def predict_biencoder(list_query_tuple, model, query_transform, context_embeddings, inference_batch_size, top_n,
                      device):
    query_embeddings = []
    if isinstance(list_query_tuple, str):
        # if the input is a single query with string
        query = list_query_tuple
        query_input = (torch.tensor(t.reshape(1, -1), dtype=torch.long).to(device) for t in query_transform(query))
        model.eval()
        with torch.no_grad():
            #query_embeddings = model.module.query_model(*query_input)[-1]
            query_embeddings = get_encoding_vector(model.module.query_model, *query_input)
            if model.module.encode_query_proj is not None:
                query_embeddings = model.module.encode_query_proj(query_embeddings)
    else:
        # if the input is a list of query tuples with (queryid, query)
        query_dataset = SimpleDataset(list_query_tuple, query_transform)
        query_dataloader = DataLoader(query_dataset, batch_size=inference_batch_size, shuffle=False)
        model.eval()
        for step, batch in enumerate(query_dataloader, start=1):
            input_batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                # query_embeddings.append(model.module.query_model(*input_batch)[-1])
                query_vec = get_encoding_vector(model.module.query_model, *input_batch)
                if model.module.encode_query_proj is not None:
                    query_vec = model.module.encode_query_proj(query_vec)
                query_embeddings.append(query_vec)
        query_embeddings = torch.cat(query_embeddings, dim=0)

    scores = dot_product_scores(query_embeddings, context_embeddings)
    sotmax_scores = F.softmax(scores, dim=1)
    output = torch.sort(sotmax_scores, dim=1, descending=True)
    values = output[0][:, 0:top_n].tolist()
    indices = output[1][:, 0:top_n].tolist()
    prediction = [list(zip(i[0], i[1])) for i in zip(indices, values)]
    return prediction


def compute_ranking_score(x: pd.DataFrame, top_n: int, rerank_weight: float = 0.0):
    ir_score_sum = x[constants.RANKING_SCORE].sum()
    if ir_score_sum == 0:
        if constants.RERANKING_SCORE not in x.columns:
            x[constants.WEIGHTED_RANKING_SCORE] = 0
        else:
            x[constants.WEIGHTED_RANKING_SCORE] = rerank_weight * x[constants.RERANKING_SCORE]
    else:
        if constants.RERANKING_SCORE not in x.columns:
            x[constants.WEIGHTED_RANKING_SCORE] = (x[constants.RANKING_SCORE] / ir_score_sum)
        else:
            x[constants.WEIGHTED_RANKING_SCORE] = rerank_weight * x[constants.RERANKING_SCORE] + (1 - rerank_weight) * (
                    x[constants.RANKING_SCORE] / ir_score_sum)
    # x = x.sort_values(constants.WEIGHTED_RANKING_SCORE, ascending=False).head(top_n)
    return x
