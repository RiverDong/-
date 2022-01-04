import json
import os
from json import JSONDecodeError
from typing import List, Tuple
import logging
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import constants
from factory.ir_model_factory import IRModelFactory
from factory.word_tokenizer_factory import WordTokenizerFactory
from information_retrieval.ir_model import BiEncoderModel, CrossEncoderModel, get_encoding_vector, SingleEncoderModel
from information_retrieval.ir_loss import dot_product_scores, BiEncoderNllLoss, BiEncoderBCELoss, TripletLoss
from information_retrieval.ir_transform import CombinedRankingTransform, RankingTransform
from information_retrieval.ir_datasets import TupleQueryOrContextDataset, TupleQueryAndContextDataset

logger = logging.getLogger(__name__)


def get_ir_model_attributes(model_name_or_path,
                            query_model_name_or_path=None,
                            architecture='cross',
                            projection_dim=0,
                            device=None):
    if 'bi' in architecture:
        if os.path.exists(os.path.dirname(model_name_or_path)):
            query_config = AutoConfig.from_pretrained(
                os.path.join(model_name_or_path, constants.QUERY_PRE_MODEL_FOLDER))
            query_tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_name_or_path, constants.QUERY_PRE_MODEL_FOLDER),
            )
            context_config = AutoConfig.from_pretrained(
                os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            context_tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER),
            )
            query_pre_model = AutoModel.from_config(query_config)
            context_pre_model = AutoModel.from_config(context_config)
        else:
            query_config = AutoConfig.from_pretrained(
                query_model_name_or_path if query_model_name_or_path else model_name_or_path,
            )

            query_tokenizer = AutoTokenizer.from_pretrained(
                query_model_name_or_path if query_model_name_or_path else model_name_or_path,
            )

            query_pre_model = AutoModel.from_pretrained(
                query_model_name_or_path if query_model_name_or_path else model_name_or_path,
                config=query_config,
            )

            context_config = AutoConfig.from_pretrained(model_name_or_path)
            context_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            context_pre_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=context_config,
            )
        model = BiEncoderModel(query_config=query_config,
                               context_config=context_config,
                               query_model=query_pre_model,
                               context_model=context_pre_model,
                               projection_dim=projection_dim)
        if device.type == 'cuda':
            model = nn.DataParallel(model)
            model.to(device)
        return model, context_tokenizer, context_config, query_tokenizer, query_config
    elif 'cross' in architecture:
        if os.path.exists(os.path.dirname(model_name_or_path)):
            config = AutoConfig.from_pretrained(os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            pre_model = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            pre_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
            )
        model = CrossEncoderModel(config, model=pre_model)
        if device.type == 'cuda':
            model = nn.DataParallel(model)
            model.to(device)
        return model, tokenizer, config, None, None
    elif 'single' in architecture:
        if os.path.exists(os.path.dirname(model_name_or_path)):
            config = AutoConfig.from_pretrained(os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_name_or_path, constants.PRE_MODEL_FOLDER))
            pre_model = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            pre_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
            )
        model = SingleEncoderModel(config, model=pre_model, projection_dim=projection_dim)
        if device.type == 'cuda':
            model = nn.DataParallel(model)
            model.to(device)
        return model, tokenizer, config, tokenizer, config
    else:
        raise ValueError("Wrong architecture name")


def get_ir_data_transform(architecture: str, tokenizer, query_tokenizer=None, max_query_passage_length=384,
                          max_query_length=32,
                          max_passage_length=384):
    query_transform = None
    transform = None
    if 'cross' in architecture:
        transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=max_query_passage_length)
    elif 'bi' in architecture:
        query_transform = RankingTransform(tokenizer=query_tokenizer, max_len=max_query_length)
        transform = RankingTransform(tokenizer=tokenizer, max_len=max_passage_length)
    elif 'single' in architecture:
        query_transform = RankingTransform(tokenizer=tokenizer, max_len=max(max_query_length, max_passage_length))
        transform = query_transform
    else:
        raise ValueError("Wrong architecture name")
    return transform, query_transform


def get_loss_function(loss_name, architecture):
    loss_function = nn.BCELoss()
    if loss_name == 'BCE':
        if 'cross' in architecture:
            loss_function = nn.BCELoss()
        elif 'bi' in architecture or 'single' in architecture:
            loss_function = BiEncoderBCELoss()
    elif loss_name == 'WHNS':
        loss_function = BiEncoderNllLoss()
    elif loss_name == 'Triplet':
        loss_function = TripletLoss()
    return loss_function

def get_context_embeddings(idx_to_doc, context_transform, model, architecture, batch_size, device,
                           context_embeddings_path=None, overwrite=False):
    context_embeddings = None
    context_embeddings_temp = None
    context_proj = None

    if 'single' in architecture or 'bi' in architecture:

        if context_embeddings_path is not None and os.path.exists(
                os.path.join(context_embeddings_path, 'context_embeddings.pt')) and not overwrite:
            logger.info('*************Loading context embeddings*****************\n')
            context_embeddings = torch.load(os.path.join(context_embeddings_path, 'context_embeddings.pt'))
            return context_embeddings

        logger.info('*************Computing context embeddings*****************\n')
        context_dataset = TupleQueryOrContextDataset(idx_to_doc, context_transform)
        context_dataloader = DataLoader(context_dataset, collate_fn=context_dataset.batchify, batch_size=batch_size)
        if device.type == 'cuda':
            context_model = model.module.context_model if 'bi' in architecture else model.module.model
            context_model = nn.DataParallel(context_model)
            context_model.to(device)
            context_proj = model.module.encode_document_proj if 'bi' in architecture else model.module.encode_proj
            if context_proj is not None:
                context_proj = nn.DataParallel(context_proj)
                context_proj.to(device)
        else:
            context_model = model.context_model if 'bi' in architecture else model.model
            context_proj = model.encode_document_proj if 'bi' in architecture else model.encode_proj

        with tqdm(total=len(context_dataloader)) as pbar:
            for step, batch in enumerate(context_dataloader, start=1):
                pbar.update(1)
                input_batch = tuple(t.to(device) for t in batch)
                context_model.eval()
                with torch.no_grad():
                    context_vec = get_encoding_vector(context_model, *input_batch)
                    if context_proj is not None:
                        context_proj.eval()
                        context_vec = context_proj(context_vec)
                    if context_embeddings_temp is None:
                        context_embeddings_temp = context_vec
                    else:
                        context_embeddings_temp = torch.cat((context_embeddings_temp, context_vec))
                    # if tensor is greater than 500MB   then copy to cpus
                    if (context_embeddings_temp.element_size() * context_embeddings_temp.nelement() / 1024 ** 3) > 0.1:
                        if context_embeddings is None:
                            context_embeddings = context_embeddings_temp.detach().cpu()
                        else:
                            context_embeddings = torch.cat((context_embeddings, context_embeddings_temp.detach().cpu()))
                        context_embeddings_temp = None
                        torch.cuda.empty_cache()
            if context_embeddings_temp is not None:
                if context_embeddings is None:
                    context_embeddings = context_embeddings_temp.detach().cpu()
                else:
                    context_embeddings = torch.cat((context_embeddings, context_embeddings_temp.detach().cpu()))
                context_embeddings_temp = None
                torch.cuda.empty_cache()

            if context_embeddings_path is not None and overwrite:
                if not os.path.exists(context_embeddings_path):
                    os.makedirs(context_embeddings_path)

                torch.save(context_embeddings, os.path.join(context_embeddings_path, 'context_embeddings.pt'))

    return context_embeddings


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
                       inference_batch_size: int = 512, model_architecture=None, max_query_passage_length=384,
                       max_query_length=32, max_passage_length=384, context_embeddings_path=None, embeddings_overwrite=False):
    """
    :param model_name_or_path: path to trained ranking model. NOTE: This path should contain training_args.bin file containing all training arguments
    :param idx_to_doc: list of documents in the collection
    :param device: whether to load model to cpu or gpu based on cuda availability
    :param inference_batch_size: batch size to use to do prediction if number of documents is really large
    :return a tuple (<trained model object>, <rank type such as BM25, bi or coss>, <query transform function>,
                     <context transform function>, <context embeddings for all documents if biencoder>)
    """
    model = None
    architecture = None
    query_transform = None
    transform = None
    context_embeddings = None

    result_dir = ''

    if model_name_or_path == constants.IR_BM25OKAPI:
        architecture = 'BM25'
        word_tokenizer = WordTokenizerFactory.create_word_tokenizer(
            constants.IR_MODELS[constants.IR_BM25OKAPI]['word_tokenizer_name'])
        model = IRModelFactory.create_ir_model(ir_model_name=model_name_or_path,
                                               ir_model_path=result_dir,
                                               corpus=idx_to_doc,
                                               tokenizer=word_tokenizer)
    elif os.path.exists(model_name_or_path):
        params = torch.load(os.path.join(model_name_or_path, 'training_args.bin'))
        architecture = params.architecture
        model, tokenizer, config, query_tokenizer, query_config = get_ir_model_attributes(
            model_name_or_path=model_name_or_path,
            architecture=params.architecture,
            projection_dim=params.projection_dim if 'projection_dim' in vars(params).keys() else None,
            device=device)
        if hasattr(model, "module"):
            model.module.load_state_dict(
                torch.load(os.path.join(model_name_or_path, 'pytorch_model.pt'), map_location=device))
        else:
            model.load_state_dict(torch.load(os.path.join(model_name_or_path, 'pytorch_model.pt'), map_location=device))

        transform, query_transform = get_ir_data_transform(params.architecture, tokenizer, query_tokenizer,
                                                           params.max_query_passage_length if 'max_query_passage_length' in vars(
                                                               params).keys() else 384,
                                                           params.max_query_length,
                                                           params.max_passage_length)
    elif not os.path.exists(model_name_or_path) and model_architecture is not None:
        architecture = model_architecture
        model, tokenizer, config, query_tokenizer, query_config = get_ir_model_attributes(
            model_name_or_path=model_name_or_path,
            architecture=model_architecture,
            device=device)
        transform, query_transform = get_ir_data_transform(model_architecture, tokenizer, query_tokenizer,
                                                           max_query_passage_length,
                                                           max_query_length,
                                                           max_passage_length)

    context_embeddings = get_context_embeddings(idx_to_doc, transform, model, architecture, inference_batch_size,
                                                device, context_embeddings_path, embeddings_overwrite)

    return model, architecture, query_transform, transform, context_embeddings


def get_ranking_evaluation(qrels_path_or_data_frame: str, passage_collection_path: str, model_name_or_path: str,
                           inference_batch_size: int, top_n: int, rank_threshold_score: float, device: torch,
                           rerank: bool = False,
                           rerank_score_weight: float = 0.5, model_architecture=None,
                           max_query_passage_length=384, max_query_length=32, max_passage_length=384,
                           context_embeddings_path=None,
                           overwrite_context_embeddings=False):
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

    model, architecture, query_transform, context_transform, context_embeddings = load_ranking_model(
        model_name_or_path=model_name_or_path,
        idx_to_doc=idx_to_doc,
        device=device,
        inference_batch_size=inference_batch_size,
        model_architecture=model_architecture,
        max_query_passage_length=max_query_passage_length,
        max_query_length=max_query_length,
        max_passage_length=max_passage_length,
        context_embeddings_path=context_embeddings_path,
        embeddings_overwrite=overwrite_context_embeddings)

    rank_results_df = evaluate_ranking_model(qrels_path_or_data_frame, collection_tuple, architecture, model, top_n,
                                             query_transform,
                                             context_transform, context_embeddings, inference_batch_size, device,
                                             rank_threshold_score, rerank=rerank,
                                             rerank_score_weight=rerank_score_weight)
    return rank_results_df


def evaluate_ranking_model(qrels_path_or_data_frame, document_collection_tuple, architecture, model, top_n,
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
    :param architecture: has values 'BM25', 'bi', 'cross'
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
                                                          constants.RANKING_INPUT_QUERY_NAME]].itertuples(index=False,
                                                                                                          name=None))
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
        if 'cross' in architecture:
            list_query_doc_tuple = [(qid, pid, query, passage, 0.0) for (qid, query) in list_query_tuple for
                                    (pid, passage) in idx_to_doc]
    if 'cross' in architecture:
        df = result_crossencoder(list_query_doc_tuple, document_collection_tuple, model, context_transform,
                                 inference_batch_size,
                                 top_n, rank_threshold_score, device, qrels=qrels, rerank=rerank,
                                 rerank_score_weight=rerank_score_weight,
                                 label_list=label_list)

    elif 'BM25' in architecture:
        df = result_bm25(list_query_tuple, document_collection_tuple, model, top_n, rank_threshold_score, qrels=qrels,
                         rerank=rerank)
    elif 'bi' in architecture or 'single' in architecture:
        df = result_biencoder(list_query_tuple, document_collection_tuple, model, architecture, query_transform, context_embeddings,
                              inference_batch_size, top_n, rank_threshold_score, device, qrels=qrels, rerank=rerank)
    else:
        raise NotImplementedError

    return df


def predict_ranking_model(list_or_data_frame, document_collection_tuple, architecture, model, top_n, query_transform,
                          context_transform, context_embeddings, inference_batch_size, rank_threshold_score, device,
                          rerank=False, rerank_score_weight=0.5):
    """
    :param qrels_path_or_data_frame: This could be a path to qrels files containing (queryid, query, relevant documentid) as json which can be read by pandas read json
                                     This could also be a dataframe containing (queryid, query, docid, document, score) that is used for predicting
                                     the ranking for the given documents in the dataframe (Usually used for reranking)
    :param document_collection_tuple: A tuple containing (<document collection dictionary with key as docid and value as processed ir document>,
                                                          <list of documents as index to document>, <dictionary of document to index>)
    :param architecture: has values 'BM25', 'bi', 'cross'
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
                                                    constants.RANKING_INPUT_QUERY_NAME]].itertuples(index=False,
                                                                                                    name=None))
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

    if 'cross' in architecture:
        df = result_crossencoder(list_query_doc_tuple, document_collection_tuple, model, context_transform,
                                 inference_batch_size,
                                 top_n, rank_threshold_score, device, qrels=None, rerank=rerank,
                                 rerank_score_weight=rerank_score_weight,
                                 label_list=None)

    elif 'BM25' in architecture:
        df = result_bm25(list_query_tuple, document_collection_tuple, model, top_n, rank_threshold_score, qrels=None,
                         rerank=rerank)
    elif 'bi' in architecture or 'single' in architecture:
        df = result_biencoder(list_query_tuple, document_collection_tuple, model, architecture, query_transform, context_embeddings,
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
        df = df.groupby([constants.RANKING_INPUT_QUERY_ID]).apply(
            lambda x: x.sort_values(by=[constants.WEIGHTED_RANKING_SCORE], ascending=False).head(top_n)).reset_index(
            drop=True)
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
        label_list = [int(docid in qrels.loc[i, constants.RANKING_INPUT_DOCUMENT_ID]) for i, predict_query in
                      enumerate(prediction)
                      for (docid, score) in predict_query]
        df[constants.RANKING_INPUT_LABEL_NAME] = label_list

    df = df[df[constants.RANKING_SCORE] > rank_threshold_score].reset_index(drop=True)
    df[constants.WEIGHTED_RANKING_SCORE] = df[constants.RANKING_SCORE]
    return df


def result_biencoder(list_query_tuple, document_collection_tuple, model, architecture, query_transform, context_embeddings,
                     rank_batch_size, top_n, rank_threshold_score, device, qrels=None, rerank=False):
    if rerank:
        raise NotImplementedError

    collection_ir, idx_to_doc, doc_to_idx = document_collection_tuple
    prediction = predict_biencoder(list_query_tuple, model, architecture, query_transform, context_embeddings, rank_batch_size, top_n,
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
    dataset = TupleQueryAndContextDataset(list_query_doc_tuple, context_transform)
    dataloader = DataLoader(dataset, collate_fn=dataset.batchify, batch_size=inference_batch_size, shuffle=False)
    prediction_score_list = []
    model.eval()
    with tqdm(total=len(dataloader)) as bar:
        for step, batch in enumerate(dataloader, start=1):
            bar.update(1)
            input_batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                out = model(*input_batch)
                prediction_score_list.extend(out[0].squeeze().tolist())
    return prediction_score_list


def predict_biencoder(list_query_tuple, model, architecture, query_transform, context_embeddings, inference_batch_size, top_n,
                      device):
    query_embeddings = None
    query_embeddings_temp = None
    query_proj = None
    if device.type == 'cuda':
        query_model = model.module.query_model if 'bi' in architecture else model.module.model
        query_model = nn.DataParallel(query_model)
        query_model.to(device)
        query_proj = model.module.encode_query_proj if 'bi' in architecture else model.module.encode_proj
        if query_proj is not None:
            query_proj = nn.DataParallel(query_proj)
            query_proj.to(device)
    else:
        query_model = model.query_model if 'bi' in architecture else model.module.model
        query_proj = model.encode_query_proj if 'bi' in architecture else model.encode_proj
    if isinstance(list_query_tuple, str):
        # if the input is a single query with string
        query = list_query_tuple
        query_input = (torch.tensor(t.reshape(1, -1), dtype=torch.long).to(device) for t in query_transform(query))
        query_model.eval()
        with torch.no_grad():
            query_embeddings = get_encoding_vector(query_model, *query_input)
            if query_proj is not None:
                query_proj.eval()
                query_embeddings = query_proj(query_embeddings)
    else:
        # if the input is a list of query tuples with (queryid, query)
        query_dataset = TupleQueryOrContextDataset(list_query_tuple, query_transform)
        query_dataloader = DataLoader(query_dataset, collate_fn=query_dataset.batchify, batch_size=inference_batch_size, shuffle=False)
        logger.info("********Computing Query Embeddings*******************")
        with tqdm(total=len(query_dataloader)) as pbar:
            for step, batch in enumerate(query_dataloader, start=1):
                pbar.update(1)
                input_batch = tuple(t.to(device) for t in batch)
                query_model.eval()
                with torch.no_grad():
                    query_vec = get_encoding_vector(query_model, *input_batch)
                    if query_proj is not None:
                        query_proj.eval()
                        query_vec = query_proj(query_vec)
                    if query_embeddings_temp is None:
                        query_embeddings_temp = query_vec
                    else:
                        query_embeddings_temp = torch.cat((query_embeddings_temp, query_vec))
                    # if tensor is greater than 3GB then copy to cpu
                    if (query_embeddings_temp.element_size() * query_embeddings_temp.nelement() / 1024 ** 3) > 0.5:
                        if query_embeddings is None:
                            query_embeddings = query_embeddings_temp.detach().cpu()
                        else:
                            query_embeddings = torch.cat((query_embeddings, query_embeddings.detach().cpu()))
                        query_embeddings_temp = None
                        torch.cuda.empty_cache()

            if query_embeddings_temp is not None:
                if query_embeddings is None:
                    query_embeddings = query_embeddings_temp.detach().cpu()
                else:
                    query_embeddings = torch.cat((query_embeddings, query_embeddings_temp.detach().cpu()))
                query_embeddings_temp = None
                torch.cuda.empty_cache()
    logger.info('Computing scores')
    prediction = []
    prev_query = 0
    jump_query = 40
    for q in tqdm(range(jump_query, len(query_embeddings) + jump_query, jump_query)):
        if q == (len(query_embeddings) // jump_query + 1) * jump_query:
            curr_query = len(query_embeddings)
        else:
            curr_query = q
        prev_context = 0
        jump_context = 100000
        score_vec = None
        torch.cuda.empty_cache()
        for c in range(jump_context, len(context_embeddings) + jump_context, jump_context):
            if c == (len(context_embeddings) // jump_context + 1) * jump_context:
                curr_context = len(context_embeddings)
            else:
                curr_context = c
            scores = dot_product_scores(query_embeddings[prev_query:curr_query, :].to(device),
                                        context_embeddings[prev_context:curr_context, :].to(device))
            if score_vec is None:
                score_vec = scores
            else:
                score_vec = torch.cat((score_vec, scores), dim=1)
            scores = None
            torch.cuda.empty_cache()
            prev_context = curr_context
        scores = None
        torch.cuda.empty_cache()
        softmax_score = F.softmax(score_vec, dim=1)
        score_vec = None
        torch.cuda.empty_cache()
        output = torch.sort(softmax_score, dim=1, descending=True)
        softmax_score = None
        torch.cuda.empty_cache()
        values = output[0][:, 0:top_n].tolist()
        indices = output[1][:, 0:top_n].tolist()
        prediction.extend([list(zip(i[0], i[1])) for i in zip(indices, values)])
        output = None
        torch.cuda.empty_cache()
        prev_query = curr_query

    # if len(context_embeddings) < 500000:
    #     logger.info('Computing scores')
    #     scores = dot_product_scores(query_embeddings, context_embeddings)
    #     sotmax_scores = F.softmax(scores, dim=1)
    #     output = torch.sort(sotmax_scores, dim=1, descending=True)
    #     values = output[0][:, 0:top_n].tolist()
    #     indices = output[1][:, 0:top_n].tolist()
    #     prediction = [list(zip(i[0], i[1])) for i in zip(indices, values)]
    # else:
    #     prediction = []
    #     prev = 0
    #     jump = 500
    #     logger.info('Computing scores by looping group of 500 query (context_embedding is large)')
    #     for i in tqdm(range(jump,len(query_embeddings),jump)):
    #         curr = i
    #         scores = dot_product_scores(query_embeddings[prev:curr,:], context_embeddings)
    #         sotmax_scores = F.softmax(scores, dim=1)
    #         output = torch.sort(sotmax_scores, dim=1, descending=True)
    #         values = output[0][:, 0:top_n].tolist()
    #         indices = output[1][:, 0:top_n].tolist()
    #         prediction.extend([list(zip(i[0], i[1])) for i in zip(indices, values)])
    #         prev = curr
    #     scores = dot_product_scores(query_embeddings[curr:len(query_embeddings),:], context_embeddings)
    #     sotmax_scores = F.softmax(scores, dim=1)
    #     output = torch.sort(sotmax_scores, dim=1, descending=True)
    #     values = output[0][:, 0:top_n].tolist()
    #     indices = output[1][:, 0:top_n].tolist()
    #     prediction.extend([list(zip(i[0], i[1])) for i in zip(indices, values)])
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
