import os

import torch

import constants
from information_retrieval.ir_utils import load_document_collection, load_ranking_model, predict_ranking_model


class RankPrediction:

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ## Collection path
    ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/model'
    passage_collection_path = os.path.join(ROOT_QASYSTEM_ARTIFACTS, 'production_collection.json')

    ## ranking parameters
    rank_model_name_or_path = 'BM25Okapi'
    rank_top_n = 100
    rank_inference_batch_size = 512
    rank_model_score_threshold = 0.0

    ## reranking parameters
    do_rerank = False
    rerank_model_name_or_path = os.path.join(ROOT_QASYSTEM_ARTIFACTS, 'ir_artifacts/rerank_crossencoder_bert/')
    rerank_top_n = 2
    rerank_inference_batch_size = 512
    rerank_model_score_threshold = 0.8
    rerank_weight_for_mixed_score = 0.50

    ## Initializing rank parameters
    collection_ir = None
    idx_to_doc = None
    doc_to_idx = None

    rank_model = None
    rank_type = None
    rank_query_transform = None
    rank_context_transform = None
    rank_context_embeddings = None

    rerank_model = None
    rerank_type = None
    rerank_query_transform = None
    rerank_context_transform = None
    rerank_context_embeddings = None


    @classmethod
    def load_data(cls):
        cls.collection_ir, cls.idx_to_doc, cls.doc_to_idx, _ = load_document_collection(cls.passage_collection_path)

    @classmethod
    def load_ir_models(cls):
        cls.rank_model, cls.rank_type, \
        cls.rank_query_transform, cls.rank_context_transform, \
        cls.rank_context_embeddings = load_ranking_model(cls.rank_model_name_or_path, cls.idx_to_doc, cls.device,
                                                 cls.rank_inference_batch_size)
        if cls.do_rerank:
            cls.rerank_model, cls.rerank_type, \
            cls.rerank_query_transform, cls.rerank_context_transform, \
            cls.rerank_context_embeddings = load_ranking_model(cls.rerank_model_name_or_path, cls.idx_to_doc, cls.device,
                                                             cls.rerank_inference_batch_size)

    @classmethod
    def get_documents(cls, query):
        if cls.rank_model is None:
            cls.load_data()
            cls.load_ir_models()
        collection_tuple = (cls.collection_ir, cls.idx_to_doc, cls.doc_to_idx)

        df = predict_ranking_model(query, collection_tuple, cls.rank_type,cls.rank_model, cls.rank_top_n, cls.rank_query_transform,
                            cls.rank_context_transform, cls.rank_context_embeddings, cls.rank_inference_batch_size,
                                   cls.rank_model_score_threshold, cls.device, rerank=False)

        if len(df) ==0:
            return df

        if cls.do_rerank:
            df = predict_ranking_model(df, collection_tuple, cls.rerank_type, cls.rerank_model, cls.rerank_top_n,
                                     cls.rerank_query_transform,
                                     cls.rerank_context_transform, cls.rerank_context_embeddings,
                                     cls.rerank_inference_batch_size, cls.rerank_model_score_threshold, cls.device,
                                     rerank=True, rerank_score_weight=cls.rerank_weight_for_mixed_score)

        return df