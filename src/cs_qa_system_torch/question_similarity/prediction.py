from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import pandas as pd
import numpy as np
import time

from ir_inference import load_collection
from ranking_metrics import map_mrr_mcoverage

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model_save_path = 'output/training_sbert_bert-base-nli-mean-tokens-2020-07-30_06-57-07'
model = SentenceTransformer(model_save_path)
# amazon_reader = AmazonDataReader('/data/QAData/MSAmazonPreProcessData')
# test_data = SentencesDataset(examples=amazon_reader.get_examples("test"), model=model)
# test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
# evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
#
# model.evaluate(evaluator)

def get_passage_embeddings(model,collection):
    keys = collection.keys()
    embeddings = model.encode(collection.values())
    return {k:v for (k,v)  in zip(keys,embeddings)}
amazon_valid = pd.read_json('/data/QAData/AmazonPreProcessRankTestAndPassageCollection-ToCompareWithBM25/rank_test-500-4.0-(445,np.inf)-original-(1,np.inf)-preprocessed.json', orient='columns', typ='frame')
passage_collection, _ = load_collection('/data/QAData/AmazonPreProcessRankTestAndPassageCollection-ToCompareWithBM25/passage_collection-500-4.0-(445,np.inf)-original-(1,np.inf)-preprocessed.json')
tic = time.time()
query_list = model.encode(amazon_valid['query'].tolist())
query_embeddings = np.stack(query_list,axis=0)
keys = list(passage_collection.keys())
collection_list = model.encode(passage_collection.values())
collection_embeddings = np.stack(collection_list,axis=0)
scores = np.matmul(query_embeddings,collection_embeddings.T)
out = np.argsort(scores)[:,::-1][:,:500].tolist()
final_passages = [[keys[i] for i in row] for row in out]
pred_dict = dict(zip(amazon_valid['qid'].tolist(),final_passages))
map_metric, mrr_metric, mcoverage_metric = map_mrr_mcoverage(amazon_valid, pred_dict, 500)
print('Retrieved top-{0:d} passages (from a pool of {1:d} passages) for {2:d} queries by {3:}: MAP@{7:d} = '
      '{4:.4f}, MRR@{7:d} = {5:.4f}, MCoverage@{7:d} = {6:.4f}'.format(500, len(passage_collection),
                                                                       len(pred_dict), 'S-BERT', map_metric,
                                                                       mrr_metric, mcoverage_metric, 500))

