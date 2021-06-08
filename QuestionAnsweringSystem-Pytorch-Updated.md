# QA System

## Question Similarity Model (QSM)

### S-Bert Train


```python
#train model
%%run -i /Users/yqinamz/Desktop/QA_BOT/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/train_QSM_Sbert.py \
--pretrain_model_path '/home/yqinamz/output/quora_sts-bert-base-nli-mean-tokens-2020-08-17_20-53-14/' \
--model_name '/home/yqinamz/output/Sbert_test' \
--train_data_path '/home/yqinamz/QA_Bot/QA_EXP/EXP_1002/Mix_2USE_8R_1009/' 



```

### Inference on Question-Pairs


```python
%%run -i /Users/yqinamz/Desktop/QA_BOT/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/QSM_inference.py \
--saved_model_path '/home/yqinamz/output/quora_sts-bert-base-nli-mean-tokens-2020-08-17_20-53-14/' \
--test_data_path '/home/yqinamz/QA_Bot/QA_EXP/EXP_1002/Mix_2USE_8R_1009/test_sample.tsv'

```

### Real-Time Prediction for QSM


```python
import sys
import json
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
from question_similarity.qsm_inference import ScoringService as qsm_sp
question = 'how to cancel my order'
data = {'query':question}
qsm_sp.predict(json.dumps(data))[1]

```

## Information Retrieval (IR)

### Training IR Model

#### Training BiEncoder model with BCE loss


```python
ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))


python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_train.py')

%run -i  $python_file \
        --train_data_path  {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/train.tsv'} \
        --test_data_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/test.tsv'} \
        --model_name_or_path 'bert-base-uncased' \
        --architecture 'bi' \
        --loss 'BiEncoderBCE' \
        --max_query_length 32 \
        --max_passage_length 384 \
        --do_lower_case \
        --train_batch_size 64 \
        --test_batch_size 128 \
        --num_train_epochs 2 \
        --gpu 0,1,2,3 \
        --save_steps 1000 \
        --print_freq 500 \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/biencoder_bertbase/'}
```

#### Training BiEncoder with Weighted Softmax Loss(with Inbatch Negatives)


```python
################## THIS CELL IS USED TO CREATE TRAINING FILE#################
## This training using a json file with query, positive doc and list of hardnegative documents for each query

import json
def process_datapoint(datapoint, data, max_hard_negs, ignore_negatives=False):
    QID = 'qid'
    PID = 'pid'
    PASSAGE = 'passage'
    QUERY = 'query'
    LABEL = 'label'
    HARD_NEGATIVES = 'hard_negatives'
    if datapoint[QID] in data:
        val = data[datapoint[QID]]
    else:
        val = dict()
        val[QUERY] = datapoint[QUERY]
        val[PASSAGE] = []
        val[HARD_NEGATIVES] = []
        
    if int(datapoint[LABEL]) == 1:
        val[PASSAGE].append(datapoint[PASSAGE])
    elif (int(datapoint[LABEL]) == 0) and (not ignore_negatives) and len(val[HARD_NEGATIVES]) < max_hard_negs:
        val[HARD_NEGATIVES].append(datapoint[PASSAGE])    
    data[datapoint[QID]] = val
                

def get_biencoder_data_amazon(input_file, output_file, max_hard_negs = 15):
    data = dict()
    with open(input_file,'r') as f:
        header = f.readline().strip().split('\t')
        count = 0
        for line in f:
            count += 1
            dataline = line.strip().split('\t')
            datapoint = dict()
            for index,title in enumerate(header):
                datapoint[title] = dataline[index]
            process_datapoint(datapoint, data, max_hard_negs = max_hard_negs)

    keys_to_del = []
    for key in data:
        if len(data[key]['passage']) == 0:
            keys_to_del.append(key)

    for key in keys_to_del:
        del data[key]

    data['701375']['hard_negatives'] = ['i\'m testing with this', 'thid hjh']
    with open(output_file, "w") as outfile:  
        json.dump(data, outfile)
        
    return data

input_file = ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/train1.tsv'
output_file = ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/train_datapoints.json'
#input_file = '/data/qyouran/QABot/d2_benchmark/final_data_bi/train.tsv'
#output_file = '/data/QAData/InformationRetrievalData/amazon/sample/train_datapoints.json'
data = get_biencoder_data(input_file, output_file, max_hard_negs = 15)

```


```python
%%time

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_train.py')

%run -i  $python_file \
        --train_data_path  {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/train_datapoints.json'} \
        --test_data_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/test.tsv'} \
        --model_name_or_path 'bert-base-uncased' \
        --architecture 'bi' \
        --loss 'BiEncoderNLL' \
        --max_query_length 32 \
        --max_passage_length 384 \
        --use_hard_negatives \
        --hard_negatives_weight_factor 0.65\
        --projection_dim 128 \
        --do_lower_case \
        --train_batch_size 8 \
        --test_batch_size 128 \
        --num_train_epochs 2 \
        --gpu 1 \
        --save_steps 1000 \
        --print_freq 500 \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/biencoder_bertbase_hardneg'}
    
```

#### Training Bi-Encoder with Adding new Architecture for model, dataset & loss function (here Biencoder model with MSMARCO Dataset & Triplet loss)


```python
%%time

## For Training, the "architecture" parameter will result in different configuration of architectures, loss function and datasets.
## If adding a new architecture,
##       (a) In ir_utils.py update "get_ir_model_attributes" to return the model, config & tokenizer for both query and document
##           for the new architecture. 
##           - It supports 'cross', 'bi' & 'single' as substring in the name of the architecture
##           - If 'cross' in architecture the model has one encoder which takes both query and document together to compute the score
##           - If 'bi' in architecture the model takes one encoder for query and another enocoder for document, computes embeddings and uses distance to compute score
##           - If 'single' in architecture this is similar to 'bi' but it uses the same encoder for encoding both query and document
##       (b) If the architecture does not fit into any of the current "bi", "cross", "single" encoder classes defined in "ir_model.py"
##           create a new encoder model class for that architecture. 
##       (c) Make sure the model Instance you use for encoder is defined in "get_encoding_vector" function inside "ir_model.py". Update that 
##           function to allow the new model instance
##       (c) Make sure correct query_transform and context_transform are instantiated in "get_ir_data_transform" function in "ir_utils.py" file
##       (d) Make sure correct train_dataset, val_dataset are created for the given architecture in ir_train.py file
##       (e) Make sure the correct loss function is passed as input argument --loss argument based on corresponding 
##           model architecture result in "ir_model.py" file and corresponding input passed to loss functions defined in "ir_loss.py" file


## NOTE: model_name_or_path COULD BE A LOCAL PATH OR CHECKPOINT DIRECTORY WHERE THE MODEL WILL START FINETINING FROM THAT POINT


import os
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_train.py')

%run -i  $python_file \
        --train_data_path  {ROOT_QASYSTEM_DATA+'MSMARCO/trainsample.tsv'} \
        --test_data_path {ROOT_QASYSTEM_DATA+'MSMARCO/trainsample.tsv'} \
        --model_name_or_path 'bert-base-uncased' \
        --architecture 'bi-msmarco-triplet' \
        --loss 'BiEncoderNLL' \
        --max_query_length 32 \
        --max_passage_length 384 \
        --use_hard_negatives \
        --hard_negatives_weight_factor 0.0\
        --do_lower_case \
        --train_batch_size 40 \
        --test_batch_size 1024 \
        --num_train_epochs 2 \
        --gpu 0,1,2,3 \
        --save_steps 500 \
        --print_freq 25 \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/msmarco_biencoder_triplet'}
```

#### Training Cross Encoder with BCE Loss


```python
%%time

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))


python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_train.py')

%run -i  $python_file \
        --train_data_path  {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/train.tsv'} \
        --test_data_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/sample/test.tsv'} \
        --model_name_or_path '/data/QAArtifacts/model/ir_artifacts/rerank_crossencoder_bert/' \
        --architecture 'cross' \
        --loss 'BCE' \
        --do_lower_case \
        --train_batch_size 32 \
        --test_batch_size 128 \
        --num_train_epochs 2 \
        --gpu 1,2 \
        --save_steps 1000 \
        --print_freq 500 \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/crossencoder_bertbase/'}
```

### Inference IR Model

#### Inference BM25


```python
%%time

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_inference.py')

%run -i  $python_file \
        --passage_collection_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/finetune_passage_collection.json'} \
        --qrels_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/finetune_qrels_rank_test.json'} \
        --rank_model_name_or_path 'BM25Okapi' \
        --rank_top_n 20 \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/BM25_inference/'}
```

#### Inference BiEncoder model


```python
################# CREATING DEV DATASET FOR MSMARCO USING BELOW CODE #######################

### This requires creation of collection & qrels json file ########

import os
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

from information_retrieval.ir_dataset import MSMARCOTripletDataset

collection_filepath = '/data/QAData/MSMARCO/collection.tsv'
queries_filepath = '/data/QAData/MSMARCO/queries.dev.tsv'
triplet_filepath = '/data/QAData/MSMARCO/qrels.dev.small.tsv'

MSMARCOTripletDataset.msmarco_create_dev_data_from_qrels(collection_filepath, queries_filepath, triplet_filepath)
```


```python
%%time

import os
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_inference.py')

%run -i  $python_file \
        --passage_collection_path {ROOT_QASYSTEM_DATA + 'MSMARCO/dev-collection.json'} \
        --qrels_path {ROOT_QASYSTEM_DATA + 'MSMARCO/dev-qrels.json'} \
        --rank_model_name_or_path {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/msmarco_biencoder_triplet'} \
        --rank_batch_size 2048 \
        --rank_top_n 10 \
        --gpu 0,1,2,3 \
        --prediction_file 'prediction.tsv' \
        --overwrite_context_embeddings \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/msmarco_biencoder_triplet_inference/'}
```

#### Inference CrossEncoder model


```python
%%time

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_inference.py')

%run -i  $python_file \
        --passage_collection_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/finetune_passage_collection.json'} \
        --qrels_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/finetune_qrels_rank_test.json'} \
        --rank_model_name_or_path {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/crossencoder_bertbase/'} \
        --rank_batch_size 256 \
        --rank_top_n 20 \
        --gpu 0,1 \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/crossencoder_bertbase_inference/'}
```

#### Inference (BM25 + BiEncoder) model


```python
%%time

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_inference.py')

%run -i  $python_file \
        --passage_collection_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/finetune_passage_collection.json'} \
        --qrels_path {ROOT_QASYSTEM_DATA+'InformationRetrievalData/amazon/finetune_qrels_rank_test.json'} \
        --rank_model_name_or_path 'BM25Okapi' \
        --rank_top_n 20 \
        --do_rerank \
        --rerank_model_name_or_path {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/biencoder_bertbase_hardneg_inference/'}
        --rerank_batch_size 256 \
        --rerank_top_n 10 \
        --rerank_threshold_score 0.0 \
        --rerank_score_weight 1.0 \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/BM25_inference/'}
```

#### Inference (BM25 + CrossEncoder) model


```python
%%time

ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'
import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

python_file = os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/information_retrieval/ir_inference.py')

%run -i $python_file \
        --gpu 0,1 \
        --passage_collection_path '/data/qyouran/QABot/d2_benchmark/final_data/passage_collection_expanded.json' \
        --qrels_path '/data/qyouran/QABot/d2_benchmark/final_data/qrels.json' \
        --rank_model_name_or_path 'BM25Okapi' \
        --rank_top_n 100 \
        --rank_threshold_score -1.0 \
        --do_rerank \
        --rerank_model_name_or_path '/data/qyouran/QABot/output_torch_train/d2_benchmark_cross_from_scratch/' \
        --rerank_batch_size 512 \
        --rerank_top_n 10 \
        --rerank_threshold_score -1.0 \
        --rerank_score_weight 0.65 \
        --prediction_file 'prediction.tsv' \
        --output_dir {ROOT_QASYSTEM_ARTIFACTS+'ir_artifacts/BM25_inference/'}
```

#### Real Time Inference for Ranking Model


```python
ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'

import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

from information_retrieval.ir_inference_realtime import RankPrediction
df = RankPrediction.get_documents([(1,'recharge'), (2,'cancel my order')])
df
```


```python
%%time
ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'
import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

from information_retrieval.ir_inference_realtime import RankPrediction

RankPrediction.passage_collection_path = '/data/qyouran/QABot/d2_benchmark/final_data/passage_collection_expanded.json'
RankPrediction.rank_top_n = 10
RankPrediction.rank_model_score_threshold = -1.0
RankPrediction.rank_model_name_or_path = '/home/yqinamz/output/ir_artifacts/biencoder_bertbase_inBatchNeg8/'
RankPrediction.do_rerank = False


df = RankPrediction.get_documents('troubleshoot kindle')

```


```python
%%time
ROOT_QASYSTEM_PACKAGE = '/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/'
ROOT_QASYSTEM_DATA = '/data/QAData/'
ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/'
import sys
import os
sys.path.append(os.path.join(ROOT_QASYSTEM_PACKAGE,'src/cs_qa_system_torch/'))

from information_retrieval.ir_inference_realtime import RankPrediction

RankPrediction.passage_collection_path = '/data/qyouran/QABot/d2_benchmark/final_data/passage_collection_expanded.json'
RankPrediction.rank_top_n = 100
RankPrediction.rank_model_score_threshold = -1.0
RankPrediction.rerank_model_name_or_path = '/data/qyouran/QABot/output_torch_train/d2_benchmark_cross_from_scratch/'
RankPrediction.rerank_top_n = 10
RankPrediction.do_rerank = True
RankPrediction.rerank_model_score_threshold = -1.0
RankPrediction.rerank_weight_for_mixed_score = 0.65

df = RankPrediction.get_documents('alexa')
df
```
