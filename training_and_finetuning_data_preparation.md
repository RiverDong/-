# Process MSMARCO data


* Input MSMARCO orginal: **/data/QAData/MSMARCO**
* Output MSMARCO formatted: **/data/QAData/MSMARCO_ORGANIZE**
* Load MSMARCO Processing package
* Generate a train.tsv & test.tsv files with 1:4 positive to negative labels with 50000 unique queries to the mentioned output path




```python
import sys
sys.path.append('src/ms_marco')
sys.path.append('src/cs_qa_system_torch')
sys.path.append('src/amazon_qa_data')
from msmarco_data import MSMARCOPassageData
```


```python
MSMARCOPassageData.generate_train_input_for_ranking('/data/QAData/MSMARCO_ORGANIZE/50000_1_4',0.85,1,4,50000)
```

* preprocess MSMARCO formatted file from above to do text preprocessing
* Input: **/data/QAData/MSMARCO_ORGANIZE/50000_1_4/** containing train & test tsv files
* Output: **/data/QAData/MSMARCO_ORGANIZE/50000_1_4-preprocessed/** containing train-preprocessed and test-preprocessed tsv files


```python
from factory.word_tokenizer_factory import english_preprocessor
```


```python
preprocessed_ms_data_path = '/data/QAData/MSMARCO_ORGANIZE/50000_1_4-preprocessed'
os.makedirs(preprocessed_ms_data_path)
```


```python
ms_train = pd.read_csv('/data/QAData/MSMARCO_ORGANIZE/50000_1_4/train.tsv', sep='\t', index_col=False)
ms_train, ms_train_query_preprocessed_col = english_preprocessor.preprocess(ms_train, 'query')
ms_train, ms_train_passage_preprocessed_col = english_preprocessor.preprocess(ms_train, 'passage')
ms_train = ms_train.drop(columns=['query', 'passage']).rename(columns={ms_train_query_preprocessed_col: 'query', ms_train_passage_preprocessed_col: 'passage'})
ms_train.to_csv(os.path.join(preprocessed_ms_data_path, 'train-preprocessed.tsv'), sep='\t', index=False)
ms_train
```


```python
ms_test = pd.read_csv('/data/QAData/MSMARCO_ORGANIZE/50000_1_4/test.tsv', sep='\t', index_col=False)
ms_test, ms_test_query_preprocessed_col = english_preprocessor.preprocess(ms_test, 'query')
ms_test, ms_test_passage_preprocessed_col = english_preprocessor.preprocess(ms_test, 'passage')
ms_test = ms_test.drop(columns=['query', 'passage']).rename(columns={ms_test_query_preprocessed_col: 'query', ms_test_passage_preprocessed_col: 'passage'})
ms_test.to_csv(os.path.join(preprocessed_ms_data_path, 'test-preprocessed.tsv'), sep='\t', index=False)
ms_test
```


```python


import numpy as np
import pandas as pd
import pickle
import os
import re
import copy
import time
import random
import json
from json import JSONDecodeError
from tqdm import tqdm
from factory.word_tokenizer_factory import english_preprocessor
from data_preparation import download_from_s3, get_files_in_folder, get_idx_collection

def is_sorted(x, ascending=True):
    if ascending:
        return all(x[i] <= x[i + 1] for i in range(len(x) - 1))
    else:
        return all(x[i] >= x[i + 1] for i in range(len(x) - 1))
    
def is_1_to_1(x, y):
    assert len(x) == len(y)
    d = dict()
    for i in tqdm(range(len(x))):
        if x[i] not in d:
            if y[i] in set(d.values()):
                return False
            d[x[i]] = y[i]
        else:
            if d[x[i]] != y[i]:
                return False
    return True
```

# Get MSMARCO data (preprocessed) and Amazon data (preprocessed & unified) for IR model training

## Get preprocessed and unified Amazon train/test data

* Used annotated data from MLDAs based on our annotation tool to get question, passage, answer triplets
* Input: **/data/QAData/ANNOTATION_TOOL/qabot-annotation-output** and **/data/QAData/ANNOTATION_TOOL/qabot-annotation-output-new**
* Output: **/data/QAData/AMAZONDATA/Data_500_4_445_np_inf_original_none_unified_passage** folder containing rank_train, rank_test & passage_collection files


```python
# The following command generates data_all, rank_train, rank_test, query_collection, passage_collection
# The codes to generate answer_train and answer_test are retained in data_preparation.py, but are skipped and never used in practice

%run -i src/amazon_qa_data/data_preparation.py \
        --root_path '/data/QAData/AmazonData/Data_500_4_445_np_inf_original_none_unified_passage' \
        --data_path '/data/QAData/ANNOTATION_TOOL/qabot-annotation-output' \
        --data_new_path '/data/QAData/ANNOTATION_TOOL/qabot-annotation-output-new' \
        --n_test 500 \
        --r_neg_pos 4 \
        --split_granularity '(445,np.inf)' \
        --text_type 'unified_passage'
```

## Split rank_train into training and testing sets, and merge them with ms_train and ms_test respectively to get the training and testing data

* Input: **/data/QAData/AMAZONDATA/Data_500_4_445_np_inf_original_none_unified_passage** folder containing rank_train, rank_test & passage_collection files and also **/data/QAData/MSMARCO_ORGANIZE/50000_1_4-preprocessed/** folder train-preprocessed and test-preprocessed of MSMARCO
*Output: **/data/QAData/InformationRetrievalData/amazon** containing train,test & valid(not anymore) tsv files
    
    


```python
import pandas as pd
ms_train = pd.read_csv('/data/QAData/MSMARCO_ORGANIZE/50000_1_4-preprocessed/train-preprocessed.tsv', sep='\t', index_col=False)
ms_test = pd.read_csv('/data/QAData/MSMARCO_ORGANIZE/50000_1_4-preprocessed/test-preprocessed.tsv', sep='\t', index_col=False)
amazon_train_test = pd.read_json('/data/QAData/AMAZONDATA/Data_500_4_445_np_inf_original_none_unified_passage/rank_train-500-4.0-(445,np.inf)-original-None-unified_passage.json', orient='columns', typ='frame')
```


```python
test_ratio = 0.1
resampling_ratio = 5

np.random.seed(0)
test_query = amazon_train_test['query'].unique()
test_query_sampled = np.random.choice(test_query, size=int(test_ratio * test_query.shape[0]), replace=False)
test_mask = amazon_train_test['query'].isin(test_query_sampled)

amazon_test = amazon_train_test.loc[test_mask, :].reset_index(drop=True)
amazon_train = amazon_train_test.loc[~test_mask, :].reset_index(drop=True)

amazon_train = pd.concat([amazon_train] * resampling_ratio, ignore_index=True)
```


```python
assert set(ms_train.columns) == set(amazon_train.columns)
train = ms_train.append(amazon_train, ignore_index=True, sort=True).sample(frac=1, random_state=0).reset_index(drop=True)
train
```


```python
assert set(ms_test.columns) == set(amazon_test.columns)
test = ms_test.append(amazon_test, ignore_index=True, sort=True).sample(frac=1, random_state=0).reset_index(drop=True)
test
```

## Get passages returned by BM25, and use these passages to get the validation data


```python
## DEPRECATED THIS CALSS ..you need to modify ir_inference_combined
%run -i src/cs_qa_system_torch/information_retrieval/ir_inference.py \
        --passage_collection_path '/data/QAData/AMAZONDATA/Data_500_4_445_np_inf_original_none_unified_passage/passage_collection-500-4.0-(445,np.inf)-original-None-unified_passage.json' \
        --qrels_path '/data/QAData/AMAZONDATA/Data_500_4_445_np_inf_original_none_unified_passage/rank_test-500-4.0-(445,np.inf)-original-None-unified_passage.json' \
        --result_dir '/data/QAData/AMAZONDATA/Data_500_4_445_np_inf_original_none_unified_passage/BM25' \
        --ir_model_name 'BM25Okapi-500-4.0-(445,np.inf)-original-None-unified_passage' \
        --word_tokenizer_name 'simple_word_tokenizer_no_stopwords_stem' \
        --n 100 \
        --add_preprocessed 1 \
        --all_k '1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100' \
        --all_rank_thres_interested '1,2,3,4,5,6,7,8,9,10,20,30,40,50'
```


```python
valid = pd.read_csv('/data/QAData/AMAZONDATA/Data_500_4_445_np_inf_original_none_unified_passage/BM25/BM25Okapi-500-4.0-(445,np.inf)-original-None-unified_passage_prediction.tsv', sep='\t', index_col=False)
valid = valid.reindex(columns=['qid', 'query', 'pid', 'passage', 'label'])
print(valid['label'].value_counts())
valid
```

## Save the training, testing and validation data


```python
 final_data_path = '/data/QAData/InformationRetrievalData/amazon'
os.makedirs(final_data_path)

train.to_csv(os.path.join(final_data_path, 'train.tsv'), sep='\t', index=False)
test.to_csv(os.path.join(final_data_path, 'test.tsv'), sep='\t', index=False)
valid.to_csv(os.path.join(final_data_path, 'valid.tsv'), sep='\t', index=False)
```

# Get Amazon data (augmented by QSM) for IR and AE model finetuning

* All the below sections takes annotation data from scientists and MLDAs from
* Input: **qabot-annotation-output-ir-b** s3 bucket and outputs
* Output: **InformationRetrievalData/amazon** and **AnswerExtractionData/amazon/** folder containing finetune_train and finetune_test files

## Get the annotated data


```python
# Need to change "aws_access_key_id" and "aws_secret_access_key" accordingly
data_path = download_from_s3(
    root_path='/data/QAData/qabot-annotation-output-ir-b', 
    s3_region='us-east-1', 
    aws_access_key_id='aws_access_key_id', 
    aws_secret_access_key='aws_secret_access_key', 
    bucket_name='qabot-annotation-output-ir-b')
```


```python
data_path = '/data/QAData/qabot-annotation-output-ir-b'
all_data_name = get_files_in_folder(data_path, end_with='.txt', not_start_with=('count.txt', ),
                                    return_full_path=False, keep_ext=False, recursively=False)
all_data = list()
all_source_path = list()
for data_name in all_data_name:
    try:
        with open(os.path.join(data_path, data_name + '.txt'), 'r') as f:
            all_data.append(json.load(f))
            all_source_path.append(os.path.join(os.path.basename(data_path.rstrip('/')), data_name + '.txt'))
    except JSONDecodeError:
        print('{:}.txt has incorrect format so it is not loaded'.format(data_name))

data = pd.DataFrame(all_data).astype({'query': str, 'passage': str, 'label': int, 'answer': str, 'user': str})
assert data.notnull().all(axis=None)

# We will sort it by filenames such that the passages of each query are sorted by their BM25 relevance scores (the first passage has the highest score)
data['source_path'] = all_source_path
data['filename_to_sort'] = data['source_path'].map(lambda x: int(x.split('/')[-1][:-4]))
data = data.sort_values('filename_to_sort').reset_index(drop=True)
```


```python
def modify_string(data, col, old, new, print_info=False):
    s = set()
    for i in data.index:
        old_string = data.loc[i, col]
        new_string = old_string.replace(old, new)
        if old_string != new_string:
            data.loc[i, col] = new_string
            if (old_string not in s) and print_info:
                print('    {0:}\n -> {1:}'.format(old_string, new_string))
            s.add(old_string)
    print('{0:d} unique {1:} are modified ({2:} -> {3:})\n'.format(len(s), col, old, new))
    
modify_string(data, 'query', '&nbsp;', '\xa0', print_info=True)
modify_string(data, 'query', '&amp;', '&', print_info=True)

modify_string(data, 'passage', '&nbsp;', '\xa0', print_info=True)
modify_string(data, 'passage', '&amp;', '&', print_info=False)

modify_string(data, 'answer', '&nbsp;', '\xa0', print_info=True)
modify_string(data, 'answer', '&amp;', '&', print_info=True)
```


```python
# This query has an answer but its "label" was incorrectly labeled as 0, so I change it to 1
wrong_labels_index = [4120]
for i in wrong_labels_index:
    data.loc[i, 'label'] = 1

duplicated_queries = ["I'd like to add an item to my existing order", "An item received as a gift", "Return item help"]
index_removed = list()
for query in duplicated_queries:
    temp = data.loc[data['query'] == query, ['query', 'passage', 'label', 'answer', 'user']]
    index_removed.extend(list(temp.iloc[10:, :].index))
    assert (temp.iloc[:10, :].reset_index(drop=True) == temp.iloc[10:, :].reset_index(drop=True)).all(axis=None)
data = data.drop(index=index_removed).reset_index(drop=True)
```


```python
df_query_all = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/all_queries_considered_in_experiment.tsv', sep='\t', index_col=False)
assert df_query_all['qid'].is_unique and df_query_all['query'].is_unique

data = pd.merge(left=data, right=df_query_all, on=['query'], how='left', validate='many_to_one', indicator=True)
assert (data['_merge'] == 'both').all()
data = data.drop(columns=['_merge'])

data['base_qid'] = data['qid']
data['base_query'] = data['query']
data['qs_score'] = 1.0

data = data.loc[:, ['base_qid', 'base_query', 'qs_score', 'qid', 'query', 'freq', 'passage', 'label', 'answer', 'user', 'source_path', 'filename_to_sort']].reset_index(drop=True)
```


```python
# Sanity check
for i in data.index:
    if data.loc[i, 'label'] == 0:
        assert data.loc[i, 'answer'] == ''
    else:
        assert data.loc[i, 'label'] == 1
        assert (data.loc[i, 'answer'] != '') and (data.loc[i, 'answer'] in data.loc[i, 'passage'])
assert data['query'].nunique() * 10 == data.shape[0]
assert (not data.duplicated(['query', 'passage']).any())
assert is_sorted(data['filename_to_sort'].to_list())
```


```python
print('Number of queries each user annotated:')
print(data['user'].value_counts() // 10)

n_covered = data.groupby('query').apply(lambda x: (x['label'] == 1).any()).sum()
n = data['query'].nunique()
print('{0:d}/{1:d} = {2:.2f}% of queries have >= 1 answers'.format(n_covered, n, 100 * n_covered / n))

data.to_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_annotation_result.tsv', sep='\t', index=False)
data
```

## Get similar queries for each query we have


```python
data = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_annotation_result.tsv', sep='\t', index_col=False).fillna('')

df_query_all = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/all_queries_considered_in_experiment.tsv', sep='\t', index_col=False)
assert df_query_all['qid'].is_unique and df_query_all['query'].is_unique

similar_queries = dict()
with open('/home/yqinamz/QA_Bot/IR/ir_data/850k_similar_top5.json', 'r') as f:
    for line in f:
        try:
            temp = json.loads(line)
        except JSONDecodeError:
            print('The format of the following line is wrong, so loading is stopped (there are already {0:d} queries in similar_queries):\n\n{1:}'.format(len(similar_queries), line))
            break
        assert len(temp['similar_q']) == 5
        similar_queries[int(temp['qid'])] = max(temp['similar_q'], key=lambda x: x[2])
```


```python
l = [v + [qid] for qid, v in similar_queries.items()]

df_query_aug = pd.DataFrame(l, columns=['base_qid', 'base_query', 'qs_score', 'qid']).astype({'base_qid': int, 'base_query': str, 'qid': int})
assert df_query_aug['qid'].is_unique

df_query_aug = df_query_aug.loc[~df_query_aug['base_qid'].isin({75844, 1124315}), :].reset_index(drop=True)

df_query_aug
```


```python
# Get back the freq and query
df_query_aug = pd.merge(left=df_query_aug, right=df_query_all, on=['qid'], how='left', validate='one_to_one', indicator=True)
assert (df_query_aug['_merge'] == 'both').all()
df_query_aug = df_query_aug.drop(columns=['_merge'])

df_query_aug
```


```python
df_query_aug['query_deduplicate'] = df_query_aug['query'].str.lower()

query_deduplicate_mask = df_query_aug.duplicated(['query_deduplicate'], keep=False)
temp = df_query_aug.loc[query_deduplicate_mask, :].groupby('query_deduplicate').apply(lambda x: x.sort_values('qs_score', ascending=False).iloc[0, :]).reset_index(drop=True)
df_query_aug = df_query_aug.loc[~query_deduplicate_mask, :].append(temp).reset_index(drop=True)

annotated_queries = set(data['query'].to_list())
annotated_queries = {s.lower() for s in annotated_queries}
annotated_queries_mask = df_query_aug['query_deduplicate'].isin(annotated_queries)
assert annotated_queries_mask.sum() == 998
df_query_aug = df_query_aug.loc[~annotated_queries_mask, :]

df_query_aug = df_query_aug.drop(columns=['query_deduplicate']).reset_index(drop=True)

df_query_aug
```


```python
# Retain only the queries with high qs_score 
similarity_threshold = 0.94
df_query_aug = df_query_aug.loc[df_query_aug['qs_score'] > similarity_threshold, :].reset_index(drop=True)
df_query_aug
```


```python
# Sanity check
assert df_query_aug['query'].str.lower().is_unique and df_query_aug['qid'].is_unique
assert is_1_to_1(df_query_aug['qid'].to_list(), df_query_aug['query'].to_list())
assert is_1_to_1(df_query_aug['base_qid'].to_list(), df_query_aug['base_query'].to_list())
```


```python
# Get the annotations 
n_original = df_query_aug.shape[0]

df_query_aug = pd.merge(left=df_query_aug, right=data.loc[:, ['base_qid', 'base_query', 'passage', 'label', 'answer', 'filename_to_sort']], how='left', on=['base_qid', 'base_query'], indicator=True)
assert (df_query_aug['_merge'] == 'both').all()
df_query_aug = df_query_aug.drop(columns=['_merge'])

assert n_original * 10 == df_query_aug.shape[0]

df_query_aug
```


```python
n_unique_original = df_query_aug['query'].str.lower().nunique()
data_columns = list(data.drop(columns=['user', 'source_path']).columns)
assert sorted(list(df_query_aug.columns)) == sorted(data_columns)
df_query_aug = df_query_aug.reindex(columns=data_columns)
df_query_aug = df_query_aug.append(data.drop(columns=['user', 'source_path']), ignore_index=True)
assert df_query_aug['query'].str.lower().nunique() - n_unique_original == 998

df_query_aug = df_query_aug.sort_values(['qid', 'filename_to_sort'], ascending=[True, True]).reset_index(drop=True)

def check(x):
    assert x['qs_score'].nunique() == 1
    assert x['freq'].nunique() == 1
    assert x['base_qid'].nunique() == 1
    assert x['base_query'].nunique() == 1
df_query_aug.groupby('qid').apply(check)
assert is_1_to_1(df_query_aug['qid'].to_list(), df_query_aug['query'].to_list())
assert is_1_to_1(df_query_aug['base_qid'].to_list(), df_query_aug['base_query'].to_list())

df_query_aug.to_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/df_query_aug.tsv', sep='\t', index=False)
df_query_aug
```

## Get the final dataset


```python
df_query_aug = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/df_query_aug.tsv', sep='\t', index_col=False).fillna('')
assert df_query_aug['query'].nunique() * 10 == df_query_aug.shape[0]
```


```python
df_query_aug_selected = copy.deepcopy(df_query_aug)
df_query_aug_selected
```


```python
def adjust_ratio(x, r_neg_to_pos=3):
    assert x['label'].isin({0, 1}).all()
    assert is_sorted(x['filename_to_sort'].to_list())
    
    mask_pos = (x['label'] == 1)
    n_pos = mask_pos.sum()
    n_neg = int(max(r_neg_to_pos, n_pos * r_neg_to_pos))
    
    df_pos = x.loc[mask_pos, :]
    df_neg = x.loc[~mask_pos, :].iloc[:n_neg, :]
    
    return df_pos.append(df_neg, ignore_index=True)

df_query_aug_selected_ratio = df_query_aug_selected.groupby('query').apply(adjust_ratio).reset_index(drop=True)
df_query_aug_selected_ratio
```


```python
# Get preprocessed passages
english_preprocessor.noise_regex['URL'] = (re.compile(r'(\(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^)\s]{2,}|www\.[a-zA-Z0-9]' \
                                                      r'[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^)\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^)\s]{2,}|www\.' \
                                                      r'[a-zA-Z0-9]+\.[^)\s]{2,})'), )

df_query_aug_selected_ratio['passage_preprocessed'] = df_query_aug_selected_ratio['passage'].map(english_preprocessor.preprocess_single)
```


```python
# Get pid and passage collection
passage_id, _ = get_idx_collection(df_query_aug_selected_ratio['passage'].to_list(), dict())
df_query_aug_selected_ratio['pid'] = passage_id
df_query_aug_selected_ratio = df_query_aug_selected_ratio.loc[:, ['base_qid', 'base_query', 'qs_score', 'qid', 'query', 'freq', 'pid', 'passage', 'passage_preprocessed', 'label', 'answer', 'filename_to_sort']]

assert is_1_to_1(df_query_aug_selected_ratio['pid'].to_list(), df_query_aug_selected_ratio['passage'].to_list())
assert is_1_to_1(df_query_aug_selected_ratio['pid'].to_list(), df_query_aug_selected_ratio['passage_preprocessed'].to_list())

passage_collection = dict()
for i in tqdm(df_query_aug_selected_ratio.index):
    pid = int(df_query_aug_selected_ratio.loc[i, 'pid'])
    if pid not in passage_collection:
        passage_collection[pid] = [df_query_aug_selected_ratio.loc[i, 'passage'], df_query_aug_selected_ratio.loc[i, 'passage_preprocessed']]
with open('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/finetune_passage_collection.json', 'w') as f:
    json.dump(passage_collection, f)
```


```python
n_covered = df_query_aug_selected_ratio.groupby('query').apply(lambda x: (x['label'] == 1).any()).sum()
n = df_query_aug_selected_ratio['query'].nunique()
print('In the augmented data, {0:d}/{1:d} = {2:.2f}% of queries have >= 1 answers'.format(n_covered, n, 100 * n_covered / n))

print(df_query_aug_selected_ratio['label'].value_counts())

df_query_aug_selected_ratio.to_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/df_query_aug_selected_ratio.tsv', sep='\t', index=False)
df_query_aug_selected_ratio
```

## Get the final training and testing datasets for IR and AE


```python
df_query_aug_selected_ratio = pd.read_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/df_query_aug_selected_ratio.tsv', sep='\t', index_col=False).fillna('')

with open('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_small/test_qid_sampled_small', 'rb') as f:
    test_qid_sampled_small = pickle.load(f)
```


```python
np.random.seed(0)
test_qid = df_query_aug_selected_ratio['qid'].unique()
test_qid_sampled = np.random.choice(test_qid, size=int(test_qid.shape[0] * 0.1), replace=False)
test_qid_sampled = set(test_qid_sampled) | set(test_qid_sampled_small)
```


```python
rank_data = copy.deepcopy(df_query_aug_selected_ratio.loc[:, ['qid', 'query', 'pid', 'passage_preprocessed', 'label']].rename(columns={'passage_preprocessed': 'passage'}))

rank_data = rank_data.sort_values(['qid', 'label'], ascending=[True, False])

mask_rank_test = rank_data['qid'].isin(test_qid_sampled)
```


```python
rank_test = rank_data.loc[mask_rank_test, :].reset_index(drop=True)

n_covered = rank_test.groupby('query').apply(lambda x: (x['label'] == 1).any()).sum()
n = rank_test['query'].nunique()
print('In rank_test, {0:d}/{1:d} = {2:.2f}% of queries have >= 1 answers'.format(n_covered, n, 100 * n_covered / n))

print(rank_test['label'].value_counts())

rank_test.to_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/finetune_rank_test.tsv', sep='\t', index=False)
rank_test
```


```python
rank_train = rank_data.loc[~mask_rank_test, :].reset_index(drop=True)

n_covered = rank_train.groupby('query').apply(lambda x: (x['label'] == 1).any()).sum()
n = rank_train['query'].nunique()
print('In rank_train, {0:d}/{1:d} = {2:.2f}% of queries have >= 1 answers'.format(n_covered, n, 100 * n_covered / n))

print(rank_train['label'].value_counts())

rank_train.to_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/finetune_rank_train.tsv', sep='\t', index=False)
rank_train
```


```python
answer_data = copy.deepcopy(df_query_aug_selected_ratio.loc[df_query_aug_selected_ratio['label'] == 1, ['qid', 'query', 'passage', 'answer']])

assert answer_data.apply(lambda x: (x['answer'] != '') and (x['answer'] in x['passage']), axis=1).all()

answer_data['query_deduplicate'] = answer_data['query'].str.lower()
assert not answer_data.duplicated(['query_deduplicate', 'passage']).any()
answer_data = answer_data.drop(columns='query_deduplicate')

answer_data = answer_data.sort_values('qid', ascending=True)

mask_answer_test = answer_data['qid'].isin(test_qid_sampled)
```


```python
answer_test = answer_data.loc[mask_answer_test, :].reset_index(drop=True)

answer_test.to_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/finetune_answer_test.tsv', sep='\t', index=False)
answer_test
```


```python
answer_train = answer_data.loc[~mask_answer_test, :].reset_index(drop=True)

answer_train.to_csv('/data/qyouran/QABot/QAData/HelpDoc_InitialLaunch/finetune_data_large/finetune_answer_train.tsv', sep='\t', index=False)
answer_train
```

# Answer Extraction Dataset

* The train and test dataset for Answer extraction are **train_answerextraction_100.tsv** and **test_answerextraction_100.tsv**
* They are create from rank_train json file of IR model above and using the function **create_train_test_split(path_to_json:str, max_answer_words:int = 100, train_test_split_ratio:float = 0.8** inside run_squad_final file
* The 100 in file name is the max_answer_length filter we applied on ran_train dataset
