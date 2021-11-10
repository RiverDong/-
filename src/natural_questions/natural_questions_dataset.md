```python
# https://huggingface.co/datasets/natural_questions
# https://github.com/huggingface/datasets
# https://github.com/huggingface/datasets/issues/2401

from datasets import list_datasets, load_dataset, list_metrics, load_metric  # The version of "datasets" I used is 1.5.0
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import psutil
import copy
import itertools

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 5000)


def insert_to_list(x, inserted):
    # inserted is a list of tuples: [(index1, value1), (index2, value2), ...], 
    # which means we will simultaneously insert value1 immediately before x[index1], value2 immediately before x[index2], ...
    x = copy.deepcopy(x)
    inserted = sorted(inserted, key=lambda x: x[0])
    for i, (index, value) in enumerate(inserted):
        x.insert(index + i, value)
    return x


def get_nq_samples(x, split_type='train', remove_html_tokens=True):
    # In most cases, x should be a single NQ datapoint, e.g., nq_dataset['train'][0] or nq_dataset['validation'][0]
    
    # However, to speed up "get_all_nq_samples_parallel", we allow x to be the index of an NQ datapoint,
    # and in this case the function will fetch the NQ datapoint from the global variable "nq_dataset" by "x = nq_dataset[split_type][x]"
    # Note that the argument "split_type" is only used at here
    if isinstance(x, int):
        x = nq_dataset[split_type][x]

    samples = []
    
    sample_idx = x['id']
    
    question = x['question']['text']
    
    document_url = x['document']['url']
    
    document_title = x['document']['title']
    
    assert len(x['document']['tokens']['token']) == len(x['document']['tokens']['is_html']), x
    token_is_html = list(zip(x['document']['tokens']['token'], x['document']['tokens']['is_html']))
    
    document_list = [token for token, is_html in token_is_html if (not remove_html_tokens) or (not is_html)]
    document = ' '.join(document_list)
    
    assert len(x['annotations']['id']) == len(x['annotations']['long_answer']), x
    assert len(x['annotations']['id']) == len(x['annotations']['short_answers']), x
    assert len(x['annotations']['id']) == len(x['annotations']['yes_no_answer']), x
    for i in range(len(x['annotations']['id'])):
        annotation_idx = x['annotations']['id'][i]
                
        s_start_token, s_end_token = x['annotations']['short_answers'][i]['start_token'], x['annotations']['short_answers'][i]['end_token']
        if len(s_start_token) >= 1 and len(s_end_token) >= 1:
            assert len(s_start_token) == len(s_end_token), x
            
            short_answers_list = []
            for j in range(len(s_start_token)):
                assert s_start_token[j] != -1 and s_end_token[j] != -1, x
                short_answers_list += [token for token, is_html in token_is_html[s_start_token[j]:s_end_token[j]] if (not remove_html_tokens) or (not is_html)]
            
            short_answers = ' '.join(short_answers_list)
            short_answers_type = 'MULTIPLE_SPANS' if len(s_start_token) > 1 else 'SINGLE_SPAN'
        else:
            short_answers = ''
            short_answers_type = 'EMPTY'
            
        l_start_token, l_end_token = x['annotations']['long_answer'][i]['start_token'], x['annotations']['long_answer'][i]['end_token']
        if l_start_token != -1 and l_end_token != -1:
            long_answer_list = [token for token, is_html in token_is_html[l_start_token:l_end_token] if (not remove_html_tokens) or (not is_html)]
            long_answer = ' '.join(long_answer_list)
        else:
            long_answer = ''
            
        if l_start_token != -1 and l_end_token != -1:
            inserted = []
            for s_start_token_j, s_end_token_j in zip(s_start_token, s_end_token):
                assert s_start_token_j != -1 and s_end_token_j != -1, x
                inserted.append((s_start_token_j, ('<hl>', False)))
                inserted.append((s_end_token_j, ('<hl>', False)))
            inserted.append((l_start_token, ('<START_OF_LONG_ANSWER>', False)))
            inserted.append((l_end_token, ('<END_OF_LONG_ANSWER>', False)))
            token_is_html_highlighted = insert_to_list(token_is_html, inserted)
            
            hl_start_token = token_is_html_highlighted.index(('<START_OF_LONG_ANSWER>', False))
            hl_end_token = token_is_html_highlighted.index(('<END_OF_LONG_ANSWER>', False))
            
            long_answer_highlighted_list = [token for token, is_html in token_is_html_highlighted[(hl_start_token + 1):hl_end_token] if (not remove_html_tokens) or (not is_html)]
            assert long_answer_highlighted_list.count('<hl>') == len(inserted) - 2, x
            assert len(long_answer_highlighted_list) - len(long_answer_list) == len(inserted) - 2, x
            assert (set(long_answer_highlighted_list) - set(long_answer_list)).issubset({'<hl>'}), x
            assert '<START_OF_LONG_ANSWER>' not in long_answer_highlighted_list and '<END_OF_LONG_ANSWER>' not in long_answer_highlighted_list, x
            long_answer_highlighted = ' '.join(long_answer_highlighted_list)
        else:
            long_answer_highlighted = ''
        
        if x['annotations']['yes_no_answer'][i] != -1:
            yes_no_answer = 'YES' if bool(x['annotations']['yes_no_answer'][i]) else 'NO'
        else:
            yes_no_answer = ''
            
        if len(s_start_token) >= 1 and len(s_end_token) >= 1:
            inserted = []
            for s_start_token_j, s_end_token_j in zip(s_start_token, s_end_token):
                assert s_start_token_j != -1 and s_end_token_j != -1, x
                inserted.append((s_start_token_j, ('<hl>', False)))
                inserted.append((s_end_token_j, ('<hl>', False)))
            token_is_html_highlighted = insert_to_list(token_is_html, inserted)

            document_highlighted_list = [token for token, is_html in token_is_html_highlighted if (not remove_html_tokens) or (not is_html)]
            assert document_highlighted_list.count('<hl>') == len(inserted), x
            assert len(document_highlighted_list) - len(document_list) == len(inserted), x
            assert (set(document_highlighted_list) - set(document_list)) == {'<hl>'}, x
            document_highlighted = ' '.join(document_highlighted_list)
        else:
            document_highlighted = document
            
        assert long_answer in document, x
        assert (short_answers_type == 'MULTIPLE_SPANS') or (short_answers in long_answer), x
        samples.append({'sample_idx': sample_idx, 'question': question, 'document_url': document_url, 'document_title': document_title, 
                        'document': document, 'document_highlighted': document_highlighted, 'annotation_idx': annotation_idx, 'long_answer': long_answer, 'long_answer_highlighted': long_answer_highlighted, 
                        'short_answers': short_answers, 'short_answers_type': short_answers_type, 'yes_no_answer': yes_no_answer})
        
    return samples


def get_all_nq_samples(dataset, split_type, remove_html_tokens=True):
    dataset_split = dataset[split_type]
    all_samples = []
    for i in tqdm(range(len(dataset_split))):
        all_samples.extend(get_nq_samples(dataset_split[i], remove_html_tokens=remove_html_tokens))
    return pd.DataFrame(all_samples)


def get_all_nq_samples_parallel(dataset, split_type, remove_html_tokens=True, n_cpu=psutil.cpu_count(logical=True)):
    with Pool(n_cpu) as pool:
        all_samples = pool.starmap(get_nq_samples, [(i, split_type, remove_html_tokens) for i in range(len(dataset[split_type]))])
    return pd.DataFrame(list(itertools.chain(*all_samples)))


def format_nq_train_val(df, train_or_val, use_long_answer_as_passage=False):
    df = df.fillna('')
    df_selected = copy.deepcopy(df.loc[df['short_answers_type'] == 'SINGLE_SPAN', :].reset_index(drop=True))
    
    for col in df_selected.columns:
        if col == 'yes_no_answer':
            assert (df_selected[col] == '').all()
        else:
            assert (df_selected[col] != '').all()
    for i in df_selected.index:
        assert df_selected.loc[i, 'long_answer'] in df_selected.loc[i, 'document']
        assert df_selected.loc[i, 'short_answers'] in df_selected.loc[i, 'long_answer']
        
        long_answer_highlighted_list = df_selected.loc[i, 'long_answer_highlighted'].split(' ')
        hl_indices = [idx for idx, token in enumerate(long_answer_highlighted_list) if token == '<hl>']
        assert len(hl_indices) == 2
        assert ' '.join(long_answer_highlighted_list[(hl_indices[0] + 1):hl_indices[1]]) == df_selected.loc[i, 'short_answers']
        
        document_highlighted_list = df_selected.loc[i, 'document_highlighted'].split(' ')
        hl_indices = [idx for idx, token in enumerate(document_highlighted_list) if token == '<hl>']
        assert len(hl_indices) == 2
        assert ' '.join(document_highlighted_list[(hl_indices[0] + 1):hl_indices[1]]) == df_selected.loc[i, 'short_answers']
    
    if train_or_val == 'val':
        # Each question in the validation set has 5 annotations
        # We keep only one annotation whose short answer is the most frequent
        index_selected = []
        for question in df_selected['question'].unique():
            sub_df = df_selected.loc[df_selected['question'] == question, :]
            answer_value_counts = copy.deepcopy(sub_df['short_answers'].value_counts().to_frame(name='short_answers_count'))
            answer_value_counts['short_answers_sort'] = answer_value_counts.index
            most_freq_answer = answer_value_counts.sort_values(['short_answers_count', 'short_answers_sort'], ascending=False)['short_answers_sort'].iloc[0]
            index_selected.append(sub_df.index[sub_df['short_answers'] == most_freq_answer][0])
        df_selected = df_selected.loc[index_selected, :].reset_index(drop=True)
    assert not df_selected['question'].duplicated().any()
    
    if use_long_answer_as_passage:
        df_selected = df_selected.loc[:, ['question', 'short_answers', 'long_answer', 'long_answer_highlighted']].rename(columns={'question': 'query', 'short_answers': 'answer', 'long_answer': 'passage', 'long_answer_highlighted': 'passage_hl'}).reset_index(drop=True)
    else:
        df_selected = df_selected.loc[:, ['question', 'short_answers', 'document', 'document_highlighted']].rename(columns={'question': 'query', 'short_answers': 'answer', 'document': 'passage', 'document_highlighted': 'passage_hl'}).reset_index(drop=True)
    
    return df_selected


def get_idx_collection(items_to_add, collection, merge_collection=True):
    """
    For each query (passage) in a given list, assign a qid (pid) to it, and add it to an existing query (passage)
    collection if the query (passage) is a valid string that has never occurred in the collection

    :param items_to_add: the list of queries (passages)
    :param collection: existing query (passage) collection, which is a dictionary:
                       {qid1: query1, qid2: query2, ...} ({pid1: passage1, pid2: passage2, ...})
    :param merge_collection: whether to merge the new collection to the existing collection
    :return: a list of qid (pid) of all the queries (passages) in the given list, and a collection
    """
    # Sanity check
    assert len(list(collection.values())) == len(set(collection.values()))

    items_new = sorted(list(set(items_to_add) - set(collection.values())))
    next_idx = max(set(collection.keys())) + 1 if len(collection) > 0 else 1
    collection_new = dict(zip(list(range(next_idx, next_idx + len(items_new))), items_new))

    reversed_collection = {v: k for k, v in collection.items()}
    reversed_collection_new = {v: k for k, v in collection_new.items()}

    if merge_collection:
        collection_final = {**collection, **collection_new}
        reversed_collection_final = {**reversed_collection, **reversed_collection_new}
    else:
        collection_final = collection_new
        reversed_collection_final = reversed_collection_new
    print('{0:d} unique items existed, {1:d} unique items added, {2:d} unique items in returned collection'.format(
        len(set(items_to_add)) - len(items_new), len(items_new), len(collection_final)))

    idx = list(map(reversed_collection_final.get, items_to_add))
    # The following assertion guarantees that:
    # when merge_collection is False, items_to_add does not contain any items in list(collection.values())
    assert all(i is not None for i in idx)

    return idx, collection_final


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


```python
# Print all the available datasets
# print(list_datasets())

# Load a dataset and print the first example in the training set
nq_dataset = load_dataset('natural_questions', cache_dir='/data/qyouran3/huggingface_datasets/natural_questions')
```


```python
nq_train_raw = get_all_nq_samples_parallel(nq_dataset, 'train')
nq_train_raw.to_csv('/data/qyouran3/QABot/natural-questions-data/nq_train_raw.tsv', sep='\t', index=False)
nq_train_raw
```


```python
nq_val_raw = get_all_nq_samples_parallel(nq_dataset, 'validation')
nq_val_raw.to_csv('/data/qyouran3/QABot/natural-questions-data/nq_val_raw.tsv', sep='\t', index=False)
nq_val_raw
```


```python
nq_train_raw = pd.read_csv('/data/qyouran3/QABot/natural-questions-data/nq_train_raw.tsv', sep='\t', index_col=False).fillna('')
# nq_train_raw.loc[nq_train_raw['sample_idx'] == 947884478026891751, 'document_title'] = 'NaN'

nq_train = format_nq_train_val(nq_train_raw, train_or_val='train')
```


```python
nq_val_raw = pd.read_csv('/data/qyouran3/QABot/natural-questions-data/nq_val_raw.tsv', sep='\t', index_col=False).fillna('')

nq_val = format_nq_train_val(nq_val_raw, train_or_val='val')
```


```python
nq_train_val = nq_train.append(nq_val).reset_index(drop=True)

qid_col, _ = get_idx_collection(nq_train_val['query'].to_list(), dict())
nq_train_val.insert(0, 'qid', qid_col)

pid_col, _ = get_idx_collection(nq_train_val['passage'].to_list(), dict())
pid_col = ['P' + str(pid) for pid in pid_col]
nq_train_val.insert(nq_train_val.shape[1], 'pid', pid_col)

assert is_1_to_1(nq_train_val['qid'].to_list(), nq_train_val['query'].to_list())
assert is_1_to_1(nq_train_val['pid'].to_list(), nq_train_val['passage'].to_list())
```


```python
nq_train_final = nq_train_val.iloc[:nq_train.shape[0], :].reset_index(drop=True)
nq_train_final.to_csv('/data/qyouran3/QABot/natural-questions-data/nq_train.tsv', sep='\t', index=False)
nq_train_final
```


```python
nq_val_final = nq_train_val.iloc[nq_train.shape[0]:, :].reset_index(drop=True)
nq_val_final.to_csv('/data/qyouran3/QABot/natural-questions-data/nq_val.tsv', sep='\t', index=False)
nq_val_final
```
