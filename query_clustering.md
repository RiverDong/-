```python
import os
import json
import torch
import copy
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from sklearn.manifold import MDS  # The version of "sklearn" I used is 0.22.2.post1
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, KMeans, AgglomerativeClustering, DBSCAN
from time import sleep
from torch import save, load
from sentence_transformers import SentenceTransformer  # The version of "sentence-transformers" I used is 0.3.2
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from scipy.stats import pearsonr

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 1000)

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

def is_sorted(x, ascending=True):
    if ascending:
        return all(x[i] <= x[i + 1] for i in range(len(x) - 1))
    else:
        return all(x[i] >= x[i + 1] for i in range(len(x) - 1))
    
def visual_clustering(x, dim, visual_method, cluster_label=None, plot_noisy_points=False, plot=True):
    x = copy.deepcopy(x)
    
    if cluster_label is None:
        cluster_label = np.array([0] * x.shape[0])
        
    counter = Counter(cluster_label)
    for i in range(cluster_label.shape[0]):
        if counter[cluster_label[i]] == 1:
            cluster_label[i] = -1
    
    n_noisy_points = list(cluster_label).count(-1)
    cluster_label_unique = list(np.sort(np.unique(cluster_label)))
    if len(cluster_label_unique) > 1:
        print('There are {0:d} clusters, including a cluster of {1:d} noisy points'.format(len(cluster_label_unique), n_noisy_points))

    assert x.shape[0] == cluster_label.shape[0]
    print('Shape of input array: {:}'.format(x.shape))
    
    if visual_method == 'pca':
        pca = PCA(n_components=dim, random_state=0)
        x_output = pca.fit_transform(x)
        print('Percentage of variance explained by the first {0:d} principal components: {1:}'.format(dim, pca.explained_variance_ratio_))
        print('Shape of PCA array: {:}'.format(x_output.shape))
    else:
        assert visual_method == 'mds'
        mds = MDS(dim, n_init=32, n_jobs=-1, random_state=0, dissimilarity='precomputed')
        # cosine_dissim = cdist(x, x, 'cosine')  # This function is slow
        cosine_dissim = 1 - x @ np.transpose(x)  # Rows of x must be normalized to 1
        x_output = mds.fit_transform(cosine_dissim)
        print('Shape of MDS array: {:}'.format(x_output.shape))
    
    if plot:
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1.0, len(cluster_label_unique)))
        if dim == 2:
            plt.rcParams['figure.figsize'] = [12, 12]
            for i, row in enumerate(x_output):
                if cluster_label[i] == -1 and (not plot_noisy_points):
                    continue
                plt.scatter(row[0], row[1], c='white')
                plt.text(row[0], row[1], cluster_label[i], c=colors[cluster_label_unique.index(cluster_label[i])], fontsize=8)
            plt.show()
        else:
            assert dim == 3
            angles = list(range(-90, 210, 30))
            for angle in angles:
                fig = plt.figure(figsize=(18, 18))
                ax = fig.add_subplot(111, projection='3d')
                for i, row in enumerate(x_output):
                    if cluster_label[i] == -1 and (not plot_noisy_points):
                        continue
                    ax.scatter(row[0], row[1], row[2], c='white')
                    ax.text(row[0], row[1], row[2], cluster_label[i], c=colors[cluster_label_unique.index(cluster_label[i])], fontsize=8)
                ax.view_init(0, angle)
                plt.show()
                if angle != angles[-1]:
                    clear_output(wait=True)
        
        if not plot_noisy_points:
            print('{0:d} noisy points not plotted'.format(n_noisy_points))
    
    return x_output

def do_clustering(df_query_common_sample, query_common_embed_sample_normlized, method, plot_noisy_points=False, agglomerative_distance_threshold=0.1):
    np.random.seed(0)
    df_query_common_sample = copy.deepcopy(df_query_common_sample)
    query_common_embed_sample_normlized = copy.deepcopy(query_common_embed_sample_normlized)
    
    tic = time.time()
    if method.startswith('optics'):
        clustering_model = OPTICS(min_samples=5, max_eps=0.05, metric='cosine', cluster_method='xi', n_jobs=-1)
    elif method.startswith('dbscan'):
        clustering_model = DBSCAN(eps=0.05, min_samples=5, metric='cosine', n_jobs=-1)
    elif method.startswith('agglomerative'):
        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='complete', distance_threshold=agglomerative_distance_threshold)
    elif method.startswith('kmeans'):
        clustering_model = KMeans(n_clusters=100, n_init=64, random_state=0, n_jobs=-1, algorithm='full')
    cluster_label = clustering_model.fit_predict(query_common_embed_sample_normlized)
    toc = time.time()
    print('Clustering takes {:.2f} second(s)'.format(toc - tic))
    
    output_pca2 = visual_clustering(query_common_embed_sample_normlized, dim=2, visual_method='pca', cluster_label=cluster_label, plot_noisy_points=plot_noisy_points)
    if query_common_embed_sample_normlized.shape[0] < 12000:
        output_mds2 = visual_clustering(query_common_embed_sample_normlized, dim=2, visual_method='mds', cluster_label=cluster_label, plot_noisy_points=plot_noisy_points)
        
    cluster_col_name = 'cluster_' + method
    df_query_common_sample[cluster_col_name] = list(cluster_label)
    
    pd.set_option('display.max_rows', min(1000, df_query_common_sample.shape[0] + 1))
    return df_query_common_sample, cluster_col_name
```

# Prepare data


```python
df_query_original = pd.read_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/help_search_queries_d2_benchmark.tsv', sep='\t', index_col=False).astype({'qid': str, 'query': str})
df_query_original = df_query_original.rename(columns={'qid': 'session'})
df_query_original
```


```python
df_value_counts = df_query_original['query'].value_counts()
dict_value_counts = dict(zip(list(df_value_counts.index), df_value_counts.to_list()))

df_query = df_query_original.drop_duplicates('query').reset_index(drop=True)
df_query['freq'] = df_query['query'].map(dict_value_counts.get)
df_query = df_query.sort_values('freq', ascending=False).reset_index(drop=True)

df_query['qid'], _ = get_idx_collection(df_query['query'], dict())
assert df_query['qid'].is_unique and (not df_query['qid'].isnull().any()) and (not df_query['query'].isnull().any())
df_query = df_query.loc[:, ['session', 'qid', 'query', 'freq']]

df_query.to_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/help_search_queries_d2_benchmark_unique.tsv', sep='\t', index=False)
df_query
```


```python
df_query = pd.read_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/help_search_queries_d2_benchmark_unique.tsv', sep='\t', index_col=False).astype({'session': str, 'qid': int, 'query': str, 'freq': int})

df_query_common = df_query.loc[df_query['freq'] > 1, :].reset_index(drop=True)

df_query_common.to_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/help_search_queries_d2_benchmark_unique_common.tsv', sep='\t', index=False)
df_query_common
```

# Apply QSM to get embedding vectors


```python
df_query_common = pd.read_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/help_search_queries_d2_benchmark_unique_common.tsv', sep='\t', index_col=False).astype({'session': str, 'qid': int, 'query': str, 'freq': int})
```


```python
model_name = '/home/yqinamz/output/EXP_1002/qa380_Mix_1USE_9R-sbert-2020-10-09_19-14-39/'
model = SentenceTransformer(model_name)

gpu = '7'
device = torch.device('cuda:{:}'.format(gpu) if torch.cuda.is_available() and gpu != '' else 'cpu')
model.to(device)
```


```python
%%time
query_common_embed = model.encode(df_query_common['query'].to_list())
np.save('/data/qyouran/QABot/d2_benchmark/query_common_embed.npy', query_common_embed)
```

# Visualization


```python
query_common_embed = np.load('/data/qyouran/QABot/d2_benchmark/query_common_embed.npy')

query_common_embed_normlized = query_common_embed / np.linalg.norm(query_common_embed, axis=1, keepdims=True)
assert np.isclose(np.diagonal(query_common_embed_normlized[:1000, :] @ np.transpose(query_common_embed_normlized[:1000, :])), np.diagonal(np.identity(1000))).all()

# Impossible to create a very big dissimilarity matrix, so we have to randomly sample some rows
np.random.seed(0)
# sampled_index = np.sort(np.random.choice(query_common_embed_normlized.shape[0], size=1000, replace=False))  # len(sampled_index) randomly sampled queries
sampled_index = np.arange(500)  # The first len(sampled_index) most frequent queries

query_common_embed_normlized_sample = query_common_embed_normlized[sampled_index, :]
```


```python
df_query_common_embed = pd.DataFrame(query_common_embed)
df_query_common_embed
```


```python
print(query_common_embed.min())
print(query_common_embed.max())
describe = df_query_common_embed.describe()
describe
```


```python
query_common_embed_normlized_sample_mds2 = visual_clustering(query_common_embed_normlized_sample, dim=2, visual_method='mds')
```


```python
query_common_embed_normlized_sample_mds3 = visual_clustering(query_common_embed_normlized_sample, dim=3, visual_method='mds')
```


```python
query_common_embed_normlized_sample_pca2 = visual_clustering(query_common_embed_normlized_sample, dim=2, visual_method='pca')
```


```python
query_common_embed_normlized_sample_pca3 = visual_clustering(query_common_embed_normlized_sample, dim=3, visual_method='pca')
```

# Clustering


```python
df_query_common = pd.read_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/help_search_queries_d2_benchmark_unique_common.tsv', sep='\t', index_col=False)
query_common_embed = np.load('/data/qyouran/QABot/d2_benchmark/query_common_embed.npy')
assert df_query_common.shape[0] == query_common_embed.shape[0] and is_sorted(df_query_common['freq'].to_list(), ascending=False) and list(df_query_common.index) == list(range(df_query_common.shape[0])) 

n = min(list(df_query_common.loc[df_query_common['freq'] >= 50, :].index)[-1] + 1, 2000)
df_query_common_sample = copy.deepcopy(df_query_common.iloc[:n, :])
query_common_embed_sample = copy.deepcopy(query_common_embed[:n, :])

query_common_embed_sample_normlized = query_common_embed_sample / np.linalg.norm(query_common_embed_sample, axis=1, keepdims=True)
assert np.isclose(np.diagonal(query_common_embed_sample_normlized[:min(n, 1000), :] @ np.transpose(query_common_embed_sample_normlized[:min(n, 1000), :])), np.diagonal(np.identity(min(n, 1000)))).all()

print('The queries we focus on are {0:d} / {1:d} = {2:.2f}% of all the help search queries'.format(df_query_common_sample['freq'].sum(), 20889412, 100 * df_query_common_sample['freq'].sum() / 20889412))
df_query_common_sample
```


```python
df_query_common_sample_clustered, cluster_col_name = do_clustering(df_query_common_sample, query_common_embed_sample_normlized, method='agglomerative_0.06', agglomerative_distance_threshold=0.06)
print(df_query_common_sample_clustered[cluster_col_name].value_counts())

assert df_query_common_sample_clustered.drop(columns=cluster_col_name).equals(df_query_common.iloc[:n, :])
df_query_common_sample_clustered.to_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/point06agglomerative_top2000/help_search_queries_d2_benchmark_unique_common_clustered_point06agglomerative_top2000.tsv', sep='\t', index=False)
```

# Select queries for annotation


```python
pd.set_option('display.max_rows', 1200)

df_clustered = pd.read_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/point06agglomerative_top2000/help_search_queries_d2_benchmark_unique_common_clustered_point06agglomerative_top2000.tsv', sep='\t', index_col=False)
cluster_col_name = 'cluster_agglomerative_0.06'
assert is_sorted(df_clustered['freq'].to_list(), ascending=False) and list(df_clustered.index) == list(range(df_clustered.shape[0]))

# n = 600 # Among the 980 unique most high-frequent queries, 600 unique queries are selected for annotation and 380 unique queries that are similar to them are removed. Although we actually annoate 600 unique queries, we "effectively" annotate 980 unique queries, which are 6833673 / 20889412 = 32.71% of all help search queries
n = 1100  # Among the 1973 unique most high-frequent queries, 1100 unique queries are selected for annotation and 873 unique queries that are similar to them are removed. Although we actually annoate 1100 unique queries, we "effectively" annotate 1973 unique queries, which are 7951698 / 20889412 = 38.07% of all help search queries

index_selected = list()
index_removed = list()
list_query_already_selected = list()
dict_query_already_selected = dict()

for i in df_clustered.index:
    cluster_label = df_clustered.loc[i, cluster_col_name]
    query = df_clustered.loc[i, 'query']
    if cluster_label not in dict_query_already_selected:
        if cluster_label != -1:
            dict_query_already_selected[cluster_label] = query
        index_selected.append(i)
    else:
        assert cluster_label != -1
        list_query_already_selected.append(dict_query_already_selected[cluster_label])
        index_removed.append(i)
    if len(index_selected) == n:
        break
assert len(index_selected) == n and len(index_removed) == i + 1 - n

df_selected = copy.deepcopy(df_clustered.loc[index_selected, :])
df_removed = copy.deepcopy(df_clustered.loc[index_removed, :])
df_removed['query_selected_for_this_cluster'] = list_query_already_selected

n_nonunique_effective = df_selected['freq'].sum() + df_removed['freq'].sum()
print('Among the {0:d} unique most high-frequent queries, {1:d} unique queries are selected for annotation and {2:d} unique queries that are similar to them are removed. Although we actually annoate {1:d} unique queries, we "effectively" annotate {0:d} unique queries, which are {3:d} / {4:d} = {5:.2f}% of all help search queries'.format(
      i + 1, n, i + 1 - n, n_nonunique_effective, 20889412, 100 * n_nonunique_effective / 20889412))
```


```python
assert is_sorted(df_selected['freq'].to_list(), ascending=False)
df_selected.iloc[:600, :].to_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/point06agglomerative_top2000/help_search_queries_d2_benchmark_selected_1-600.tsv', sep='\t', index=False)
df_selected.iloc[600:, :].to_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/point06agglomerative_top2000/help_search_queries_d2_benchmark_selected_601-1100.tsv', sep='\t', index=False)
df_selected
```


```python
assert is_sorted(df_removed['freq'].to_list(), ascending=False)
df_removed.to_csv('/data/qyouran/QABot/d2_benchmark/help_search_queries/point06agglomerative_top2000/help_search_queries_d2_benchmark_removed.tsv', sep='\t', index=False)
df_removed
```


```python
# Sanity check
query_common_embed = np.load('/data/qyouran/QABot/d2_benchmark/query_common_embed.npy')
for i in df_removed.index:
    selected_query = df_removed.loc[i, 'query_selected_for_this_cluster']
    index = df_clustered.loc[df_clustered['query'] == selected_query, :].index
    assert len(index) == 1 and (1 - cdist(query_common_embed[i].reshape(1, -1), query_common_embed[index[0]].reshape(1, -1), 'cosine'))[0, 0] >= 0.94
```
