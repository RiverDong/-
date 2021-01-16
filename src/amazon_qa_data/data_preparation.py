import argparse
import copy
import os
import re
import json
import time
import utils
import constants
import math
import difflib
import boto3
import nltk
import nltk.tokenize.punkt as pkt
import pandas as pd
import numpy as np
import psutil
from multiprocessing import Pool
from utils import plot_distr
from json import JSONDecodeError
from factory.word_tokenizer_factory import WordTokenizerFactory, english_preprocessor

parser = argparse.ArgumentParser(description='This module is used to prepare training and testing data for '
                                             'passage ranking and answer extraction models')

parser.add_argument('--root_path', type=str,
                    help='local root path to save data and result')
parser.add_argument('--data_path', type=str,
                    help='local path to save raw QA data where urls and passages are given')
parser.add_argument('--data_new_path', type=str,
                    help='local path to save raw QA data where urls and passages are added by annotators')

parser.add_argument('--aws_access_key_id', type=str,
                    help='aws access key id of the S3 buckets where raw QA data are stored')
parser.add_argument('--aws_secret_access_key', type=str,
                    help='aws secret access key of the S3 buckets where raw QA data are stored')
parser.add_argument('--s3_region', type=str, default='us-east-1',
                    help='region of the S3 buckets where raw QA data are stored')
parser.add_argument('--bucket_name', type=str, default='qabot-annotation-output',
                    help='name of the S3 bucket to store raw QA data where urls and passages are given')
parser.add_argument('--bucket_new_name', type=str, default='qabot-annotation-output-new',
                    help='name of the S3 bucket to store raw QA data where urls and passages are added by annotators')

parser.add_argument('--data_all_name', type=str, default='data_all',
                    help='file name of all passage ranking data that are saved to local path')
parser.add_argument('--rank_test_name', type=str, default='rank_test',
                    help='file name of the passage ranking testing data that are saved to local path')
parser.add_argument('--rank_train_name', type=str, default='rank_train',
                    help='file name of the passage ranking training data that are saved to local path')
parser.add_argument('--answer_test_name', type=str, default='answer_test',
                    help='file name of the answer extraction testing data that are saved to local path')
parser.add_argument('--answer_train_name', type=str, default='answer_train',
                    help='file name of the answer extraction training data that are saved to local path')
parser.add_argument('--query_collection_name', type=str, default='query_collection',
                    help='file name of the query collection that is saved to local path')
parser.add_argument('--passage_collection_name', type=str, default='passage_collection',
                    help='file name of the passage collection that is saved to local path')

parser.add_argument('--n_words_before_distr_plot_name', type=str, default='n_words_before_distr',
                    help='file name of the plot of distribution of number of words per passage before splitting')
parser.add_argument('--n_words_after_distr_plot_name', type=str, default='n_words_after_distr',
                    help='file name of the plot of distribution of number of words per passage after splitting')
parser.add_argument('--snippet_distr_plot_name', type=str, default='snippet_distr',
                    help='file name of the plot of distribution of number of snippets per query')
parser.add_argument('--rank_label_distr_plot_name', type=str, default='rank_label_distr',
                    help='file name of the plot of distribution of number of label 1 per query in '
                         'all passage ranking data')
parser.add_argument('--answer_label_distr_plot_name', type=str, default='answer_label_distr',
                    help='file name of the plot of distribution of number of label 1 per query in '
                         'answer extraction training data')

parser.add_argument('--n_test', type=int,
                    help='number of randomly sampled queries in the testing data')
parser.add_argument('--r_neg_pos', type=float,
                    help='number of negative samples over number of positive samples in passage ranking training data')

parser.add_argument('--split_granularity', type=str, default='(np.inf,np.inf)',
                    help='a string that will be parsed as a tuple of two thresholds used to specify the granularity '
                         'we work on; if it is "(np.inf,np.inf)", split_mode and split_param have no effect')
parser.add_argument('--word_tokenizer_name', type=str, default='simple_word_tokenizer',
                    help='name of the word tokenizer we use to split text into words and count words')
parser.add_argument('--split_mode', type=str, default='original',
                    choices=['original', 'partition_by_n_snippets', 'partition_by_len_snippets', 'all_snippets'],
                    help='how to aggregate paragraphs and/or sentences to form snippets of a passage; if it is '
                         '"original" (i.e., use each original paragraph and/or sentence), split_param has no effect')
parser.add_argument('--split_param', type=str, default='None',
                    help='parameters for a given split_mode, which will be parsed as None or an integer or a tuple')

parser.add_argument('--text_type', type=str, choices=['original', 'preprocessed', 'unified_passage'],
                    help='whether to use 1) original queries, passages, answers or 2) preprocessed queries, passages, '
                         'answers or 3) preprocessed queries, preprocessed unified passages, preprocessed answers')


def get_files_in_folder(root, start_with=None, end_with=None, not_start_with=None, not_end_with=None,
                        return_full_path=False, keep_ext=False, recursively=True):
    """
    Get file names of all the files in a given folder, where you can also
    (1) add some requirements to the file names
    (2) decide whether you want to get full paths of all the files
    (3) decide whether you want file names to have file extensions
    (4) decide whether you want to do this recursively for all the sub-folders

    :param root: path of the folder
    :param start_with: string or a tuple of strings that the file name must start with
    :param end_with: string or a tuple of strings that the file name must end with
    :param not_start_with: string or a tuple of strings that the file name must not start with
    :param not_end_with: string or a tuple of strings that the file name must not end with
    :param return_full_path: whether you want to get full paths of all the files
    :param keep_ext: whether you want file names to have file extensions
    :param recursively: whether you want to do this recursively for all the sub-folders
    :return: a list of file names, or a list of (file name, file path) tuple
    """
    all_file_path = []
    for path, _, files in os.walk(root):
        all_file_path.extend([(file, path) for file in files])
        if not recursively:
            break

    all_file_eligible = {file_path[0]: True for file_path in all_file_path}
    for f in all_file_eligible:
        is_start_with = f.startswith(tuple(start_with)) if (start_with is not None) else True
        is_end_with = f.endswith(tuple(end_with)) if (end_with is not None) else True
        is_not_start_with = (not f.startswith(tuple(not_start_with))) if (not_start_with is not None) else True
        is_not_end_with = (not f.endswith(tuple(not_end_with))) if (not_end_with is not None) else True
        if not (is_start_with and is_end_with and is_not_start_with and is_not_end_with):
            all_file_eligible[f] = False

    if not return_full_path:
        if not keep_ext:
            return [os.path.splitext(file_path[0])[0] for file_path in all_file_path if all_file_eligible[file_path[0]]]
        else:
            return [file_path[0] for file_path in all_file_path if all_file_eligible[file_path[0]]]
    else:
        if not keep_ext:
            return [(os.path.splitext(file_path[0])[0], file_path[1]) for file_path in all_file_path if
                    all_file_eligible[file_path[0]]]
        else:
            return [(file_path[0], file_path[1]) for file_path in all_file_path if all_file_eligible[file_path[0]]]


def download_from_s3(root_path, s3_region, aws_access_key_id, aws_secret_access_key, bucket_name,
                     folder='', filename_prefix=''):
    """
    Download all the files recursively from a given folder in an S3 bucket

    :param root_path: local root path
    :param s3_region: region of the S3 bucket
    :param aws_access_key_id: aws access key id of the S3 bucket
    :param aws_secret_access_key: aws secret access key of the S3 bucket
    :param bucket_name: bucket name of the S3 bucket
    :param folder: folder in the S3 bucket we want to donwload files from
                  (by default, we download the files that are in the bucket but not in any sub-folders of the bucket)
    :param filename_prefix: we only download the files whose file names start with filename_prefix
    :return: local path where the downloaded files are saved
    """
    # The second argument of "os.path.join" should not start with '/', so we make the following assertion
    # Note that "s3_path" and "filename" will never start with '/':
    # if the file is in a folder "a/b/c" of the S3 bucket, s3_path = 'a/b/c'
    # if the file is not in any folder of the S3 bucket, s3_path = ''
    assert not folder.startswith('/')

    s3 = boto3.resource('s3', region_name=s3_region,
                        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    bucket = s3.Bucket(bucket_name)

    count = 0
    for s3_object in bucket.objects.all():
        s3_path, filename = os.path.split(s3_object.key)
        if (not (s3_path.rstrip('/') + '/').startswith(folder.rstrip('/') + '/')) or (len(filename) == 0) or (not filename.startswith(filename_prefix)):
            continue
        local_path = os.path.join(root_path, s3_path)
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        bucket.download_file(s3_object.key, os.path.join(local_path, filename))
        count += 1
        if count % 1000 == 0:
            print('Downloading files from s3 folder {0:} ({1:d} files downloaded)'.format(s3_path, count))

    if folder != '':
        print('\n{0:d} files downloaded from the folder {1:} in S3 bucket {2:}\n'.format(count, folder, bucket_name))
    else:
        print('\n{0:d} files downloaded from the S3 bucket {1:}\n'.format(count, bucket_name))

    return os.path.join(root_path, folder)


def upload_to_s3(folder, root_path, s3_region, aws_access_key_id, aws_secret_access_key, bucket_name, filename_prefix=None):
    s3 = boto3.resource('s3', region_name=s3_region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    all_name_path = get_files_in_folder(os.path.join(root_path, folder), start_with=filename_prefix, return_full_path=True, keep_ext=True, recursively=True)
    for name, path in all_name_path:
        s3.meta.client.upload_file(os.path.join(path, name), bucket_name, os.path.join(os.path.relpath(path, root_path), name))
    print('{0:d} files in {1:} uploaded to the folder {2:} in S3 bucket {3:}'.format(len(all_name_path), os.path.join(root_path, folder), folder, bucket_name))


def get_data(data_path, excluded_fullname_prefix=None):
    """
    Collect all txt files that are in json format in a given folder, and put them in a pandas dataframe

    :param data_path: folder from which we want to collect txt files
    :param excluded_fullname_prefix: a string or a tuple of full names or prefixes of the files you want to exclude
    :return: a pandas dataframe containing all the data
    """
    all_data_name = get_files_in_folder(data_path, end_with='.txt', not_start_with=excluded_fullname_prefix,
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

    # Put data in a dataframe, make sure all entries in the dataframe are strings, fill None by '', and rename columns
    data = pd.DataFrame(all_data, dtype=str).fillna('').rename(columns={
        'question': constants.RANKING_INPUT_QUERY_NAME,
        'content': constants.RANKING_INPUT_DOCUMENT_NAME,
        'bool_train': 'ignore',
        'bool_answer': 'display'})
    data['source_path'] = all_source_path

    return data


def is_valid_str(x):
    # Check whether x is a string that contains at least a letter or a number. We call such a string a valid string
    return isinstance(x, str) and (re.search(r'(?ias)[a-zA-Z0-9]', x) is not None)


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
    assert all((isinstance(k, int) and (k >= 1) and is_valid_str(v)) for k, v in collection.items())
    assert all(is_valid_str(i) for i in items_to_add)

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


class CustomizedLanguageVars(pkt.PunktLanguageVars):
    # Overwrite the _period_context_fmt of pkt.PunktLanguageVars such that NLTK's punkt sentence tokenizer will
    # preserve all the characters ], ), }, ', ", \s at the end of a sentence
    _period_context_fmt = r"""
        \S*
        %(SentEndChars)s
        [])}'"\s]*
        (?=(?P<after_tok>
            %(NonWord)s
            |
            (?P<next_tok>\S+)
        ))"""


def merge_invalid_short_str(all_s, thres):
    # For each string in all_s, merge it with the string before or after it, if it is invalid or shorter than a threshold
    # More specifically, if the invalid or short string ends with any line boundary, merge it with the string before it;
    # otherwise merge it with the string after it
    if len(all_s) <= 1:
        return all_s

    next_must_be_merged_to_last = False
    result = ['']
    for s in all_s:
        ends_with_line_boundary = (re.search(r'(?ias)[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029] *$', s) is not None)
        if next_must_be_merged_to_last or (not is_valid_str(s)) or ((len(s) < thres) and ends_with_line_boundary):
            result[-1] += s
        else:
            result.append(s)
        next_must_be_merged_to_last = is_valid_str(s) and ((len(s) < thres) and (not ends_with_line_boundary))

    if result[-1] == all_s[-1] and next_must_be_merged_to_last:
        # If result[-1] == all_s[-1] and all_s[-1] should be merged with the string after it (which is impossible since
        # it is the last one), we merge all_s[-1] with the string before it
        # Also note that it is impossible to have len(result) == 1 and next_must_be_merged_to_last == True at the same
        # time, so as long as the above conditions hold, then len(result) >= 2
        result[-2] += result[-1]
        result = result[:-1]

    if len(result) == 1 or is_valid_str(result[0]):
        return result
    else:
        result[1] = result[0] + result[1]
        return result[1:]


def sentence_tokenizer(text, n_char_lb=10):
    """
    Split a text into sentences, where
    (1) No character is dropped or duplicated (i.e. the sentences form an exact partition of the text)
    (2) Every sentence is guaranteed to be valid, if the text is valid
    (3) We try our best to make the length of every sentence >= n_char_lb, if the length of text is >= n_char_lb

    :param text: text to split (a string)
    :param n_char_lb: minimum acceptable number of characters in the sentences split from the text
    :return: a list of sentences split from the text
    """
    if (not is_valid_str(text)) or (len(text) < n_char_lb):
        return [text]

    lines = text.splitlines(keepends=True)  # punkt won't break sentences separated by \n, etc., so we split by splitlines
    sentences = list()
    for line in lines:
        if is_valid_str(line) and (len(line) >= n_char_lb):
            s = pkt.PunktSentenceTokenizer(lang_vars=CustomizedLanguageVars()).tokenize(line)
            s[-1] += re.search(r'(?ius)\s*$', line).group()  # punkt still drops trailing r'(?ius)\s', so we append them
        else:
            s = [line]
        sentences.extend(s)
    result = merge_invalid_short_str(sentences, thres=n_char_lb)

    assert ''.join(result) == text
    return result


def paragraph_tokenizer(text, n_char_lb=10):
    """
    Split a text into paragraphs, where
    (1) No character is dropped or duplicated (i.e. the paragraphs form an exact partition of the text)
    (2) Every paragraph is guaranteed to be valid, if the text is valid
    (3) We try our best to make the length of every paragraph >= n_char_lb, if the length of text is >= n_char_lb

    :param text: text to split (a string)
    :param n_char_lb: minimum acceptable number of characters in the paragraphs split from the text
    :return: a list of paragraphs split from the text
    """
    if (not is_valid_str(text)) or (len(text) < n_char_lb):
        return [text]

    try:
        paragraphs = nltk.tokenize.TextTilingTokenizer().tokenize(text)
        result = merge_invalid_short_str(paragraphs, thres=n_char_lb)
    except ValueError:
        result = [text]

    assert ''.join(result) == text
    return result


def text_tokenizer(text, split_granularity, word_tokenizer):
    """
    Split a text into paragraphs or sentences according to split_granularity, a tuple (n_word_to_para, n_word_to_sent),
    in the following two steps:

    Step 1: For the text,
            if its number of words < n_word_to_para, keep it; otherwise split it into paragraphs
    Step 2: For the text or each paragraph obtained in Step 1,
            if its number of words < n_word_to_sent, keep it; otherwise split it into sentences

    All possible combinations of n_word_to_para and n_word_to_sent, as well as descriptions are as follows, where
    (0, np.inf) means the value is between 0 and np.inf:

    n_word_to_para  n_word_to_sent  description
    ====================================================================================================================
    np.inf          np.inf          keep the original text
    np.inf          0               split text into sentences
    np.inf          (0, np.inf)     split text into sentences if text has many words
    --------------------------------------------------------------------------------------------------------------------
    0               np.inf          split text into paragraphs
    0               0               split text into paragraphs, and split each paragraph into sentences
    0               (0, np.inf)     split text into paragraphs, and split a paragraph into sentences if the paragraph has many words
    --------------------------------------------------------------------------------------------------------------------
    (0, np.inf)     np.inf          split text into paragraphs if text has many words
    (0, np.inf)     0               split text into paragraphs if text has many words, and split text or each paragraph into sentences
    (0, np.inf)     (0, np.inf)     split text into paragraphs if text has many words, and split text or a paragraph
                                    into sentences if the text or paragraph has many words

    :param text: text to split (a string)
    :param split_granularity: a tuple (n_word_to_para, n_word_to_sent) used to specify the granularity
    :param word_tokenizer: a function to tokenize text into words such that we can count the number of words in the text
    :return: a list of paragraphs and/or sentences split from the text, or the list [text]
    """
    assert isinstance(split_granularity, tuple)

    if len(word_tokenizer(text)) < split_granularity[0]:
        result_step1 = [text]
    else:
        result_step1 = paragraph_tokenizer(text)

    result_step2 = list()
    for i in result_step1:
        if len(word_tokenizer(i)) < split_granularity[1]:
            result_step2.extend([i])
        else:
            result_step2.extend(sentence_tokenizer(i))

    assert ''.join(result_step2) == text
    return result_step2


def aggregator(text_tokenized, split_mode, split_param):
    """
    This function provides four ways to aggregate paragraphs and/or sentences split from a text to form several snippets
    Examples for each of these ways are as follows, where 'x1', 'x2', etc. can either be a paragraph or a sentence

    (1) Partition the text into a given number, e.g., 3 of snippets, where we try to make every snippet consist of an
        equal number of paragraphs or sentences
    aggregator(['x1', 'x2', 'x3', 'x4', 'x5'], 'partition_by_n_snippets', 3) -> ['x1 x2', 'x3 x4', 'x5']

    (2) Partition the text into snippets, where we try to make every snippet consist of a given number, e.g., 3
        of paragraphs or sentences
    aggregator(['x1', 'x2', 'x3', 'x4', 'x5', 'x6'], 'partition_by_len_snippets', 3) -> ['x1 x2 x3', 'x4 x5 x6']

    (3) Generate all possible snippets (spans) whose lengths are between a lower bound and an upper bound, e.g., 2 and 3
    aggregator(['x1', 'x2', 'x3'], 'all_snippets', (1, np.inf)) -> ['x1', 'x1 x2', 'x1 x2 x3', 'x2', 'x2 x3', 'x3']
    aggregator(['x1', 'x2', 'x3'], 'all_snippets', (2, 3)) -> ['x1 x2', 'x1 x2 x3', 'x2 x3']
    aggregator(['x1', 'x2', 'x3'], 'all_snippets', (2, 2)) -> ['x1 x2', 'x2 x3']
    aggregator(['x1', 'x2', 'x3'], 'all_snippets', (4, 5)) -> []

    (4) Use the original paragraphs or sentences, without aggregating them
    aggregator(['x1', 'x2', 'x3'], 'original', None) -> ['x1', 'x2', 'x3']

    :param text_tokenized: a list of paragraphs and/or sentences split from a text, which we will aggregate in some way
    :param split_mode: the way in which we will aggregate the paragraphs and/or sentences
    :param split_param: parameters for paragraph and sentence aggregation
    :return: a list of snippets (aggregated paragraphs and/or sentences)
    """
    assert isinstance(text_tokenized, list)
    l = len(text_tokenized)
    if l <= 1:
        return text_tokenized

    if split_mode == 'partition_by_n_snippets':
        assert isinstance(split_param, int) and (l >= split_param >= 1)
        start_end_index = utils.get_start_end_index(l, split_param)
        return [''.join(text_tokenized[start:end]) for start, end in start_end_index]
    elif split_mode == 'partition_by_len_snippets':
        assert isinstance(split_param, int) and (l >= split_param >= 1)
        return [''.join(text_tokenized[i:(i + split_param)]) for i in range(0, l, split_param)]
    elif split_mode == 'all_snippets':
        assert isinstance(split_param, tuple) and (1 <= split_param[0] <= split_param[1]) and (split_param[0] != np.inf)
        return [''.join(text_tokenized[i:(i + j)]) for i in range(0, l)
                for j in range(split_param[0], min(l - i, split_param[1]) + 1)]
    else:
        assert split_mode == 'original'
        return text_tokenized


def split_text(text, split_granularity, word_tokenizer, split_mode, split_param):
    """
    To split a text into several snippets, we will follow two steps:
    (1) split it into several paragraphs and/or sentences using the function "text_tokenizer"
    (2) aggregate these paragraphs and/or sentences in some way using the function "aggregator"
    This function is a combination of these two steps

    :param text: text to split (a string)
    :param split_granularity: granularity we will split the text into, in the first step
    :param word_tokenizer: word tokenizer we use to split text into words and count words, in the first step
    :param split_mode: the way in which we will aggregate the paragraphs and/or sentences, in the second step
    :param split_param: parameters for paragraph and sentence aggregation, in the second step
    :return: a list of snippets
    """
    return aggregator(text_tokenizer(text, split_granularity, word_tokenizer), split_mode, split_param)


def split_text_series_to_df(s, text_idx, split_granularity, word_tokenizer, split_mode, split_param):
    """
    This function accepts a pandas series (like the one shown below) as input

    col1     |  v1
    col2     |  v2
    ...      |  ...
    text_idx |  text

    (1) If "split_text" returns [snippet1, ..., snippetn], this function returns a pandas dataframe shown below as output

    col1  col2 ... text_idx
    ------------------------
    v1    v2   ... snippet1
    v1    v2   ... snippet2
               ...
    v1    v2   ... snippetn

    (2) If "split_text" returns [], this function returns a pandas dataframe which is equal to the original series

    col1  col2 ... text_idx
    ------------------------
    v1    v2   ... text

    :param s: a pandas series, whose s.loc[text_idx] is the text to split
    :param text_idx: index name of the text
    :param split_granularity: granularity we will split the text into, in the first step of "split_text"
    :param word_tokenizer: word tokenizer we use to split text into words and count words, in the first step of "split_text"
    :param split_mode: the way in which we will aggregate the paragraphs and/or sentences, in the second step of "split_text"
    :param split_param: parameters for paragraph and sentence aggregation, in the second step of "split_text"
    :return: a pandas dataframe
    """
    text_split = split_text(s.loc[text_idx], split_granularity, word_tokenizer, split_mode, split_param)
    df = pd.DataFrame(columns=s.index)
    if len(text_split) > 0:
        df = df.append([s] * len(text_split), ignore_index=True)
        df[text_idx] = text_split
    else:
        df = df.append([s], ignore_index=True)
    return df


def make_col_unique(df, col):
    # Helper function: set all values in the column of a dataframe as "no", if any value in this columns is "no"
    if (df[col] == 'no').any():
        df[col] = 'no'
    return df


def add_remove_neg(df, r_neg_pos, pid_col, passage_col, label_col, collection):
    # Helper function: add negative passages (and their pids) from the passage collection, or remove negative passages
    # such that, in the dataframe, the number of negative labels (0) / the number of positive labels (1) = r_neg_pos

    # We only change the pid_col, passage_col and label_col for the added passages, but the passage_display_col is not
    # changed accordingly (i.e., it may not match the added passages), as we assume this column will be dropped later
    n_neg_original = (df[label_col] == 0).sum()
    n_pos_original = (df[label_col] == 1).sum()
    neg_pid = sorted(list(set(collection.keys()) - set(df[pid_col])))

    if r_neg_pos == np.inf:
        n_neg_add_remove = len(neg_pid)
    else:
        assert r_neg_pos >= 0
        if n_pos_original == 0:
            return df
        n_neg_add_remove = int(r_neg_pos * n_pos_original) - n_neg_original

    if n_neg_add_remove < 0:
        index_remove = df.loc[df[label_col] == 0, :].sample(frac=1, random_state=0).index.values[n_neg_add_remove:]
        result = df.drop(index=index_remove).reset_index(drop=True)
    elif n_neg_add_remove > 0:
        np.random.seed(0)
        neg_added = pd.DataFrame([df.iloc[0, :]] * n_neg_add_remove)
        neg_added[pid_col] = np.random.choice(neg_pid, size=n_neg_add_remove, replace=(n_neg_add_remove > len(neg_pid)))
        neg_added[passage_col] = [collection[i] for i in neg_added[pid_col]]
        neg_added[label_col] = 0
        result = df.append(neg_added, ignore_index=True)
    else:
        result = df

    if r_neg_pos == np.inf:
        assert sorted(result[pid_col].to_list()) == sorted(list(collection.keys()))
    else:
        assert math.isclose((result[label_col] == 0).sum() / (result[label_col] == 1).sum(), r_neg_pos)

    return result


def get_answer_label(df, answer_col, sent_comb_col, label_col):
    # Helper function: if a sentence combination is the shortest among all the sentence combinations that contain
    # answer as a substring, then label it as 1 and label all the other sentence combinations as 0
    assert len(set(df[answer_col])) == 1
    correct_sent_comb_mask = df.apply(lambda x: x.loc[answer_col] in x.loc[sent_comb_col], axis=1)
    len_sent_comb_mask = (df[sent_comb_col].map(len) == df.loc[correct_sent_comb_mask, sent_comb_col].map(len).min())
    df[label_col] = (correct_sent_comb_mask & len_sent_comb_mask).astype(int)
    return df


def is_almost_same(text1, text2):
    text1_split = text1.split()
    text2_split = text2.split()
    thres_ndiff = min(int(0.1 * max(len(text1_split), len(text2_split))), 10)

    if abs(len(text1_split) - len(text2_split)) > thres_ndiff:
        return False

    plus_count = -1
    minus_count = -1
    for s in difflib.unified_diff(text1_split, text2_split, n=0):
        plus_count += s.startswith('+')
        minus_count += s.startswith('-')
        if max(plus_count, minus_count) > thres_ndiff:
            return False

    return True


def is_in(text1, text2):
    if not ((text1 in text2) or (text2 in text1)):
        return False
    else:
        text1_split = text1.split()
        text2_split = text2.split()
        return min(len(text1_split), len(text2_split)) >= min(0.5 * max(len(text1_split), len(text2_split)), 10)


def get_adjacency_matrix(start, end, s):
    n = len(s)
    adjacency_matrix_start_end = [[False for _ in range(n)] for _ in range(end - start)]
    for i in range(start, end):
        for j in range(i):
            adjacency_matrix_start_end[i - start][j] = is_in(s[i], s[j]) or is_almost_same(s[i], s[j])
    return adjacency_matrix_start_end


def get_start_end_index_arithmetic_progression(l, n, verbose=False):
    assert l >= 2
    targeted_group_sum = max(int((((l - 1) * l) / 2) / n), 1)
    sep = list()

    running_sum = 0
    for i in range(l):
        running_sum += i
        if running_sum >= targeted_group_sum:
            sep.append(i)
            running_sum = 0

    start_end_index = [(0, sep[0] + 1)] + [(sep[j] + 1, sep[j + 1] + 1) for j in range(len(sep) - 1)]
    if sep[-1] + 1 < l:
        start_end_index += [(sep[-1] + 1, l)]

    actual_group_sum = [sum(list(range(start, end))) for start, end in start_end_index]
    assert (start_end_index[0][0] == 0) and (start_end_index[-1][1] == l) and (sum(actual_group_sum) == ((l - 1) * l) / 2)
    if verbose:
        print('Arithmetic progression 0, 1, ..., {0:d} are partitoned into {1:d} groups with actual group sum {2:} '
              '(targeted group sum: {3:d}) and start and end indices {4:}\n'.format(l - 1, len(start_end_index),
              ', '.join([str(s) for s in actual_group_sum]), targeted_group_sum, ', '.join([str(idx) for idx in start_end_index])))

    return start_end_index


def get_unified_text(text_df, input_col):
    s = text_df[input_col]
    n = len(s)

    n_cpu = psutil.cpu_count(logical=True)
    all_start_end_index = get_start_end_index_arithmetic_progression(n, n_cpu)
    input_list = [(start, end, s.to_list()) for start, end in all_start_end_index]
    tic = time.time()
    with Pool(len(input_list)) as pool:
        adjacency_matrix_list = pool.starmap(get_adjacency_matrix, input_list)
    print('A {0:d} * {0:d} adjacency matrix for passage unification created using {1:d} processes in {2:.2f} second(s)\n'.format(
        n, len(input_list), time.time() - tic))

    adjacency_matrix = list()
    for l in adjacency_matrix_list:
        adjacency_matrix.extend(l)
    assert (len(adjacency_matrix) == n) and all(len(row) == n for row in adjacency_matrix)

    for i in range(n):
        adjacency_matrix[i][i] = True
        for j in range(i + 1, n):
            adjacency_matrix[i][j] = adjacency_matrix[j][i]

    graph = {i: [j for j in range(n) if adjacency_matrix[i][j] and (j != i)] for i in range(n)}
    visited = [None for _ in range(n)]
    component_id = 0
    path = list()
    sorted_node_id = sorted(list(range(n)), key=lambda i: len(graph[i]), reverse=True)
    for node_id in sorted_node_id:
        if visited[node_id] is None:
            get_connected_component_dfs(graph, node_id, visited, component_id, path, max_search_depth=1)
            component_id += 1
    assert all(i is not None for i in visited)
    print('Information about passage groups after passage unification:')
    print(pd.Series(visited).value_counts().value_counts().sort_index(ascending=False).reset_index(drop=False).rename(
        columns={'index': 'Size of group', 0: 'Count of groups with this size'}))

    unified_text_dict = dict()
    for i in sorted(list(set(visited))):
        s_i = s.loc[np.array(visited) == i].to_frame(name='passage_s_i').copy()
        s_i['len_s_i'] = s_i['passage_s_i'].map(lambda x: len(x.split()))
        s_i['coverage_s_i'] = s_i['passage_s_i'].map(lambda x: sum(z in x for z in s_i['passage_s_i']))
        unified_text_dict[i] = s_i.sort_values(by=['coverage_s_i', 'len_s_i'], ascending=False)['passage_s_i'].iloc[0]

    output_col = input_col + '_unified'
    text_df[output_col] = [unified_text_dict[i] for i in visited]

    return text_df, output_col


def get_connected_component_dfs(graph, node_id, visited, component_id, path, max_search_depth):
    path.append(node_id)
    visited[node_id] = component_id
    for next_node_id in graph[node_id]:
        if (visited[next_node_id] is None) and (len(path) <= max_search_depth):
            get_connected_component_dfs(graph, next_node_id, visited, component_id, path, max_search_depth)
    path.pop()


def data_preparation(args):
    suffix = '{0:}-{1:}-{2:}-{3:}-{4:}-{5:}'.format(
        args.n_test, args.r_neg_pos, args.split_granularity, args.split_mode, args.split_param, args.text_type)

    qid_col = constants.RANKING_INPUT_QUERY_ID
    query_col = constants.RANKING_INPUT_QUERY_NAME
    pid_col = constants.RANKING_INPUT_DOCUMENT_ID
    passage_col = constants.RANKING_INPUT_DOCUMENT_NAME
    label_col = constants.RANKING_INPUT_LABEL_NAME
    answer_col = 'answer'
    ignore_col = 'ignore'
    display_col = 'display'
    triplet_col = 'qa_triplet'
    user_col = 'user'
    url_col = 'url'
    source_path_col = 'source_path'
    passage_display_col = 'passage_display'
    sent_comb_col = 'sent_comb'

    os.makedirs(args.root_path, exist_ok=False)

    nltk.download('punkt', download_dir=args.root_path)
    nltk.download('stopwords', download_dir=args.root_path)
    if args.root_path not in nltk.data.path:
        nltk.data.path = [args.root_path] + nltk.data.path

    word_tokenizer = WordTokenizerFactory.create_word_tokenizer(args.word_tokenizer_name)

    # Download raw QA data from S3 buckets if args.data_path or args.data_new_path is not specified ####################
    if args.data_path is None:
        # Raw QA data where urls and passages are given have not been downloaded from the S3 bucket
        data_path = download_from_s3(os.path.join(args.root_path, args.bucket_name), args.s3_region,
                                     args.aws_access_key_id, args.aws_secret_access_key, args.bucket_name)
    else:
        # Raw QA data where urls and passages are given have been downloaded from the S3 bucket and
        # saved to args.data_path
        data_path = args.data_path

    if args.data_new_path is None:
        # Raw QA data where urls and passages are added by annotators have not been downloaded from the S3 bucket
        data_new_path = download_from_s3(os.path.join(args.root_path, args.bucket_new_name), args.s3_region,
                                         args.aws_access_key_id, args.aws_secret_access_key, args.bucket_new_name)
    else:
        # Raw QA data where urls and passages are added by annotators have been downloaded from the S3 bucket and
        # saved to args.data_new_path
        data_new_path = args.data_new_path

    # Clean the raw data, put them in the desired format and extract the part we will use ##############################
    data = get_data(data_path)
    data_new = get_data(data_new_path, excluded_fullname_prefix=('count.txt', ))
    data[triplet_col] = 'qat'
    data_new[ignore_col] = 'no'
    data_all = data.append(data_new, ignore_index=True, sort=False)
    n0 = data_all.shape[0]
    print('\n{:d} records loaded\n'.format(n0))

    # Extract the records with ignore = no -----------------------------------------------------------------------------
    assert data_all[ignore_col].isin({'yes', 'no'}).all()
    data_all = data_all.loc[data_all[ignore_col] == 'no', :].reset_index(drop=True)
    n1 = data_all.shape[0]
    print('{:d} records with ignore = yes dropped\n'.format(n0 - n1))

    # Make sure all queries and answers are valid ----------------------------------------------------------------------
    assert data_all.loc[:, [query_col, answer_col]].applymap(is_valid_str).all().all()

    # Get preprocessed queries, passages, answers, which will be used to drop duplications, etc. even if we may not need
    # them in the final data -------------------------------------------------------------------------------------------
    data_all, query_preprocessed_col = english_preprocessor.preprocess(data_all, query_col)
    data_all, passage_preprocessed_col = english_preprocessor.preprocess(data_all, passage_col)
    data_all, answer_preprocessed_col = english_preprocessor.preprocess(data_all, answer_col)

    # Make each answer have unique display label -----------------------------------------------------------------------
    assert data_all[display_col].isin({'yes', 'no'}).all()
    n_display_answer = (data_all[display_col] == 'yes').sum()
    data_all = data_all.groupby(data_all[answer_preprocessed_col]).apply(lambda x: make_col_unique(x, display_col))
    nondisplayable_answer = data_all.loc[data_all[display_col] == 'no', answer_col].unique()
    print('\n{:d} answers that were considered as displayable are now considered as nondisplayable\n'.format(
        n_display_answer - (data_all[display_col] == 'yes').sum()))

    # Extract the records with qa_triplet = qat and make sure all extracted records have valid passages
    # that contain answers as substrings -------------------------------------------------------------------------------
    assert data_all[triplet_col].isin({'qat', 'qa'}).all()

    answer_in_passage_mask = data_all[passage_col].map(is_valid_str) & data_all.apply(
        lambda x: x.loc[answer_col] in x.loc[passage_col], axis=1)

    wrong_qa_mask = (data_all[triplet_col] == 'qa') & answer_in_passage_mask
    data_all.loc[wrong_qa_mask, triplet_col] = 'qat'
    print('{:d} records that were marked as "qa" are now marked as "qat" because their passages are valid and contain '
          'answers as substrings'.format(wrong_qa_mask.sum()))

    wrong_qat_mask = (data_all[triplet_col] == 'qat') & (~answer_in_passage_mask)
    data_all.loc[wrong_qat_mask, triplet_col] = 'qa'
    print('{:d} records that were marked as "qat" are now marked as "qa" because their passages are invalid or do not '
          'contain answers as substrings'.format(wrong_qat_mask.sum()))

    data_all = data_all.loc[data_all[triplet_col] == 'qat', :].reset_index(drop=True)
    n2 = data_all.shape[0]
    print('{:d} records with qa_triplet = qa dropped\n'.format(n1 - n2))

    # Drop the records whose preprocessed passages contain no more than five words -------------------------------------
    short_passage_mask = data_all[passage_preprocessed_col].map(lambda x: len(word_tokenizer(x)) <= 5)
    data_all = data_all.loc[~short_passage_mask, :].reset_index(drop=True)
    n3 = data_all.shape[0]
    print('{:d} records whose preprocessed passages contain no more than five words dropped\n'.format(n2 - n3))

    # Drop the records with duplicated preprocessed queries ------------------------------------------------------------
    duplicated_mask = data_all[query_preprocessed_col].duplicated()
    data_all = data_all.loc[~duplicated_mask, :].reset_index(drop=True)
    assert data_all.groupby(query_preprocessed_col).apply(lambda x: x[query_col].nunique() == 1).all()
    assert data_all.groupby(query_col).apply(lambda x: x[query_preprocessed_col].nunique() == 1).all()
    n4 = data_all.shape[0]
    print('{:d} records with duplicated preprocessed queries dropped\n'.format(n3 - n4))

    # Split the passages into snippets or keep the original passages, depending on args.split_granularity --------------
    n_words_before_distr = data_all.groupby(passage_col).apply(lambda x: len(word_tokenizer(x[passage_col].iloc[0])))
    plot_distr(n_words_before_distr, os.path.join(args.root_path, args.n_words_before_distr_plot_name + '-' + suffix +
               '.pdf'), 'Distribution of number of words per passage before splitting (' + suffix + ')',
               thres_interested=(445, 512))

    tic = time.time()
    data_all = pd.concat(list(data_all.apply(lambda x: split_text_series_to_df(
        x, passage_col, eval(args.split_granularity), word_tokenizer, args.split_mode, eval(args.split_param)),
        axis=1))).reset_index(drop=True)
    print('{0:d} records split to {1:d} records in {2:.2f} second(s) ({3:})\n'.format(
        n4, data_all.shape[0], time.time() - tic, suffix))

    n_words_after_distr = data_all.groupby(passage_col).apply(lambda x: len(word_tokenizer(x[passage_col].iloc[0])))
    plot_distr(n_words_after_distr, os.path.join(args.root_path, args.n_words_after_distr_plot_name + '-' + suffix +
               '.pdf'), 'Distribution of number of words per passage after splitting (' + suffix + ')',
               thres_interested=(445, 512))

    snippet_distr = data_all.groupby(query_col).apply(len)
    plot_distr(snippet_distr, os.path.join(args.root_path, args.snippet_distr_plot_name + '-' + suffix + '.pdf'),
               'Distribution of number of snippets per query (' + suffix + ')')

    # Add ranking labels: label = 1 if the answer is a substring of the passage or snippet; label = 0 otherwise --------
    data_all[label_col] = data_all.apply(lambda x: x.loc[answer_col] in x.loc[passage_col], axis=1).astype(int)

    rank_label_distr = data_all.groupby(query_col).apply(lambda x: x[label_col].sum())
    plot_distr(rank_label_distr, os.path.join(args.root_path, args.rank_label_distr_plot_name + '-' + suffix + '.pdf'),
               'Distribution of number of label 1 per query in all passage ranking data (' + suffix + ')')

    # Decide query_col_used, passage_col_used, answer_col_used by text_type (unify the passages if needed) -------------
    if (args.text_type == 'preprocessed') or (args.text_type == 'unified_passage'):
        data_all, passage_preprocessed_col = english_preprocessor.preprocess(data_all, passage_col)
        print('Number of unique passages reduced from {0:d} to {1:d} by preprocessing (no records dropped)\n'.format(
            data_all[passage_col].nunique(), data_all[passage_preprocessed_col].nunique()))

        # Drop queries if 1) any of their preprocessed queries, passages, answers doesn't contain any words or is invalid
        # or 2) the preprocessed answer is no longer contained in the preprocessed passage
        n_query_before = data_all[query_col].nunique()
        data_all = data_all.groupby(query_col).filter(lambda x:
            x.loc[:, [query_preprocessed_col, passage_preprocessed_col, answer_preprocessed_col]].applymap(lambda z:
                (len(word_tokenizer(z)) > 1) and is_valid_str(z)).all().all() and
            x.loc[x[label_col] == 1, :].apply(lambda z:
                z.loc[answer_preprocessed_col] in z.loc[passage_preprocessed_col], axis=1).all().all()
        ).reset_index(drop=True)
        print("{:d} queries dropped since 1) any of their preprocessed queries, passages, answers doesn't contain any words "
              "or is invalid or 2) the preprocessed answer is no longer contained in the preprocessed passage\n".format(
               n_query_before - data_all[query_col].nunique()))

        additional_col_types = {query_preprocessed_col: str, passage_preprocessed_col: str, answer_preprocessed_col: str}
        query_col_used, passage_col_used, answer_col_used = query_preprocessed_col, passage_preprocessed_col, answer_preprocessed_col

        if args.text_type == 'unified_passage':
            data_all, passage_preprocessed_unified_col = get_unified_text(data_all, passage_preprocessed_col)
            print('Number of unique passages further reduced from {0:d} to {1:d} by unification (no records dropped)\n'.format(
                data_all[passage_preprocessed_col].nunique(), data_all[passage_preprocessed_unified_col].nunique()))

            additional_col_types.update({passage_preprocessed_unified_col: str})
            passage_col_used = passage_preprocessed_unified_col
    else:
        additional_col_types = dict()
        query_col_used, passage_col_used, answer_col_used = query_col, passage_col, answer_col

    # Add passage_display: passage_display = 'no' if a passage or snippet contains any nondisplayable answer
    # as a substring; passage_display = 'yes' otherwise ----------------------------------------------------------------
    data_all[passage_display_col] = data_all[passage_col].map(lambda x:
        any((ans in x) for ans in nondisplayable_answer)).replace({True: 'no', False: 'yes'})
    data_all = data_all.groupby(data_all[passage_col_used]).apply(lambda x: make_col_unique(x, passage_display_col))

    # Assign qid and pid, and create query collection and passage collection -------------------------------------------

    # Assign qid to queries, add these queries to query collection, and save it for future use
    print('Query collection:')
    query_id, query_collection = get_idx_collection(data_all[query_col_used].to_list(), dict())
    data_all[qid_col] = query_id
    with open(os.path.join(args.root_path, args.query_collection_name + '-' + suffix + '.json'), 'w') as f:
        json.dump(query_collection, f)

    # Assign pid to displayable passages, add these passages to passage collection, and save it for future use
    print('\nDisplayable passage collection:')
    display_passage_id, display_passage_collection = get_idx_collection(
        data_all.loc[data_all[passage_display_col] == 'yes', passage_col_used].to_list(), dict())
    data_all.loc[data_all[passage_display_col] == 'yes', pid_col] = display_passage_id
    with open(os.path.join(args.root_path, args.passage_collection_name + '-' + suffix + '.json'), 'w') as f:
        json.dump(display_passage_collection, f)

    # Assign pid to nondisplayable passages, but do not add them to the above passage collection
    print('\nNondisplayable passage collection:')
    nondisplay_passage_id, _ = get_idx_collection(
        data_all.loc[data_all[passage_display_col] == 'no', passage_col_used].to_list(), display_passage_collection, False)
    data_all.loc[data_all[passage_display_col] == 'no', pid_col] = nondisplay_passage_id

    # Make sure the data have desired data types, and save all passage ranking data to a json file for future use ------
    col_types = {qid_col: int, query_col: str, pid_col: int, passage_col: str, label_col: int, answer_col: str,
                 ignore_col: str, display_col: str, triplet_col: str, passage_display_col: str,
                 user_col: str, url_col: str, source_path_col: str}
    col_types.update(additional_col_types)

    data_all.astype(col_types).reindex(columns=list(col_types.keys())).reset_index(drop=True).to_json(
        os.path.join(args.root_path, args.data_all_name + '-' + suffix + '.json'), orient='columns')
    print('\nFinally, all passage ranking data have {:d} records\n'.format(data_all.shape[0]))

    # Get testing and training data for passage ranking ################################################################

    # Split all passage ranking data into a testing set and a training set by queries, and make sure all queries in
    # the testing set 1) contain at least a label 1 and 2) don't have nondisplayble passages or snippets ---------------
    np.random.seed(0)
    test_query = data_all.groupby(query_col).filter(lambda x:
        (x[label_col].sum() > 0) and (x[passage_display_col] == 'yes').all())[query_col].unique()
    test_query_sampled = np.random.choice(test_query, size=args.n_test, replace=False)
    test_mask = data_all[query_col].isin(test_query_sampled)

    # Get testing data for passage ranking -----------------------------------------------------------------------------
    rank_test = copy.deepcopy(data_all.loc[test_mask & (data_all[label_col] == 1), :].reset_index(drop=True))
    rank_test.reindex(columns=[qid_col, query_col_used, pid_col, passage_col_used]).rename(
        columns={query_col_used: query_col, passage_col_used: passage_col}).to_json(
        os.path.join(args.root_path, args.rank_test_name + '-' + suffix + '.json'), orient='columns')
    print('Data for testing passage ranking model have in total {:d} records\n'.format(rank_test.shape[0]))

    # Get training data for passage ranking ----------------------------------------------------------------------------
    rank_train = copy.deepcopy(data_all.loc[~test_mask, :].reset_index(drop=True))
    print('Data for training passage ranking model have {0:d} negative records, {1:d} positive records and in total {2:d}'
          ' records'.format((rank_train[label_col] == 0).sum(), (rank_train[label_col] == 1).sum(), rank_train.shape[0]))

    if args.r_neg_pos is not None:
        rank_train = rank_train.groupby(query_col).apply(lambda x: add_remove_neg(x, args.r_neg_pos,
            pid_col, passage_col_used, label_col, display_passage_collection)).reset_index(drop=True)
        print('After adding or removing negative records, data for training passage ranking model have {0:d} negative '
              'records, {1:d} positive records and in total {2:d} records\n'.format(
               (rank_train[label_col] == 0).sum(), (rank_train[label_col] == 1).sum(), rank_train.shape[0]))

    rank_train.reindex(columns=[qid_col, query_col_used, pid_col, passage_col_used, label_col]).rename(
        columns={query_col_used: query_col, passage_col_used: passage_col}).to_json(
        os.path.join(args.root_path, args.rank_train_name + '-' + suffix + '.json'), orient='columns')

    # Get testing and training data for answer extraction ##############################################################

    # Get testing data for answer extraction ---------------------------------------------------------------------------
    answer_test = copy.deepcopy(rank_test)
    answer_test.reindex(columns=[qid_col, query_col, pid_col, passage_col, answer_col]).to_json(
        os.path.join(args.root_path, args.answer_test_name + '-' + suffix + '.json'), orient='columns')
    print('Data for testing answer extraction model have in total {:d} records\n'.format(answer_test.shape[0]))

    # Get training data for answer extraction --------------------------------------------------------------------------
    answer_train = copy.deepcopy(rank_train.loc[rank_train[label_col] == 1, :].reset_index(drop=True))

    # Since the answer extraction model is ranking based, we split a passage or a snippet into all possible contiguous
    # sentence combinations, and train a model to rank the sentence combination that can answer the query as the first
    answer_train[sent_comb_col] = answer_train[passage_col]
    tic = time.time()
    answer_train = pd.concat(list(answer_train.apply(lambda x: split_text_series_to_df(
        x, sent_comb_col, (np.inf, 0), word_tokenizer, 'all_snippets', (1, np.inf)),
        axis=1))).reset_index(drop=True)
    print('{0:d} records whose passages or snippets contain answers split to {1:d} records in {2:.2f} second(s) '
          '(all sentence combinations generated)\n'.format(
           (rank_train[label_col] == 1).sum(), answer_train.shape[0], time.time() - tic))

    # Add answer extraction labels: label = 1 if the sentence combination is the shortest among all the sentence
    # combinations that contain answer as a substring; label = 0 otherwise
    answer_train = answer_train.groupby(query_col).apply(lambda x:
        get_answer_label(x, answer_col, sent_comb_col, label_col))

    answer_label_distr = answer_train.groupby(query_col).apply(lambda x: x[label_col].sum())
    plot_distr(answer_label_distr, os.path.join(args.root_path, args.answer_label_distr_plot_name + '-' + suffix + '.pdf'),
               'Distribution of number of label 1 per query in answer extraction training data (' + suffix + ')')

    answer_train.reindex(columns=[qid_col, query_col, answer_col, sent_comb_col, label_col]).to_json(
        os.path.join(args.root_path, args.answer_train_name + '-' + suffix + '.json'), orient='columns')
    print('Data for training answer extraction model have {0:d} negative records, {1:d} positive records and in total'
          ' {2:d} records\n'.format(
           (answer_train[label_col] == 0).sum(), (answer_train[label_col] == 1).sum(), answer_train.shape[0]))


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    data_preparation(arg_parser)
