import utils
import constants
import math
import time
import psutil
import pprint
import bottleneck
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, Process
from collections import Counter
from scipy.sparse import csr_matrix
pprint.sorted = lambda x, key=None: x

class IRModelFactory:
    """
    This factory class returns information retrieval (IR) model object depending on the input argument
    """
    @staticmethod
    def create_ir_model(ir_model_name: str, ir_model_path: str, corpus: list, tokenizer: callable, **kwargs):
        if ir_model_name.startswith(constants.IR_BM25OKAPI):
            ir_model_param = {k: v if ((k not in kwargs) or (kwargs[k] is None)) else kwargs[k] for k, v in constants.IR_MODELS[constants.IR_BM25OKAPI].items()}
            return BM25Okapi(name=ir_model_name, root_path=ir_model_path, corpus=corpus, tokenizer=tokenizer, **ir_model_param)
        else:
            raise NotImplementedError()


class BM25(object):

    def __init__(self, name, root_path, corpus=None, tokenizer=None, n_cpu=psutil.cpu_count(logical=True)):
        """
        Initialize the BM25 instance by indexing all the passages in the corpus
        :param name: name of the BM25 instance
        :param root_path: path of the BM25 instance (it will be used when we save this instance)
        :param corpus: corpus with the format [(pid_1, text_1), (pid_2, text_2), ...]
        :param tokenizer: function to tokenize the text
        :param n_cpu: number of cpus we want to use to preprocess all the passages in the corpus
        """
        try:
            self.load(name, root_path)
        except FileNotFoundError:
            self._init_process(name, root_path, corpus, tokenizer, n_cpu)

    def _init_process(self, name, root_path, corpus, tokenizer, n_cpu):
        self.name = name
        self.root_path = root_path
        self.passage_count = len(corpus)
        self.tokenizer = tokenizer

        # Get some information from each passage using the function self._get_passage_info with multiple cpus
        tic = time.time()
        with Pool(n_cpu) as pool:
            passage_info = pool.map(self._get_passage_info, corpus)
        toc = time.time()
        # print('Information of {0:d} passages obtained using {1:d} processes in {2:.2f} second(s)'.format(
        #     self.passage_count, n_cpu, toc - tic))

        # Get the index to pid mapping (self.idx2pid), the word count for each passage (self.word_count_array),
        # the number of passages containing each word (self.num_passage_containing_word) and
        # the total word count (self.total_word_count)
        tic = time.time()
        self.idx2pid = dict()
        self.word_count_array = np.zeros((self.passage_count, ), dtype='int32')
        self.num_passage_containing_word = Counter()
        for i, tuple_info in enumerate(passage_info):
            pid, word_count, _, word_indicator = tuple_info
            self.idx2pid[i] = pid
            self.word_count_array[i] = word_count
            self.num_passage_containing_word.update(word_indicator)  # "update" is much faster than "+="
        self.total_word_count = self.word_count_array.sum()
        toc = time.time()
        # print('idx2pid, word_count_array, num_passage_containing_word and total_word_count obtained '
        #       'in {:.2f} second(s)'.format(toc - tic))

        # Get the word to index mapping (self.word2idx), the idf for each word (self.idf_array)
        tic = time.time()
        self.word2idx = {word: (i + 1) for i, word in enumerate(self.num_passage_containing_word)}
        self.idf_array = self._get_idf_array()
        toc = time.time()
        # print('word2idx and idf_array obtained in {:.2f} second(s)'.format(toc - tic))

        # Get the word-passage frequency array (self.freq_array). It is an n * m array, where n is the total word
        # count, m is the total passage count, the (i, j)th entry is the number of times word i occurs in passage j
        # (i and j are the indices in self.word2idx and self.idx2pid)
        # This array is sparse and consists of small integers, so here we use a sparse matrix with dtype "uint32" to
        # reduce the memory
        tic = time.time()
        col = list()
        row = list()
        data = list()
        for i in range(self.passage_count):
            for word, freq in passage_info[i][2].items():
                col.append(i)
                row.append(self.word2idx[word])
                if freq >= 4294967296:
                    # print('The word "{0:}" occurs {1:d} times in passage with pid {2:d}, '
                    #       'and we truncate its occurrence frequency to 4294967295'.format(word, freq, self.idx2pid[i]))
                    freq = 4294967295
                data.append(freq)
        self.freq_array = csr_matrix((data, (row, col)), dtype='uint32')
        toc = time.time()
        # print('freq_array obtained in {:.2f} second(s)'.format(toc - tic))

        assert (self.freq_array.toarray().sum(axis=0) == self.word_count_array).all()
        assert (self.freq_array.toarray()[0] == 0).all()

    def _get_passage_info(self, passage_tuple):
        """
        Get some information from a passage
        :param passage_tuple: (pid, text)
        :return: pid,
                 number of words in the tokenized text,
                 word frequency in the tokenized text,
                 word indicator to mark each word occuring in the text, e.g., it is {'how': 1, 'are': 1, 'you': 1} for
                 "how are you" (the word indicator is only used to calculate self.num_passage_containing_word)
        """
        text_tokenized = self.tokenizer(passage_tuple[1])
        word_count = len(text_tokenized)
        word_freq = Counter(text_tokenized)
        word_indicator = Counter(dict.fromkeys(word_freq, 1))
        return passage_tuple[0], word_count, word_freq, word_indicator

    def save(self, removed=None):
        """
        Save the BM25 instance to hard disk for future use
        :param removed: the private variables we don't want to save
        """
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path, exist_ok=False)
        if removed is None:
            utils.save_list_dict(self.__dict__, '{:}_saved'.format(self.name), self.root_path, protocol=4)
            # print('{0:} saved to {1:}'.format(self.name, self.root_path))
        else:
            assert isinstance(removed, list) and set(removed).issubset(set(self.__dict__.keys()))
            dict_to_save = {k: v for k, v in self.__dict__.items() if k not in removed}
            utils.save_list_dict(dict_to_save, '{:}_saved'.format(self.name), self.root_path, protocol=4)
            # print('{0:} saved to {1:} with {2:} removed'.format(self.name, self.root_path, ', '.join(removed)))

    def load(self, name, root_path):
        """
        Load the BM25 instance previously saved in the hard disk
        :param name: name of the BM25 instance
        :param root_path: path of the BM25 instance
        """
        self.__dict__ = utils.load_list_dict('{:}_saved'.format(name), root_path)
        old_root_path = self.root_path.rstrip('/')
        new_root_path = root_path.rstrip('/')
        if old_root_path != new_root_path:
            self.change_root_path(new_root_path)
        # print('{0:} loaded from {1:}'.format(name, root_path))

    def change_name(self, new_name):
        """
        Change the name of the BM25 instance
        :param new_name: new name of the BM25 instance
        """
        old_name = self.name
        self.name = new_name
        # print('name changed from {0:} to {1:}'.format(old_name, new_name))

    def change_root_path(self, new_root_path):
        """
        Change the root path of the BM25 instance
        :param new_root_path: new root path of the BM25 instance
        """
        new_root_path = new_root_path.rstrip('/')
        old_root_path = self.root_path.rstrip('/')
        self.root_path = new_root_path
        # print('root_path changed from {0:} to {1:}'.format(old_root_path, new_root_path))

    def _get_idf_array(self):
        raise NotImplementedError()

    def get_score_single(self, query):
        raise NotImplementedError()

    def get_score_single_for_passage(self, query, pid):
        """
        Get the score for a given query and a given passage pid
        This function could be slow and should be used for development only
        :param query: query
        :param pid: pid
        :return: score between the query and the passage
        """
        score = self.get_score_single(query)
        idx = utils.get_key_with_value(self.idx2pid, pid)
        assert idx is not None
        return score[idx]

    def get_top_n_single(self, query, n=1):
        """
        For a given query, get pids of the sorted top n relevant passages
        :param query: query
        :param n: number of relevant passages
        :return: a list of pids of the sorted top n relevant passages and their scores
        """
        score = self.get_score_single(query)
        if n == self.passage_count:
            top_n_idx_sorted = np.argsort(score)[::-1][:n]
        else:
            # This step takes a cost of O(self.passage_count), but top_n_idx returned by bottleneck is unsorted
            top_n_idx = bottleneck.argpartition(score, self.passage_count - n)[-n:]
            # This step takes a cost of O(nlogn)
            top_n_idx_sorted = top_n_idx[np.argsort(-score[top_n_idx])]
        return [self.idx2pid[i] for i in top_n_idx_sorted], score[top_n_idx_sorted]

    def _get_top_n_range(self, start, end, df, query_col, n):
        """
        For each query within a specific range of the column of a pandas dataframe, get pids of the sorted top n
        relevant passages and save the new dataframe to hard disk
        It is used as a helper function for parallel computing in the "get_top_n" function
        We must save the dataframe in a dictionary because some of its entries are Python lists
        (if we save it as a tsv file, these lists will be saved as strings)
        :param start: starting index
        :param end: ending index
        :param df: pandas dataframe
        :param query_col: name of the column containing queries
        :param n: number of relevant passages
        """
        result = df[query_col].iloc[start:end].map(lambda x: self.get_top_n_single(x, n=n)[0])
        result.name = 'relevant_pid_pred'
        df_concat = pd.concat((df.iloc[start:end, :], result), axis=1)
        df_concat_name = '{0:d}-{1:d}'.format(start, end)
        utils.save_df_as_dict(df_concat, df_concat_name, self.root_path)

    def get_top_n(self, df, query_col, n=1, n_cpu=psutil.cpu_count(logical=True)):
        """
        For each query in the column of a pandas dataframe, get pids of the sorted top n relevant passages
        When we have a large number of queries, it takes a lot of time to do this, so here we use parallel computing
        :param df: pandas dataframe
        :param query_col: name of the column containing queries
        :param n: number of relevant passages
        :param n_cpu: number of cpus we want to use to retrieve information for a large number of queries
        :return: new pandas dataframe with the column "relevant_pid_pred" inserted
        """
        tic = time.time()
        processes = []
        start_end_index = utils.get_start_end_index(df.shape[0], n_cpu)
        for i, start_end in enumerate(start_end_index):
            start, end = start_end
            p = Process(target=self._get_top_n_range, name='{0:d}-{1:d}'.format(start, end),
                        args=(start, end, df, query_col, n), daemon=True)
            # os.system('taskset -p -c {0:d} {1:d}'.format(i % n_cpu, os.getpid()))
            processes.append(p)
            p.start()
            # print('Process {:} is running...'.format(p.name))
        for p in processes:
            p.join()

        df_result = pd.DataFrame()
        for start, end in start_end_index:
            df_concat_name = '{0:d}-{1:d}'.format(start, end)
            df_concat = utils.load_df_from_dict(df_concat_name, self.root_path)
            df_result = df_result.append(df_concat, ignore_index=False)
            os.remove(os.path.join(self.root_path, df_concat_name))
        toc = time.time()
        # print('Top-{0:d} relevant passages for {1:d} queries found using {2:d} processes in {3:.2f} second(s)'.format(
        #     n, df_result.shape[0], n_cpu, toc - tic))

        return df_result


class BM25Okapi(BM25):
    def __init__(self, name, root_path, corpus=None, tokenizer=None, n_cpu=psutil.cpu_count(logical=True),
                 k1=None, b=None, delta=None):
        """
        Initialize the BM25Okapi instance by indexing all the passages in the corpus (BM25Okapi is a subclass of BM25)
        :param name: name of the BM25 instance
        :param root_path: path of the BM25 instance (it will be used when we save this instance)
        :param corpus: corpus with the format [(pid_1, text_1), (pid_2, text_2), ...]
        :param tokenizer: function to tokenize the text
        :param n_cpu: number of cpus we want to use to preprocess all the passages in the corpus
        :param k1: hyperparameter for BM25Okapi
        :param b: hyperparameter for BM25Okapi
        :param delta: hyperparameter for BM25Okapi
        """
        try:
            self.load(name, root_path)
        except FileNotFoundError:
            self._init_process(name, root_path, corpus, tokenizer, n_cpu)

            self.k1 = k1
            self.b = b
            self.delta = delta
            # This value doesn't rely on the query, so we calculate it in advance to speed up the "get_score_single" function
            self.part_of_denominator = \
                self.k1 * (1 - self.b + self.b * self.word_count_array / (self.total_word_count / self.passage_count))

    def _get_idf_array(self, eps=0.25):
        """
        Get the idf for each word in the corpus
        The idf can be negative if the word is contained in more than half of the passages,
        so we collect words with negative idf and set them as "eps * (idf_sum / len(idf_array))"
        :return: idf array for each word in the corpus
        """
        idf_array = np.zeros((len(self.word2idx) + 1, ), dtype='float32')
        idf_negative = []
        idf_sum = 0
        for word, freq in self.num_passage_containing_word.items():
            idf = math.log(self.passage_count - freq + 0.5) - math.log(freq + 0.5)
            idf_array[self.word2idx[word]] = idf
            if idf < 0:
                idf_negative.append(word)
            idf_sum += idf
        for word in idf_negative:
            idf_array[self.word2idx[word]] = eps * (idf_sum / len(idf_array))
        return idf_array

    def get_score_single(self, query):
        """
        For a given query, get scores of all the passages in the corpus
        :param query: query
        :return: scores of all the passages in the corpus
        Comments: (1) freq_array_query * (self.k1 + 1) / (freq_array_query + self.part_of_denominator) + self.delta is
                      an np.matrix of shape (n, m), and idf_array_query is an np.array of shape (n, ), where n is the
                      word count in the query, and m is the total passage count.
                      np.matmul will make idf_array_query have shape (1, n) and perform the matrix multiplication to
                      get an np.matrix of shape (1, m)
                  (2) If A is an np.array of shape (n, ) and B is an np.array (n, m) where n != m, then the element-wise
                      multiplication A * B is not allowed.
                      However, if A is an np.array of shape (m, ) and B is an np.array (n, m) where n != m, then the
                      element-wise multiplication A * B is allowed and A will be broadcasted
        """
        query_index = [self.word2idx.get(word, 0) for word in self.tokenizer(query)]
        freq_array_query = self.freq_array[query_index]
        idf_array_query = self.idf_array[query_index]
        return np.array(np.matmul(idf_array_query,
            freq_array_query * (self.k1 + 1) / (freq_array_query + self.part_of_denominator) + self.delta))[0]