# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
import operator
import flask

import time
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from string import punctuation
import re
import argparse



class ScoringService(object):
    embed = None                # Where we keep the model when it's loaded
    qa_pairs = None
    questions = None
    qEmbeddings = None
    prefix = '/home/yqinamz/model_in_prod/'
    model_path = os.path.join(prefix, 'model')
    
    EMBED_SIZE=768
    THRESHOLD=0.94
    blist = ['a credit', 'a discount', 'a new order', 'a phone call i received', 'a recent order', 'about my order', 'account question', 'account specialist', 'add to my order', 'amazon', 'amazon email', 'amazon order', 'amazon pay', 'an email i received', 'an email i recieved', 'an item i canceled', 'an item price', 'an order', 'another account', 'assistance', 'billing issue', 'call instead', 'call me', 'call me instead', 'call me please', 'can i talk to an amazon assitant', 'can you call me', 'can you help me', 'can you please call me now', 'claim status', 'complaint', 'costumer service', 'credit', 'customer', 'delivery date', 'did not order', 'discounts', 'echo loop', 'email', 'email from amazon', 'email i go', 'email i got', 'email i received', 'email i recieved', 'email offer', 'emails', 'fraudulent charges', 'good morning', 'hello', 'help me', 'help with a claim', 'help with an order', 'help with my order', 'help with order', 'hi there', 'how do i email the seller?', 'i accidentally canceled an order', 'i accidentally cancelled an order', 'i got an email', 'i have an issue', 'i need a phone call', 'i need help', 'i need help from amazon 2.0', 'i need more help', 'i need to call amazon', 'i placed an order', 'i want a phone call', 'information', 'issue with an order', 'issue with order', 'it is the end of the day', 'item information', 'item price', 'last order', 'lists', 'live help', 'memberships', 'missing', 'mistake', 'more help', 'movie', 'multiple accounts', 'my claim', 'my credit', 'my last order', 'my lists', 'my most recent order', 'my order', 'my recent order', 'need a phone call', 'need help', 'never mind', 'nevermind', 'new order', 'none of the above', 'none of these', 'order', 'order issues', 'order item', 'order number', 'order problem', 'order question', 'ordering', 'other', 'phone call', 'phone call from amazon', 'phone call i received', 'phone calls', 'phone number', 'place an order', 'placing order', 'please call me', 'please call me about an order', 'pre order', 'previous order', 'price', 'price issue', 'price question', 'problem with order', 'question', 'question about an email', 'question about an order', 'question about order', 'shipping', 'shipping times', 'something else', 'status', 'supervisor', 'support', 'tablet', "that didn't work", 'what is your phone number', 'when will', 'when will i get it', 'when will it ship?', 'why is it taking so long?', 'why so long', 'why so long?', 'why was my order canceled', 'wrongfully charged', 'yes that order','got disconnected', 'accidentally cancelled order', 'disconnected', 'error', 'mistake', 'return', 'email']
    pattern = '((associate|assistant|agent|representative|operator|artificial|chat|rep|human)s?|(need |chat |ask |speak |talk |call |help |contact )(to |with )?(a |an |from )?(live |real |customer service |physical )?(person|someone|customer (service|support)))( please)?$'
     
    @classmethod
    def clean_text(cls, text):
        text = re.sub('^"', "", text)
        text = re.sub('"$', "", text)
        text = re.sub('e-mail', "email", text)
        text = re.sub('\$', " ", text)
        text = re.sub('\?', " ", text)
        text = re.sub(r"\(", " ", text)
        text = re.sub(r"\)", " ", text)
        text = re.sub(r"\<.*\>", " ", text)
        text = re.sub(r"\[.*\]", " ", text)
        text = re.sub(r"\(.*\)", " ", text)
        text = re.sub(r"\{.*\}", " ", text)
        text = re.sub(r"^ ", " ", text)
        text = re.sub(r":", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"_", " ", text)
        text = re.sub(r"\(", " ", text)
        text = re.sub(r"\)", " ", text)
        text = re.sub(r";", " ", text)
        text = re.sub(r"@", " ", text)
        text = re.sub(r"/", " ", text)
        text = re.sub(r"\\", " ", text)
        text = re.sub(r"~", " ", text)
        text = re.sub(r"\|", " ", text)
        text = re.sub(r"\{", " ", text)
        text = re.sub(r"\}", " ", text)
        text = re.sub(r"#", " ", text)
        text = re.sub(r"\. ", "", text)
        text = re.sub(r"\!", " ", text)
        text = re.sub(r"\+", " ", text)
        text = re.sub(r"\n", " ", text)

        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        email_regex = '(^[a-z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
        happy_emoticon_regex = ': *\)'
        sad_emoticon_regex = ': *\('
        amazon_regex = '.*amazon.*'
        number_regex = 'num'
        white_space_regex = '\s+'
        extra_white_space_regex = '^\s+|\s+$'

        words = re.split('\s+', text)

        for w in range(len(words)):
            if re.match(email_regex, words[w]): 
                words[w] = ' email '
            elif re.match(url_regex, words[w]):
                words[w] = ' url '
            elif re.match(amazon_regex, words[w], re.IGNORECASE):
                words[w] = ' amazon '

        words[w] = re.sub(number_regex, 'num', words[w])

        text = ' '.join(words)
        text = re.sub(happy_emoticon_regex, ' happyemot ', text)
        text = re.sub(sad_emoticon_regex, ' sademot ', text)

        text = re.sub(white_space_regex, ' ', text)
        text = re.sub(extra_white_space_regex, '', text)

        text = re.sub(r"\s{1,}", ' ', text)

        return text.lower().strip()  
    

    @classmethod
    def blacklist(cls, text):            
        if text in ['hbo','tax','tip','app','tv']:
            return False
        elif len(text)>180 or len(text)<4:
            return True
        elif re.search('\d{3}-\d{7}-\d{7}',text) != None:
            return True
        elif text in cls.blist:
            return True
        elif re.search(cls.pattern,text) != None:
            return True
        else:
            return False

    
    @classmethod
    def get_model(cls):
        if cls.embed == None:
            """Get the model object for this instance, loading it if it's not already loaded."""
            full_model_path = os.path.join(cls.model_path, 'Sbert_model')
            pickle_file_path = os.path.join(cls.model_path, 'question-answer-pairs.pkl')
            cls.embed = SentenceTransformer(full_model_path)
            cls.qa_pairs = pickle.load(open(pickle_file_path,'rb'))
            cls.questions = list(cls.qa_pairs.keys())
            cls.qEmbeddings= cls.generateQuestionEmbed()
        return cls.embed


    @classmethod
    def generateQuestionEmbed(cls):
        qa_bank = [cls.clean_text(q) for q in cls.questions]
        qEmbed = cls.embed.encode(qa_bank)
        return qEmbed

 
    @classmethod
    def getQueryEmbed(cls,query):
        query=cls.clean_text(query)
        if cls.blacklist(query):
            return []
        else:
            return cls.embed.encode([query])
    
    
    @classmethod
    def predict(cls, data):
        """For the input text utterance, do the prediction and return it.
    
        Args:
            data: a json object containing the non-stemmed and stemmed utterance"""
        embed = cls.get_model()
        input_json = json.loads(data)
        query = input_json['query']
        queryEmbed = cls.getQueryEmbed(query)
        if len(queryEmbed) == 0:
            return [None,[]]
        else:
            scores = np.squeeze(1 - cdist(cls.qEmbeddings, queryEmbed, 'cosine'), axis=1)
            if (np.max(scores) > cls.THRESHOLD):
                return cls.qa_pairs[cls.questions[np.argmax(scores)]]
            else:
                return [None,[]]
            
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module is used to train Question Similarity model')

    parser.add_argument('--model_name', type=str,
                        help='model name')
    parser.add_argument('--qa_bank_json', type=str,
                        help='path of json qa_bank')
    # parser.add_argument('--gpu', type=str, default='0', 
    #                     help='gpus to run model on')
    parser.add_argument('--test_data_path', type=str,
                    help='test data path')


    args = parser.parse_args()


    def get_repo(qa_repo_path, model):
        f_in = open(qa_repo_path, "r")
        data = json.load(f_in)
        qa_repo = []
        qa_id = []

        for i in range(len(data['qa_bank'])):
            qa_repo += data['qa_bank'][i]['questions']
            qa_id += [data['qa_bank'][i]['id']] * len(data['qa_bank'][i]['questions'])

        qa_embed = model.encode(qa_repo)
        return qa_embed, qa_repo, qa_id
     
     
    def GetSimilarQ(query, model, question_embed, uttr_list, id_list, topn=1):
        query_embed = model.encode([query])
        cos_dist = np.squeeze(1 - cdist(question_embed, query_embed, 'cosine'), axis=1)
        idx_list = cos_dist.argsort()[-topn:][::-1]
        return uttr_list[idx_list[0]], cos_dist[idx_list[0]], id_list[idx_list[0]]


    def replay(input_file, model, qa_embed, qa_repo, qa_id):
        with open(input_file,'r',errors='ignore') as f_in:
            out_list = []
            n_sample = 0 
            for idx, ln in enumerate(f_in):
                if idx !=0:
                    n_sample += 1
                    query, true_matched_q_idx = ln.split('\t')
                    true_matched_q_idx = true_matched_q_idx.replace('\n','')
                    if ScoringService.blacklist(query):
                        matched_q = ''
                        score = 0
                        matched_q_idx = None
                    else:
                        matched_q, score, matched_q_idx = GetSimilarQ(query, model, qa_embed, qa_repo, qa_id)
                    correct_match = 1 if true_matched_q_idx == matched_q_idx else 0
                    out_list.append([query, true_matched_q_idx, matched_q, score, matched_q_idx, correct_match])
        df=pd.DataFrame(out_list)
        df.columns=['query','true_matched_q_idx','matched_q','score','matched_q_idx','correct_match']
        return df, n_sample
     

    model = SentenceTransformer(args.model_name)
    qa_embed, qa_repo, qa_id = get_repo(args.qa_bank_json, model)

    start = time.time()
    df, n_sample = replay(args.test_data_path, model, qa_embed, qa_repo, qa_id)
    end = time.time()
    print('{0:d} question pairs were scored in {1:.2f} seconds (in average {2:.2f} ms per query)'.format(n_sample, end - start, 1000 * (end - start) / n_sample))

    # Save the information retrieval result for future use
    test_infer_res_path = os.path.join(args.model_name, 'test_infer_res.tsv')
    df.to_csv(test_infer_res_path, sep='\t')

    # Calculate the information retrieval metrics
    print('{} question pairs were scored by {}: \n'
        '    Number of correct match: {}\n'
        '    Percentage: {:.4f}'.format(n_sample, args.model_name, df.correct_match.sum(), df.correct_match.sum()/len(df)))
