import os
import re
import json
import pandas as pd

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult, squad_convert_example_to_features, \
    squad_convert_example_to_features_init, SquadExample

from ir_and_ae.answer_formatter import reformat_answer
from information_retrieval.ir_utils import predict_ranking_model, load_ranking_model, load_document_collection


class ScorePrediction:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## Collection path
    ROOT_QASYSTEM_ARTIFACTS = '/data/QAArtifacts/model'
    passage_collection_path = os.path.join(ROOT_QASYSTEM_ARTIFACTS, 'production_collection.json')
    passage_url_path = os.path.join(ROOT_QASYSTEM_ARTIFACTS, 'production_collection_url.json')

    ## ranking parameters
    rank_model_name_or_path = 'BM25Okapi'
    rank_top_n = 10
    rank_inference_batch_size = 512
    rank_model_score_threshold = 0.0

    ## reranking parameters
    do_rerank = True
    rerank_model_name_or_path = os.path.join(ROOT_QASYSTEM_ARTIFACTS, 'ir_artifacts/rerank_crossencoder_bert/')
    rerank_top_n = 1
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

    passage_url = None
    collection_ae = None

    answer_extraction_model_name_or_path = '/data/qyouran/QABot/output_torch_train/ae_finetuned_large'
    answer_extraction_model = None
    answer_extraction_tokenizer = None

    do_lower_case = True
    max_seq_length = 384
    max_query_length = 64
    doc_stride = 128
    model_type_ae = 'bert'
    n_best_size = 10
    max_answer_length = 150
    min_answer_length = 5

    ################ blacklist code ########################################################################
    blist = ['a credit', 'a discount', 'a new order', 'a phone call i received', 'a recent order', 'about my order',
             'account question', 'account specialist', 'add to my order', 'amazon', 'amazon email', 'amazon order',
             'amazon pay', 'an email i received', 'an email i recieved', 'an item i canceled', 'an item price',
             'an order', 'another account', 'assistance', 'billing issue', 'call instead', 'call me', 'call me instead',
             'call me please', 'can i talk to an amazon assitant', 'can you call me', 'can you help me',
             'can you please call me now', 'claim status', 'complaint', 'costumer service', 'credit', 'customer',
             'delivery date', 'did not order', 'discounts', 'echo loop', 'email', 'email from amazon', 'email i go',
             'email i got', 'email i received', 'email i recieved', 'email offer', 'emails', 'fraudulent charges',
             'good morning', 'hello', 'help me', 'help with a claim', 'help with an order', 'help with my order',
             'help with order', 'hi there', 'how do i email the seller?', 'i accidentally canceled an order',
             'i accidentally cancelled an order', 'i got an email', 'i have an issue', 'i need a phone call',
             'i need help', 'i need help from amazon 2.0', 'i need more help', 'i need to call amazon',
             'i placed an order', 'i want a phone call', 'information', 'issue with an order', 'issue with order',
             'it is the end of the day', 'item information', 'item price', 'last order', 'lists', 'live help',
             'memberships', 'missing', 'mistake', 'more help', 'movie', 'multiple accounts', 'my claim', 'my credit',
             'my last order', 'my lists', 'my most recent order', 'my order', 'my recent order', 'need a phone call',
             'need help', 'never mind', 'nevermind', 'new order', 'none of the above', 'none of these', 'order',
             'order issues', 'order item', 'order number', 'order problem', 'order question', 'ordering', 'other',
             'phone call', 'phone call from amazon', 'phone call i received', 'phone calls', 'phone number',
             'place an order', 'placing order', 'please call me', 'please call me about an order', 'pre order',
             'previous order', 'price', 'price issue', 'price question', 'problem with order', 'question',
             'question about an email', 'question about an order', 'question about order', 'shipping', 'shipping times',
             'something else', 'status', 'supervisor', 'support', 'tablet', "that didn't work",
             'what is your phone number', 'when will', 'when will i get it', 'when will it ship?',
             'why is it taking so long?', 'why so long', 'why so long?', 'why was my order canceled',
             'wrongfully charged', 'yes that order', 'got disconnected', 'accidentally cancelled order', 'disconnected',
             'error', 'mistake', 'return', 'email', 'a return', 'returns', 'help', 'please help', 'a question',
             'questions',
             'i have a question', 'i have questions', 'gift', 'a gift', 'gifts', 'gift received', 'a gift received',
             'gifts received',
             'gift i received', 'a gift i received', 'gifts i received', 'gift refund', 'gift refunds',
             'customer service',
             'placing an order', 'placing orders', 'place order', 'place orders', 'orders', 'my orders']
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
        if text in ['hbo', 'tax', 'tip', 'app', 'tv']:
            return False
        elif len(text) > 180 or len(text) < 4:
            return True
        elif re.search('\d{3}-\d{7}-\d{7}', text) != None:
            return True
        elif text in cls.blist:
            return True
        elif re.search(cls.pattern, text) != None:
            return True
        else:
            return False

    ############## end black list code ######################################################################

    @classmethod
    def load_data(cls):
        cls.collection_ir, cls.idx_to_doc, cls.doc_to_idx, cls.collection_ae = load_document_collection(cls.passage_collection_path)
        passage_data = json.load(open(cls.passage_url_path, 'r'))
        cls.passage_url = {k: (v1, v2) for k, [v1, v2] in passage_data.items()}

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
            cls.load_ae_models()
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


    @classmethod
    def get_original_passage(cls, x, passage_collection_ae, passage_url):
        output = ''
        sentinel = ''
        output_pid = ''
        output_url = ''
        output_title = ''
        sentinel_url = ''
        for i in x.values:
            if i in passage_collection_ae:
                output = output + sentinel + passage_collection_ae[i]
                sentinel = '.'
                output_pid = output_pid + sentinel_url + str(i)
                output_url = output_url + sentinel_url + passage_url[i][0]
                output_title = output_title + sentinel_url + passage_url[i][1]
                sentinel_url = '::'
        return pd.Series({'pid': output_pid, 'passage': output, 'url': output_url, 'title': output_title})

    @classmethod
    def to_list(cls, tensor):
        return tensor.detach().cpu().tolist()

    @classmethod
    def load_ae_models(cls):
        cls.answer_extraction_model = AutoModelForQuestionAnswering.from_pretrained(cls.answer_extraction_model_name_or_path)  # , force_download=True)
        cls.answer_extraction_tokenizer = AutoTokenizer.from_pretrained(cls.answer_extraction_model_name_or_path, do_lower_case=cls.do_lower_case)
        cls.answer_extraction_model.to(cls.device)
        return True

    @classmethod
    def get_answer(cls, query):
        df = cls.get_documents(query)
        prediction_df = df.groupby(['qid', 'query']).apply(
            lambda x: cls.get_original_passage(x['pid'], cls.collection_ae, cls.passage_url)).reset_index()
        prediction_df.columns = ['qid', 'query', 'pid', 'passage', 'url', 'title']

        example = SquadExample(
            qas_id=0,
            question_text=query,
            context_text=prediction_df.iloc[0]['passage'],
            answer_text=None,
            start_position_character=None,
            title=None,
            answers=None,
        )
        squad_convert_example_to_features_init(cls.answer_extraction_tokenizer)
        features = squad_convert_example_to_features(example, cls.max_seq_length, cls.doc_stride, cls.max_query_length,
                                                     False)

        new_features = []
        unique_id = 1000000000
        for feature in features:
            feature.example_index = 0
            feature.unique_id = unique_id
            new_features.append(feature)
            unique_id += 1
        features = new_features
        del new_features

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index,
                                all_cls_index, all_p_mask)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=len(dataset))

        all_results = []
        for batch in dataloader:
            cls.answer_extraction_model.eval()
            batch = tuple(t.to(cls.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if cls.model_type_ae in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                    del inputs["token_type_ids"]

                feature_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if cls.model_type_ae in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(cls.answer_extraction_model, "config") and hasattr(cls.answer_extraction_model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * 1).to(cls.device)}
                        )
                outputs = cls.answer_extraction_model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [cls.to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

        predictions_realtime = compute_predictions_logits(
            [example],
            features,
            all_results,
            cls.n_best_size,
            cls.max_answer_length,
            cls.do_lower_case,
            None,
            None,
            None,
            True,
            True,
            0.0,
            cls.answer_extraction_tokenizer,
        )
        # return predictions_realtime[0]
        if len(predictions_realtime[0].split()) > cls.min_answer_length:
            return [prediction_df.iloc[0]['pid'], reformat_answer(predictions_realtime[0], prediction_df.iloc[0]['url'],
                                                                  prediction_df.iloc[0]['title'])]
        else:
            return [None, reformat_answer('', None, None)]

