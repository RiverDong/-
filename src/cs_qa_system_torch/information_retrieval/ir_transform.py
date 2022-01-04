import numpy as np

class RankingTransform(object):
  def __init__(self, tokenizer, max_len=512):
    self.tokenizer = tokenizer
    self.max_len = max_len

    self.cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    self.sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    self.pad_id = 0

  def __call__(self, text):
    if isinstance(text, list):
        tokenized_dict = self.tokenizer.batch_encode_plus(text,
                                                max_length=self.max_len,
                                                padding=True,
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=True,
                                                return_tensors='pt')
    else:
        tokenized_dict = self.tokenizer.encode_plus(text,
                                                    text_pair=None,
                                                    max_length=self.max_len,
                                                    padding=True,
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    return_token_type_ids=True,
                                                    return_tensors='pt')
    input_ids, attention_mask, segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']

    return input_ids, attention_mask, segment_ids

  def __str__(self) -> str:
    return ''

class CombinedRankingTransform(object):
  def __init__(self, tokenizer, max_len=512):
    self.tokenizer = tokenizer
    self.max_len = max_len

    self.cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    self.sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    self.pad_id = 0

  def __call__(self, text1, text2=None):
    if isinstance(text1, list):
        tokenized_dict = self.tokenizer.batch_encode_plus(text1,
                                                max_length=self.max_len,
                                                padding=True,
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=True,
                                                return_tensors='pt')
    else:
        tokenized_dict = self.tokenizer.encode_plus(text1,
                                                    text_pair=text2,
                                                    max_length=self.max_len,
                                                    padding=True,
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    return_token_type_ids=True,
                                                    return_tensors='pt')
    input_ids, attention_mask, segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']

    return input_ids, attention_mask, segment_ids

  def __str__(self) -> str:
    return ''
