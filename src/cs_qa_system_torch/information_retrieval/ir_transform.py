import numpy as np

class RankingTransform(object):
  def __init__(self, tokenizer, max_len=512, bool_np_array = False):
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.bool_np_array = bool_np_array

    self.cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    self.sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    self.pad_id = 0

  def __call__(self, text):
    if isinstance(text, list):
        # tokenized_dict = self.tokenizer.batch_encode_plus(text,
        #                                         add_special_tokens=True,
        #                                         max_length=self.max_len,
        #                                         pad_to_max_length=True,
        #                                         truncation=True,
        #                                         return_attention_mask=True,
        #                                         return_token_type_ids=True)
        tokenized_dict = self.tokenizer.batch_encode_plus(text,
                                                max_length=self.max_len,
                                                padding=True,
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=True,
                                                return_tensors='pt')
    else:
        # tokenized_dict = self.tokenizer.encode_plus(text,
        #                                         text_pair=None,
        #                                         add_special_tokens=True,
        #                                         max_length=self.max_len,
        #                                         pad_to_max_length=True,
        #                                         truncation=True,
        #                                         return_attention_mask=True,
        #                                         return_token_type_ids=True)
        tokenized_dict = self.tokenizer.encode_plus(text,
                                                    text_pair=None,
                                                    max_length=self.max_len,
                                                    padding=True,
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    return_token_type_ids=True,
                                                    return_tensors='pt')
    input_ids, attention_mask, segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']

    if self.bool_np_array:
        return np.array(input_ids), np.array(attention_mask), np.array(segment_ids)
    else:
        return input_ids, attention_mask, segment_ids

  def __str__(self) -> str:
    return ''

class CombinedRankingTransform(object):
  def __init__(self, tokenizer, max_len=512, bool_np_array = False):
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.bool_np_array = bool_np_array

    self.cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    self.sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    self.pad_id = 0

  def __call__(self, text1, text2):
    tokenized_dict = self.tokenizer.encode_plus(text1,
                                                text_pair=text2,
                                                add_special_tokens=True,
                                                max_length=self.max_len,
                                                pad_to_max_length=True,
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=True)
    input_ids, attention_mask, segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']

    assert len(input_ids) == self.max_len
    assert len(segment_ids) == self.max_len
    assert len(attention_mask) == self.max_len

    if self.bool_np_array:
        return np.array(input_ids), np.array(attention_mask), np.array(segment_ids)
    else:
        return input_ids, attention_mask, segment_ids

  def __str__(self) -> str:
    return ''
