import utils
import string
import re
import copy
import time

def all_x_in_y(x, y):
    # check whether all characters in x are in y, where x and y are two strings
    assert isinstance(x, str) and isinstance(y, str)
    return (len(x) > 0) and set(list(x)).issubset(set(list(y)))


def is_nonempty_str(x):
    return isinstance(x, str) and (re.fullmatch(r'(?ias)\s*', x) is None)


class EnglishPreprocessing(object):
    def __init__(self, name, root_path,
                 do_noise=None,
                 do_scrub_token=None,
                 do_acronym=None,
                 do_number=None,
                 do_apostr_contract=None,
                 do_apostr_possess=None,
                 do_hyphen=None,
                 do_punc=None,
                 do_lower=None,
                 do_final_replacement=None,
                 do_other_special_char=None,
                 do_spell_correct=None,
                 do_remove_word=None,
                 do_stem_or_lemmatize=None,
                 verbose=None,
                 noise_marker=None,
                 noise_regex=None,
                 scrub_token_marker=None,
                 scrub_token_collapse=None,
                 scrub_token_regex=None,
                 acronym_marker=None,
                 acronym_regex=None,
                 number_marker=None,
                 number_regex=None,
                 apostr_contract_collapse=None,
                 apostr_contract_regex=None,
                 apostr_possess_marker=None,
                 apostr_possess_collapse=None,
                 apostr_possess_regex=None,
                 hyphen_collapse=None,
                 hyphen_regex=None,
                 punc_trans=None,
                 final_replacement=None,
                 sym_spell=None,
                 max_lookup_edit_distance=None,
                 remove_created_words=None,
                 word_to_remove=None,
                 stemmer_or_lemmatizer=None
                 ):
        try:
            self.load(name, root_path)
        except FileNotFoundError:
            self.name = name

            self.root_path = root_path

            self.do_noise = do_noise
            self.do_scrub_token = do_scrub_token
            self.do_acronym = do_acronym
            self.do_number = do_number
            self.do_apostr_contract = do_apostr_contract
            self.do_apostr_possess = do_apostr_possess
            self.do_hyphen = do_hyphen
            self.do_punc = do_punc
            self.do_lower = do_lower
            self.do_final_replacement = do_final_replacement
            self.do_other_special_char = do_other_special_char
            self.do_spell_correct = do_spell_correct
            self.do_remove_word = do_remove_word
            self.do_stem_or_lemmatize = do_stem_or_lemmatize

            self.verbose = verbose

            self.noise_marker = noise_marker
            self.noise_regex = noise_regex
            self.scrub_token_marker = scrub_token_marker
            self.scrub_token_collapse = scrub_token_collapse
            self.scrub_token_regex = scrub_token_regex
            self.acronym_marker = acronym_marker
            self.acronym_regex = acronym_regex
            self.number_marker = number_marker
            self.number_regex = number_regex
            self.apostr_contract_collapse = apostr_contract_collapse
            self.apostr_contract_regex = apostr_contract_regex
            self.apostr_possess_marker = apostr_possess_marker
            self.apostr_possess_collapse = apostr_possess_collapse
            self.apostr_possess_regex = apostr_possess_regex
            self.hyphen_collapse = hyphen_collapse
            self.hyphen_regex = hyphen_regex
            self.punc_trans = punc_trans
            self.final_replacement = final_replacement
            self.sym_spell = sym_spell
            self.max_lookup_edit_distance = max_lookup_edit_distance
            self.remove_created_words = remove_created_words
            self.word_to_remove = word_to_remove
            self.stemmer_or_lemmatizer = stemmer_or_lemmatizer

            self.created_words = set()
            self.digit_to_word = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                                  '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

    def save(self, removed=None):
        if removed is None:
            utils.save_list_dict(self.__dict__, '{:}_saved'.format(self.name), self.root_path)
            print('{0:} saved to {1:}'.format(self.name, self.root_path))
        else:
            assert isinstance(removed, list) and set(removed).issubset(set(self.__dict__.keys()))
            dict_to_save = {k: v for k, v in self.__dict__.items() if k not in removed}
            utils.save_list_dict(dict_to_save, '{:}_saved'.format(self.name), self.root_path)
            print('{0:} saved to {1:} with {2:} removed'.format(self.name, self.root_path, ', '.join(removed)))

    def load(self, name, root_path):
        self.__dict__ = utils.load_list_dict('{:}_saved'.format(name), root_path)
        old_root_path = self.root_path.rstrip('/')
        new_root_path = root_path.rstrip('/')
        if old_root_path != new_root_path:
            self.change_root_path(new_root_path)
        print('{0:} loaded from {1:}'.format(name, root_path))

    def change_name(self, new_name):
        old_name = self.name
        self.name = new_name
        print('name changed from {0:} to {1:}'.format(old_name, new_name))

    def change_root_path(self, new_root_path):
        new_root_path = new_root_path.rstrip('/')
        old_root_path = self.root_path.rstrip('/')
        self.root_path = new_root_path
        print('root_path changed from {0:} to {1:}'.format(old_root_path, new_root_path))

    def _replace_matchobj(self, matchobj, new_without_marker, marker):
        old = matchobj.group()
        new = new_without_marker + marker
        if new != '':
            assert all_x_in_y(new, string.ascii_letters)
            self.created_words.add(new)
        if self.verbose:
            print('{0:} --> {1:}'.format(old, new))
        return ' ' + new + ' '

    def _collapse_matchobj(self, matchobj, remove, marker, map_digit_to_word=False):
        old = matchobj.group()
        if map_digit_to_word:
            old = old.translate(str.maketrans(self.digit_to_word))
        new = old.translate(str.maketrans('', '', remove)) + marker
        if new != '':
            assert all_x_in_y(new, string.ascii_letters)
            self.created_words.add(new)
        if self.verbose:
            print('{0:} --> {1:}'.format(old, new))
        return ' ' + new + ' '

    def _separate_matchobj(self, matchobj, remove, map_digit_to_word=False):
        old = matchobj.group()
        if map_digit_to_word:
            old = old.translate(str.maketrans(self.digit_to_word))
        new = old.translate(str.maketrans(remove, ' ' * len(remove)))
        assert all(all_x_in_y(i, string.ascii_letters) for i in new.split())
        if self.verbose:
            print('{0:} --> {1:}'.format(old, new))
        return ' ' + new + ' '

    def _apostr_contract(self, matchobj, prefix, suffix, apostr_len, collapse):
        old = matchobj.group()
        if prefix is None:
            prefix = old[:-apostr_len]
        if collapse:
            new = prefix + suffix
            assert all_x_in_y(new, string.ascii_letters)
            self.created_words.add(new)
        else:
            new = prefix + ' ' + suffix
            assert all_x_in_y(prefix, string.ascii_letters)
            assert all_x_in_y(suffix, string.ascii_letters)
        if self.verbose:
            print('{0:} --> {1:}'.format(old, new))
        return ' ' + new + ' '

    def _apostr_possess(self, matchobj, marker, collapse):
        old = matchobj.group()
        if collapse:
            new = old[:-2] + marker
            assert all_x_in_y(new, string.ascii_letters)
            self.created_words.add(new)
        else:
            new = old[:-2]
            assert all_x_in_y(new, string.ascii_letters)
        if self.verbose:
            print('{0:} --> {1:}'.format(old, new))
        return ' ' + new + ' '

    def _handle_noise(self, text):
        if not self.do_noise:
            return text
        for noise, regex in self.noise_regex.items():
            for r in regex:
                text = re.sub(r, lambda x: self._replace_matchobj(x, noise, self.noise_marker), text)
        return text

    def _handle_scrub_token(self, text):
        if not self.do_scrub_token:
            return text
        if self.scrub_token_collapse:
            return re.sub(self.scrub_token_regex, lambda x: self._collapse_matchobj(x, chr(8226) + '_<>', self.scrub_token_marker), text)
        else:
            return re.sub(self.scrub_token_regex, lambda x: self._replace_matchobj(x, 'SCRUBTOKEN', self.scrub_token_marker), text)

    def _handle_acronym(self, text):
        if not self.do_acronym:
            return text
        return re.sub(self.acronym_regex, lambda x: self._collapse_matchobj(x, '. ', self.acronym_marker, True), text)

    def _handle_number(self, text):
        if not self.do_number:
            return text
        return re.sub(self.number_regex, lambda x: self._replace_matchobj(x, '', self.number_marker), text)

    def _handle_apostr_contract(self, text):
        if not self.do_apostr_contract:
            return text
        for _, regex in self.apostr_contract_regex.items():
            text = re.sub(regex[0], lambda x: self._apostr_contract(x, regex[1], regex[2], regex[3], self.apostr_contract_collapse), text)
        return text

    def _handle_apostr_possess(self, text):
        if not self.do_apostr_possess:
            return text
        return re.sub(self.apostr_possess_regex, lambda x: self._apostr_possess(x, self.apostr_possess_marker, self.apostr_possess_collapse), text)

    def _handle_hyphen(self, text):
        if not self.do_hyphen:
            return text
        if self.hyphen_collapse:
            return re.sub(self.hyphen_regex, lambda x: self._collapse_matchobj(x, '-', ''), text)
        else:
            return re.sub(self.hyphen_regex, lambda x: self._separate_matchobj(x, '-'), text)

    def _handle_punc(self, text):
        if not self.do_punc:
            return text
        punc_trans_copy = copy.deepcopy(self.punc_trans)
        punc_kept = []
        for k, v in self.punc_trans.items():
            assert (len(k) == 1) and (k in string.punctuation)
            if all_x_in_y(v, string.ascii_letters):
                self.created_words.add(v)
            elif v != '':
                assert k == v
                punc_kept.append('\\' * int(v in {'-', ']'}) + v)
            punc_trans_copy[k] = ' ' + v + ' '
        text = text.translate(str.maketrans(punc_trans_copy))
        text = re.sub(r'(?ius)([^a-zA-Z0-9' + ''.join(punc_kept) + '])', ' ', text)
        return text

    def _lower(self, text):
        if not self.do_lower:
            return text
        self.created_words = {k.lower() for k in self.created_words}
        return text.lower()

    def _final_replacement(self, text):
        if not self.do_final_replacement:
            return text
        for replace_with, regex in self.final_replacement.items():
            text = re.sub(regex, lambda x: self._replace_matchobj(x, replace_with, ''), text)
        return text

    def _other_special_char(self, text):
        if not self.do_other_special_char:
            return text
        if re.search(r'(?ias)([^a-zA-Z0-9 !"#$%&\'()*+,./:;<=>?@[\]^_`{|}~-])', text) is not None:
            return ''
        else:
            return text

    def denoise_and_tokenize_single(self, text):
        if not is_nonempty_str(text):
            return text
        text = self._handle_noise(text)
        text = self._handle_scrub_token(text)
        text = self._handle_acronym(text)
        text = self._handle_number(text)
        text = self._handle_apostr_contract(text)
        text = self._handle_apostr_possess(text)
        text = self._handle_hyphen(text)
        text = self._handle_punc(text)
        text = self._lower(text)
        text = self._final_replacement(text)
        text = self._other_special_char(text)
        return ' '.join(text.split())

    def spell_correct_single(self, text):
        if (not self.do_spell_correct) or (not is_nonempty_str(text)):
            return text
        assert text.islower()
        # self.max_lookup_edit_distance: max edit distance per lookup (<= max_dictionary_edit_distance)
        words_do_not_correct = self.created_words - set(self.sym_spell._words.keys())
        for k in words_do_not_correct:
            assert self.sym_spell.create_dictionary_entry(key=k, count=1)
        suggestions = self.sym_spell.lookup_compound(text, self.max_lookup_edit_distance, transfer_casing=True)
        for k in words_do_not_correct:
            assert self.sym_spell.delete_dictionary_entry(key=k)
        assert len(suggestions) == 1
        text = suggestions[0].term.replace('\ufeff', '')
        return text

    def remove_word_single(self, text):
        if (not self.do_remove_word) or (not is_nonempty_str(text)):
            return text
        if self.remove_created_words:
            text = ' '.join([w for w in text.split() if w not in (self.word_to_remove | self.created_words)])
        else:
            text = ' '.join([w for w in text.split() if w not in self.word_to_remove])
        return text

    def stem_or_lemmatize_single(self, text):
        if (not self.do_stem_or_lemmatize) or (not is_nonempty_str(text)):
            return text
        text = ' '.join([self.stemmer_or_lemmatizer(w) for w in text.split()])
        return text

    def preprocess_single(self, text):
        self.created_words = set()
        if not is_nonempty_str(text):
            return text
        text_original = text
        text = self.denoise_and_tokenize_single(text)
        text = self.spell_correct_single(text)
        text = self.remove_word_single(text)
        text = self.stem_or_lemmatize_single(text)
        text = self.remove_word_single(text)  # stemming and lemmatization may yield some words in self.word_to_remove, so we need to remove these words again
        if self.verbose and (text_original == text):
            print('Text intact during preprocessing')
        return text

    def preprocess(self, text_df, input_col):
        tic = time.time()
        output_col = input_col + '_preprocessed'
        text_df[output_col] = text_df[input_col].map(self.preprocess_single)
        if self.verbose and (text_df[output_col] == text_df[input_col]).all():
            print('Text intact during preprocessing')
        toc = time.time()
        mask_empty = (text_df[output_col].str.len() == 0)
        print('{0:d} preprocessed utterances obtained in {1:.2f} second(s), where {2:d} utterances contain special characters '
              'or become empty after preprocessing so they are converted to empty strings'.format(text_df.shape[0], toc - tic, mask_empty.sum()))
        return text_df, output_col

