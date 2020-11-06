import string
import copy
import english_preprocessing_constants
from english_preprocessing import EnglishPreprocessing

class WordTokenizerFactory:
    """
    This factory class returns a word tokenizer function depending on the input argument
    """
    @staticmethod
    def create_word_tokenizer(word_tokenizer_name: str):
        if word_tokenizer_name == 'simple_word_tokenizer':
            return simple_word_tokenizer
        elif word_tokenizer_name == 'simple_word_tokenizer_no_stopwords':
            return simple_word_tokenizer_no_stopwords
        elif word_tokenizer_name == 'simple_word_tokenizer_no_stopwords_stem':
            return simple_word_tokenizer_no_stopwords_stem
        elif word_tokenizer_name == 'advanced_word_tokenizer':
            return advanced_word_tokenizer
        else:
            raise NotImplementedError()


def simple_word_tokenizer(x):
    # Tokenize a text x after lowering all its letters, and removing punctuation at the beginning and end of the text
    return x.strip(string.punctuation).lower().split()


def simple_word_tokenizer_no_stopwords(x):
    # Tokenize a text x after lowering all its letters, removing punctuation at the beginning and end of the text, and
    # removing stopwords
    return [w for w in x.strip(string.punctuation).lower().split() if w not in english_preprocessing_constants.word_to_remove_bm25]


def simple_word_tokenizer_no_stopwords_stem(x):
    result = [english_preprocessing_constants.stemmer_or_lemmatizer(w) for w in x.strip(string.punctuation).lower().split()
              if w not in english_preprocessing_constants.word_to_remove_bm25]
    return [w for w in result if w not in english_preprocessing_constants.word_to_remove_bm25]


english_preprocessor = EnglishPreprocessing(
    name='english_preprocessor',
    root_path='None',
    do_noise=True,
    do_scrub_token=False,
    do_acronym=False,
    do_number=True,
    do_apostr_contract=True,
    do_apostr_possess=True,
    do_hyphen=True,
    do_punc=True,
    do_lower=True,
    do_final_replacement=True,
    do_other_special_char=False,
    do_spell_correct=False,
    do_remove_word=True,
    do_stem_or_lemmatize=False,
    verbose=False,
    noise_marker=english_preprocessing_constants.noise_marker,
    noise_regex=english_preprocessing_constants.noise_regex,
    scrub_token_marker=english_preprocessing_constants.scrub_token_marker,
    scrub_token_collapse=english_preprocessing_constants.scrub_token_collapse,
    scrub_token_regex=english_preprocessing_constants.scrub_token_regex,
    acronym_marker=english_preprocessing_constants.acronym_marker,
    acronym_regex=english_preprocessing_constants.acronym_regex,
    number_marker=english_preprocessing_constants.number_marker,
    number_regex=english_preprocessing_constants.number_regex,
    apostr_contract_collapse=english_preprocessing_constants.apostr_contract_collapse,
    apostr_contract_regex=english_preprocessing_constants.apostr_contract_regex,
    apostr_possess_marker=english_preprocessing_constants.apostr_possess_marker,
    apostr_possess_collapse=english_preprocessing_constants.apostr_possess_collapse,
    apostr_possess_regex=english_preprocessing_constants.apostr_possess_regex,
    hyphen_collapse=english_preprocessing_constants.hyphen_collapse,
    hyphen_regex=english_preprocessing_constants.hyphen_regex,
    punc_trans=english_preprocessing_constants.punc_trans,
    final_replacement=english_preprocessing_constants.final_replacement,
    sym_spell=copy.deepcopy(english_preprocessing_constants.sym_spell),
    max_lookup_edit_distance=english_preprocessing_constants.max_lookup_edit_distance,
    remove_created_words=english_preprocessing_constants.remove_created_words,
    word_to_remove=english_preprocessing_constants.word_to_remove,
    stemmer_or_lemmatizer=english_preprocessing_constants.stemmer_or_lemmatizer
)


def advanced_word_tokenizer(x):
    return english_preprocessor.preprocess_single(x).split()
