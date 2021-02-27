# available vocab
VOCAB_BASE = 'vocab_base'
VOCAB_BERT = 'vocab_bert'
VOCAB_DISTIL_BERT = 'vocab_distil_bert'

VOCABS = {
    VOCAB_BASE: {
        'max_size': 30000,
        'lower': False
    },
    VOCAB_BERT: {
        'name': 'bert_12_768_12',
        'dataset_name': 'book_corpus_wiki_en_cased'
    },
    VOCAB_DISTIL_BERT: {
        'name': 'distilbert_6_768_12',
        'dataset_name': 'distilbert_book_corpus_wiki_en_uncased'
    }
}

# available Sequence transform
TRANSFORM_BASE = 'transform_base'
TRANSFORM_BERT = 'transform_bert'
TRANSFORM_DISTIL_BERT = 'transform_distil_bert'
TRANSFORMS = {
    TRANSFORM_BASE: {
        'max_seq_length': 512,
        'pad': True,
        'lower': False
    },
    TRANSFORM_BERT: {
        'max_seq_length': 512,
        'pad': True,
        'lower': False
    },
    TRANSFORM_DISTIL_BERT : {
        'max_seq_length': 512,
        'pad': True,
        'lower': False
    }
}

# available Encoders
ENCODER_BASE = 'encoder_base'
ENCODER_PRETRAINED_BERT = 'encoder_pretrained_bert'
ENCODER_PRETRAINED_BERT_POLY = 'encoder_pretrained_bert_poly'
ENCODER_PRETRAINED_DISTIL_BERT = 'encoder_pretrained_distil_bert'
ENCODERS = {
    ENCODER_BASE: {
        'num_reps': 1,
        'units': 512,
        'num_layers': 2,
        'hidden_size': 2048,
        'max_length': 512,
        'num_heads': 4,
        'dropout': 0.0,
        'precision_type': 'float32'
    },
    ENCODER_PRETRAINED_BERT: {
        'name': 'bert_12_768_12',
        'dataset_name': 'book_corpus_wiki_en_cased'
    },
    ENCODER_PRETRAINED_DISTIL_BERT: {
        'name': 'distilbert_6_768_12',
        'dataset_name': 'distilbert_book_corpus_wiki_en_uncased'
    },
    ENCODER_PRETRAINED_BERT_POLY: {
        'name': 'bert_12_768_12',
        'dataset_name': 'book_corpus_wiki_en_cased',
        'num_reps': 1,
        'units': 512,
        'max_length':512,
        'precision_type':'float32'
    }
}

RANKING_BIENCODER = 'ranking_biencoder'
RANKING_CROSSENCODER = 'ranking_cross_encoder'
RANKING_POLYENCODER = 'ranking_poly_encoder'
RANKING_MODELS = {
    RANKING_BIENCODER: {},
    RANKING_CROSSENCODER: {},
    RANKING_POLYENCODER: {}
}

# available IR models
IR_BM25OKAPI = 'BM25Okapi'
# Recommended combinations of (k1, b): (1.2, 0.75), (0.6, 0.8), (0.5, 0.45)
IR_MODELS = {
    IR_BM25OKAPI: {
        'k1': 3.6,
        'b': 0.6,
        'delta': 0.0
    }
}

# loss function
LOSS_SIGMOID_BINARY_CROSS_ENTROPY = 'sigmoid_binary_cross_entropy'
LOSS_BINARY_CROSS_ENTROPY = 'binary_cross_entropy'

# Input Train/Test File Columns
RANKING_INPUT_QUERY_ID = 'qid'
RANKING_INPUT_DOCUMENT_ID = 'pid'
RANKING_INPUT_QUERY_NAME = 'query'
RANKING_INPUT_DOCUMENT_NAME = 'passage'
RANKING_INPUT_LABEL_NAME = 'label'
ANSWER_EXTRACTION_ANSWER = 'answer'
RANKING_INPUT_HARD_NEGATIVES = 'hard_negatives'

INFERENCE_PREDICTIONS = 'predictions'
INFERENCE_ACTUALS = 'actuals'
INFERECE_QUERY_IDS = 'query_ids'


TRANSFORM_TYPE_COMBINE = 'combine'
TRANSFORM_TYPE_SPLIT = 'split'
TRANSFORM_TYPE_SPLIT_QUERY = 'query'
TRANSFORM_TYPE_SPLIT_DOCUMENT = 'document'

RANKING_TYPE_CROSS = 'cross'
RANKING_TYPE_POLY = 'poly'
RANKING_TYPE_BI = 'bi'


PREDICTION_TYPE_TEST_FILE = 'TEST_FILE'
PREDICTION_TYPE_PREDICTION_FILE = 'PREDICT_FILE'
PREDICTION_TYPE_QUERY_FILE = 'QUERY_FILE'
PREDICTION_TYPE_QUERY = 'QUERY'

ARGUMENT_TRANSFORM_TYPE = 'transform_type'
ARGUMENT_TRANSFORM_NAME = 'transform_name'
ARGUMENT_TRANSFORM_PARAMS = 'transform_params'
ARGUMENT_QUERY_TRANSFORM_NAME = 'query_transform_name'
ARGUMENT_QUERY_TRANSFORM_PARAMS = 'query_transform_params'
ARGUMENT_DOCUMENT_TRANSFORM_NAME = 'document_transform_name'
ARGUMENT_DOCUMENT_TRANSFORM_PARAMS = 'document_transform_params'


ARGUMENT_ENCODER_NAME = 'encoder_name'
ARGUMENT_ENCODER_PARAMS = 'encoder_params'
ARGUMENT_QUERY_ENCODER_NAME = 'query_encoder_name'
ARGUMENT_QUERY_ENCODER_PARAMS = 'query_encoder_params'
ARGUMENT_DOCUMENT_ENCODER_NAME = 'document_encoder_name'
ARGUMENT_DOCUMENT_ENCODER_PARAMS = 'document_encoder_params'


ARGUMENT_RANKING_TYPE = 'ranking_type'
ARGUMENT_RANKING_MODEL_NAME = 'ranking_model_name'
ARGUMENT_RANKING_MODEL_PARAMS = 'ranking_model_params'