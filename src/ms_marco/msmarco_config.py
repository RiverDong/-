# THIS FILE CONTAINS ALL THE MSMARCO DATA ACCESS CONFIGURATIONS ##################

ROOT_PATH = '/data/QAData/MSMARCO'
PASSAGE_FILE = 'collection.tsv'
QUERIES_TRAIN = 'queries.train.tsv'
QRELS_TRAIN = 'qrels.train.tsv'
QID_PID_REL_NONREL_PAIRS = 'qidpidtriples.train.full.tsv'

QUERIES_DEV = 'queries.dev.tsv'
QRELS_DEV = 'qrels.dev.tsv'
TOP1000_DEV = 'top1000.dev.tsv'

QUERIES_EVAL = 'queries.eval.tsv'

OUTPUT_TRAIN_FILE = 'train.tsv'
OUTPUT_TEST_FILE = 'test.tsv'
OUTPUT_VALID_FILE = 'valid.tsv'