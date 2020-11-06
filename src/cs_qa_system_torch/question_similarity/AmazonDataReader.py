from sentence_transformers.readers import InputExample
import csv
import gzip
import os


class AmazonDataReader(object):
    """
    Reads the amazon dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        f = open(os.path.join(self.dataset_folder, filename+'.tsv'), 'r', encoding='utf-8')
        data = csv.DictReader(f, delimiter='\t')
        examples = []
        id = 0
        for row in data:
            guid = "%s-%s-%s" % (filename, row['pid'], row['qid'])
            id += 1
            examples.append(InputExample(guid=guid, texts=[row['query'], row['passage']], label=self.map_label(row['label'])))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"0": 0, "1": 1}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]