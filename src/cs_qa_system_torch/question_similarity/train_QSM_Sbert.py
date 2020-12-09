import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.chdir("/home/yqinamz/work/QA_BOT/CS-QASystem-Torch_old/src/CS-QASystem-Torch/src/cs_qa_system_torch")

from Reader.QA200DataReader import QA200DataReader
from torch.utils.data import DataLoader
import math
from sentence_transformers import losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description='This module is used to train Question Similarity model')

parser.add_argument('--pretrain_model_path', type=str,
                    help='path of pre-trained model')
parser.add_argument('--model_name', type=str,
                    help='model name')
parser.add_argument('--train_data_path', type=str,
                    help='training data path')
# parser.add_argument('--work_dir', type=str,
#                     help='working directory')

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def train_QSM_Sbert(args):
    # Read the dataset
    train_batch_size = 16
    num_epochs = 2
    model_save_path = args.model_name +'-sbert-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = SentenceTransformer(args.pretrain_model_path)
    qa200_reader = QA200DataReader(args.train_data_path)

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read All train dataset")
    train_num_labels = qa200_reader.get_num_labels()
    train_dataset = SentencesDataset(qa200_reader.get_examples('train'), model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.SoftmaxLoss(model = model, sentence_embedding_dimension = model.get_sentence_embedding_dimension(), num_labels = train_num_labels)


    logging.info("Read dev dataset")
    # dev_data = SentencesDataset(examples=qa200_reader.get_examples('dev'), model=model)
    # dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    # evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(qa200_reader.get_examples('dev'), name='dev')


    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))
    # Train the model
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=5000,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    train_QSM_Sbert(arg_parser)