import glob
import os
import time
import argparse
import numpy as np
from tqdm import tqdm, trange
import random
import shutil

import logging
from ir_metrics import get_mrr
from ir_utils import get_ir_model_attributes, get_ir_data_transform, load_document_collection, \
    get_context_embeddings, evaluate_ranking_model, get_loss_function
from utils import logging_config



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import constants
from ir_dataset import RankingDataset, CombinedRankingDataset, BiencoderRankingDataset, MSMARCOTripletDataset
from ir_transform import RankingTransform, CombinedRankingTransform

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu != '':
      torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader, threshold=0.5, loss_fct=nn.BCELoss()):
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for eval_step, eval_batch in enumerate(dataloader, start=1):
        input_batch = tuple(t.to(device) for t in eval_batch[:-2])
        label_batch = eval_batch[-2].to(device)
        with torch.no_grad():
            out = model(*input_batch)
            e_loss = loss_fct(*out, label_batch)

        if isinstance(out,tuple):
            if len(out) > 1:
                scores = torch.sum(out[0] * out[1], dim=1)
            else:
                scores = out[0]
        else:
            scores = out
        eval_hit_times += torch.sum((scores >= threshold).float())
        eval_loss += e_loss.sum()

        nb_eval_examples += label_batch.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_hit_times*1.0 / nb_eval_examples
    result = {
        'train_loss': tr_loss / nb_tr_steps,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'epoch': epoch,
        'global_step': global_step,
    }
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str, help='Directory to store all the model artifacts')
    parser.add_argument('--gpu', type=str, default="", help = 'gpus to use for training')
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, help="This will be used as a "
                                                                                            "base model for Cross "
                                                                                            "encoder architecture. "
                                                                                            "For Biencoder this will "
                                                                                            "be used as context encder")
    parser.add_argument("--query_model_name_or_path", default=None, type=str, help="This will be used only for "
                                                                                   "Biencoder "
                                                                                   "as query encoder")

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--architecture", required=True, type=str, help='[bi, cross]')
    parser.add_argument("--projection_dim", default=0, type=int, help="Extra linear layer on top of standard bert/roberta encoder")

    parser.add_argument("--train_data_path", default='data', type=str)
    parser.add_argument("--test_data_path", default='data', type=str)
    parser.add_argument(
        "--use_hard_negatives", action="store_true", help="Set this to the json format train data path file with hard negatives for each query."
    )
    parser.add_argument("--hard_negatives_weight_factor", default=0.0, type=float, help="weight factor for hard negatives")

    parser.add_argument("--max_query_passage_length", default=512, type=int, help='Required for cross encoder')
    parser.add_argument("--max_passage_length", default=384, type=int, help='Not Required for CrossEncoder architecture.It uses max_query_passage_length argument')
    parser.add_argument("--max_query_length", default=32, type=int, help='Not Required for CrossEncoder architecture.It uses max_query_passage_length argument')


    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=2, type=int, help="Total batch size for eval.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--print_freq", default=100, type=int, help="The frequency to print output")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")


    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=2000, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--loss", default='BCE', type=str, help = "['BCE', 'BiEncoderNLL', BiEncoderBCE]")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != '' else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device


    set_seed(args)

    # Setup logging
    logging_config(args.output_dir, name='train')
    logger.info('\n'*3 + '=' * 80 + '\n')
    logger.info("The input args are: \n{}".format(args))
    logger.info('\n' + '=' * 80 + '\n'*3)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pre_model_save_dir = os.path.join(args.output_dir, constants.PRE_MODEL_FOLDER)
    query_pre_model_save_dir = os.path.join(args.output_dir, constants.QUERY_PRE_MODEL_FOLDER)
    if not os.path.exists(pre_model_save_dir):
        os.makedirs(pre_model_save_dir)
    if args.query_model_name_or_path and not os.path.exists(query_pre_model_save_dir):
        os.makedirs(query_pre_model_save_dir)

    logger.info('\n'*3 + '=' * 80 + '\n')
    logger.info("The Model artifacts will be saved to: {}".format(args.output_dir))
    logger.info('\n' + '=' * 80 + '\n'*3)

    ## init dataset and ir model
    model, tokenizer, config, query_tokenizer, query_config = get_ir_model_attributes(model_name_or_path=args.model_name_or_path,
                                                                      query_model_name_or_path=args.query_model_name_or_path,
                                                                      architecture=args.architecture,
                                                                      projection_dim=args.projection_dim,
                                                                      do_lower_case=args.do_lower_case,
                                                                      device = device)

    transform, query_transform = get_ir_data_transform(args.architecture, tokenizer, query_tokenizer,
                                                       args.max_query_passage_length, args.max_query_length, args.max_passage_length)

    logger.info('\n'*3 + '=' * 80 + '\n')
    logger.info('Train dir: {}'.format(args.train_data_path))
    logger.info('Output dir: {}'.format(args.output_dir))
    logger.info('\n' + '=' * 80 + '\n'*3)

    if args.architecture == 'cross':
        train_dataset = CombinedRankingDataset(args.train_data_path, transform)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True, collate_fn=train_dataset.batchify_join_str, drop_last=True)
    elif args.architecture == 'bi' or args.architecture == 'single':
        if args.use_hard_negatives:
            train_dataset = BiencoderRankingDataset(args.train_data_path,
                                           query_transform, transform)
        else:
            train_dataset = RankingDataset(args.train_data_path,
                                           query_transform, transform)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True, collate_fn=train_dataset.batchify_join_str, drop_last=True)
    elif args.architecture == 'bi-msmarco-triplet':
        train_dataset = MSMARCOTripletDataset(collection_filepath = os.path.join(os.path.dirname(args.train_data_path), 'collection.tsv'),
                                              queries_filepath = os.path.join(os.path.dirname(args.train_data_path), 'queries.train.tsv'),
                                              qidpidtriples_filepath = args.train_data_path,
                                              query_transform = query_transform,
                                              context_transform = transform)
        MSMARCOTripletDataset.msmarco_create_dev_data_from_triplets(collection_filepath = os.path.join(os.path.dirname(args.test_data_path), 'collection.tsv'),
                                              queries_filepath = os.path.join(os.path.dirname(args.test_data_path), 'queries.train.tsv'),
                                              triplet_filepath = args.test_data_path,
                                              num_dev_queries = 500,
                                              num_max_dev_negatives = 200)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True, collate_fn=train_dataset.batchify_join_str, drop_last=True)
    else:
        raise ValueError("Wrong architecture name")

    t_total = len(train_dataloader) // args.gradient_accumulation_steps  * args.num_train_epochs
    print_freq = args.print_freq
    logger.info('\n'*3 + '=' * 80 + '\n')
    logger.info('print freq steps: {}'.format(print_freq))
    logger.info('checkpoint save steps: {}'.format(args.save_steps))
    logger.info('\n' + '=' * 80 + '\n'*3)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            checkpoint = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.pt'), map_location=device)
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")
            if hasattr(model, "module"):
                model.module.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.pt'), map_location=device))
            else:
                model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.pt'), map_location=device))



    #define the loss function to use
    loss_function = get_loss_function(args.loss)

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
    )

    model.zero_grad()
    for epoch in train_iterator:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader)) as bar:
            for step, batch in enumerate(train_dataloader):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()

                ##### Make changes based on New Architectures ############
                # Input_batch and label_batch is based on defined dataset and corresponding batchify
                # make sure model output and inputs passed to lss function are accurate
                inputs_batch = tuple(t.to(device) for t in batch[:-2])
                labels_batch = batch[-2].to(device)
                output = model(*inputs_batch)
                if args.use_hard_negatives and args.hard_negatives_weight_factor > 0:
                    loss = loss_function(*output, labels_batch, batch[-1], args.hard_negatives_weight_factor)
                else:
                    loss = loss_function(*output, labels_batch)
                ###### END of changes based on new Architecture ###############
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item() # .item() function copies to cpu and converts the tensory to plain python number
                nb_tr_examples += inputs_batch[0].size(0)
                nb_tr_steps += 1

                loss.backward()

                global_step += 1

                ## Update progress bar
                if print_freq > 0 and global_step % print_freq == 0:
                    bar.update(min(print_freq, step))
                    logger.info('global step: {},\tepoch: {},\tnumber_of_training_samples: {},'
                                '\ttotal_loss: {}\n'.format(global_step, epoch, nb_tr_examples, tr_loss / nb_tr_steps))

                ## Save check points
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    tokenizer.save_pretrained(os.path.join(output_dir, constants.PRE_MODEL_FOLDER))
                    config.save_pretrained(os.path.join(output_dir, constants.PRE_MODEL_FOLDER))
                    if 'bi' in args.architecture:
                        query_tokenizer.save_pretrained(os.path.join(output_dir, constants.QUERY_PRE_MODEL_FOLDER))
                        query_config.save_pretrained(os.path.join(output_dir, constants.QUERY_PRE_MODEL_FOLDER))
                    model_to_save = model.module if hasattr(model, "module") else model
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(output_dir,'pytorch_model.pt'))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

        ## Saving the model after each epoch
        tokenizer.save_pretrained(pre_model_save_dir)
        config.save_pretrained(pre_model_save_dir)
        if 'bi' in args.architecture:
            query_tokenizer.save_pretrained(query_pre_model_save_dir)
            query_config.save_pretrained(query_pre_model_save_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.pt'))
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving model to {} after epoch {}".format(args.output_dir, epoch))
        ## Removing checkpoint folders
        dir_list = glob.iglob(os.path.join(args.output_dir, "checkpoint-*"))
        for path in dir_list:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=False, onerror=None)

        # add a eval step after each epoch
        if os.path.isfile(os.path.join(os.path.dirname(args.test_data_path), 'dev-collection.json')):
            logger.info('Evaluating the model on dev set. Collection : {},\nqrels : {}'.format(os.path.join(os.path.dirname(args.test_data_path), 'dev-collection.json'),
                                                                                               os.path.join( os.path.dirname(args.test_data_path),'dev-qrels.json')))
            dev_collection_ir, dev_idx_to_doc, dev_doc_to_idx, _ = load_document_collection(os.path.join(os.path.dirname(args.test_data_path), 'dev-collection.json'))
            dev_collection_tuple = (dev_collection_ir, dev_idx_to_doc, dev_doc_to_idx)
            transform, query_transform = get_ir_data_transform(args.architecture, tokenizer, query_tokenizer,
                                                               args.max_query_passage_length, args.max_query_length,
                                                               args.max_passage_length,
                                                               bool_np_array=True)
            context_embeddings = get_context_embeddings(dev_idx_to_doc, transform, model, args.architecture, args.test_batch_size, device)
            dev_top_n = 10
            dev_rank_threshold_score = 0.0
            dev_rerank = False
            dev_rerank_score_weight = 0.5
            n_mrr = 10
            proportion_rank_index = [5,3,2]
            rank_results_df = evaluate_ranking_model(os.path.join(os.path.dirname(args.test_data_path), 'dev-qrels.json'),
                                                     dev_collection_tuple,
                                                     args.architecture,
                                                     model,
                                                     dev_top_n,
                                                     query_transform,
                                                     transform,
                                                     context_embeddings,
                                                     args.test_batch_size,
                                                     device,
                                                     dev_rank_threshold_score,
                                                     dev_rerank,
                                                     dev_rerank_score_weight)

            mrr, proportion_ranks = get_mrr(rank_results_df, n_mrr, proportion_rank_index)
            logger.info('\n' * 3 + '=' * 40 + 'Evaluation Metrics' + '=' * 40 + '\n')
            logger.info('mrr@{}: {}'.format(n_mrr,mrr))
            for i,rank in enumerate(proportion_rank_index):
                logger.info('proportion rank@{}: {}'.format(rank, proportion_ranks[i]))
            logger.info('\n' + '=' * 80 + '\n' * 3)

