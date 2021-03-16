import glob
import os
import time
import argparse
import numpy as np
from tqdm import tqdm, trange
import random
import shutil

import logging

from ir_loss import BiEncoderNllLoss, BiEncoderBCELoss
from ir_model import BiEncoderModel, CrossEncoderModel
from ir_utils import get_ir_model_attributes
from utils import logging_config



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import constants
from ir_dataset import RankingDataset, CombinedRankingDataset, BiencoderRankingDataset
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

    parser.add_argument("--max_query_passage_length", default=512, type=int, help='Required for cross encoder')
    parser.add_argument("--max_passage_length", default=384, type=int, help='Not Required for CrossEncoder architecture.It uses max_query_passage_length argument')
    parser.add_argument("--max_query_length", default=32, type=int, help='Not Required for CrossEncoder architecture.It uses max_query_passage_length argument')


    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=2, type=int, help="Total batch size for eval.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--print_freq", default=500, type=int, help="The frequency to print output")
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
    logger.info("The input args are: \n{}".format(args))


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pre_model_save_dir = os.path.join(args.output_dir, constants.PRE_MODEL_FOLDER)
    query_pre_model_save_dir = os.path.join(args.output_dir, constants.QUERY_PRE_MODEL_FOLDER)
    if not os.path.exists(pre_model_save_dir):
        os.makedirs(pre_model_save_dir)
    if args.query_model_name_or_path and not os.path.exists(query_pre_model_save_dir):
        os.makedirs(query_pre_model_save_dir)

    logger.info("The Model artifacts will be saved to: {}".format(args.output_dir))


    ## init dataset and ir model
    model, tokenizer, config, query_tokenizer, query_config = get_ir_model_attributes(model_name_or_path=args.model_name_or_path,
                                                                      query_model_name_or_path=args.query_model_name_or_path,
                                                                      architecture=args.architecture,
                                                                      projection_dim=args.projection_dim,
                                                                      do_lower_case=args.do_lower_case)

    transform = None
    query_transform = None
    context_transform = None
    if 'cross' in args.architecture:
        transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=args.max_query_passage_length)
    else:
        query_transform = RankingTransform(tokenizer=query_tokenizer, max_len=args.max_query_length)
        context_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_passage_length)

    logger.info('=' * 80)
    logger.info('Train dir: {}'.format(args.train_data_path))
    logger.info('Output dir: {}'.format(args.output_dir))
    logger.info('=' * 80)

    if 'cross' in args.architecture:
        train_dataset = CombinedRankingDataset(args.train_data_path, transform)
        val_dataset = CombinedRankingDataset(args.test_data_path, transform)
    else:
        if args.use_hard_negatives:
            train_dataset = BiencoderRankingDataset(args.train_data_path,
                                           query_transform, context_transform)
        else:
            train_dataset = RankingDataset(args.train_data_path,
                                           query_transform, context_transform)
        val_dataset = RankingDataset(args.test_data_path,
                                         query_transform, context_transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True, collate_fn=train_dataset.batchify_join_str)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.test_batch_size, shuffle=False,
                                collate_fn=val_dataset.batchify_join_str)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps  * args.num_train_epochs
    # tr_total = int(
    #     train_dataset.__len__() / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    print_freq = args.print_freq
    logger.info('Print freq:'.format(print_freq))

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
            checkpoint = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")
            model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.pt')))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.zero_grad()

    #
    #loss_function = nn.BCEWithLogitsLoss()
    if args.loss == 'BCE':
        loss_function = nn.BCELoss()
    elif args.loss == 'BiEncoderNLL':
        loss_function = BiEncoderNllLoss()
    elif args.loss == 'BiEncoderBCE':
        loss_function = BiEncoderBCELoss()
    else:
        loss_function = nn.BCELoss()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
    )

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
                optimizer.zero_grad()
                inputs_batch = tuple(t.to(device) for t in batch[:-2])
                labels_batch = batch[-2].to(device)
                output = model(*inputs_batch)
                loss = loss_function(*output, labels_batch)


                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                nb_tr_examples += inputs_batch[0].size(0)
                nb_tr_steps += 1


                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()

                global_step += 1

                if print_freq > 0 and global_step % print_freq == 0:
                    bar.update(min(print_freq, step))
                    logger.info('global step: {},\t total_loss: {}\n'.format(global_step, tr_loss / nb_tr_steps))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    #pre_model_to_save = pre_model.module if hasattr(pre_model, "module") else pre_model
                    #pre_model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(os.path.join(output_dir, constants.PRE_MODEL_FOLDER))
                    config.save_pretrained(os.path.join(output_dir, constants.PRE_MODEL_FOLDER))
                    if args.architecture == 'bi':
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

        #pre_model_to_save = pre_model.module if hasattr(pre_model, "module") else pre_model
        #pre_model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(pre_model_save_dir)
        config.save_pretrained(pre_model_save_dir)
        if args.architecture == 'bi':
            query_tokenizer.save_pretrained(query_pre_model_save_dir)
            query_config.save_pretrained(query_pre_model_save_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.pt'))
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving model to %s", args.output_dir)
        ## Removing checkpoint folders
        dir_list = glob.iglob(os.path.join(args.output_dir, "checkpoint-*"))
        for path in dir_list:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=False, onerror=None)

        # add a eval step after each epoch
        if 'cross' in args.architecture == 'cross':
            eval_loss_function = nn.BCELoss()
        else:
            eval_loss_function = BiEncoderBCELoss()
        val_result = eval_running_model(val_dataloader,loss_fct=eval_loss_function)
        logger.info('Epoch {}, Global Step {} VAL res:{}\n'.format(epoch, global_step, val_result))
        logger.info(str(val_result) + '\n')

        # if val_result['eval_loss'] <= best_eval_loss:
        #     best_eval_loss = val_result['eval_loss']
        #     val_result['best_eval_loss'] = best_eval_loss
        #     # save model
        #     print('[Saving model at]', state_save_path)
        #     log_wf.write('[Saving model at] %s\n' % state_save_path)
        #     torch.save(model.module.state_dict(), state_save_path)



        logger.info('global step: {},\t total_loss: {}\n'.format(global_step, tr_loss / nb_tr_steps))

