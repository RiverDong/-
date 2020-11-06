
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from ranking_dataset import RankingDataset, CombinedRankingDataset
from ranking_model import BertBiEncoderModel, BertCrossEncoderModel
from ranking_transform import RankingTransform, CombinedRankingTransform
from ranking_utils import MODEL_CLASSES


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #   torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader, threshold=0.5):
    #loss_fct = nn.BCEWithLogitsLoss()
    loss_fct = nn.BCELoss()
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for eval_step, eval_batch in enumerate(dataloader, start=1):
        input_batch = tuple(t.to(device) for t in eval_batch[:-2])
        label_batch = eval_batch[-2].to(device)
        with torch.no_grad():
            out = model(*input_batch)
            e_loss = loss_fct(out, label_batch)

        eval_hit_times += torch.sum((out >= threshold).float())
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
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--model_type", default='bert', type=str, help='[bert, distilbert]')
    parser.add_argument("--model_type_name", default='bert-base-uncased', type=str)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_file", type=str)
    parser.add_argument("--architecture", required=True, type=str, help='[bi, cross]')

    parser.add_argument("--train_data_path", default='data', type=str)
    parser.add_argument("--test_data_path", default='data', type=str)

    parser.add_argument("--max_passage_length", default=256, type=int)
    parser.add_argument("--max_query_length", default=32, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=2, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=100, type=int, help="Total batch size for eval.")

    parser.add_argument("--poly_m", default=16, type=int, help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=2000, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--gpu', type=str, default="")
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args)


    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    ## init dataset and bert model
    if args.use_checkpoint:
        tokenizer = TokenizerClass.from_pretrained(args.checkpoint_dir)
    else:
        tokenizer = TokenizerClass.from_pretrained(args.model_type_name)
        tokenizer.save_pretrained(args.checkpoint_dir)
        tokenizer.save_pretrained(args.save_dir)

    transform = None
    query_transform = None
    context_transform = None
    if args.architecture == 'cross':
        transform = CombinedRankingTransform(tokenizer=tokenizer, max_len=512)
    else:
        query_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_query_length)
        context_transform = RankingTransform(tokenizer=tokenizer, max_len=args.max_passage_length)

    print('=' * 80)
    print('Train dir:', args.train_data_path)
    print('Output dir:', args.save_dir)
    print('=' * 80)

    if args.architecture == 'cross':
        train_dataset = CombinedRankingDataset(args.train_data_path, transform)
        val_dataset = CombinedRankingDataset(args.test_data_path, transform)
    else:
        train_dataset = RankingDataset(args.train_data_path,
                                       query_transform, context_transform)
        val_dataset = RankingDataset(args.test_data_path,
                                     query_transform, context_transform)
    print(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True, collate_fn=train_dataset.batchify_join_str)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.test_batch_size, shuffle=False,
                                collate_fn=val_dataset.batchify_join_str)
    t_total = len(train_dataloader) // args.train_batch_size * args.num_train_epochs

    epoch_start = 1
    global_step = 0
    best_eval_loss = float('inf')
    best_test_loss = float('inf')

    log_wf = open(os.path.join(args.save_dir, 'log.txt'), 'a', encoding='utf-8')

    state_save_path = os.path.join(args.save_dir, 'pytorch_model.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ########################################
    # ## build BERT encoder
    # ########################################

    bert_config = ConfigClass.from_pretrained(args.model_type_name)
    bert = BertModelClass.from_pretrained(args.model_type_name, config=bert_config)

    if args.architecture == 'poly':
        model = BertBiEncoderModel(bert_config, bert=bert)
    elif args.architecture == 'bi':
        model = BertBiEncoderModel(bert_config, bert=bert)
    elif args.architecture == 'cross':
        model = BertCrossEncoderModel(bert_config, bert=bert)
    else:
        raise Exception('Unknown architecture.')

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

    tr_total = int(
        train_dataset.__len__() / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    print_freq = args.print_freq
    print('Print freq:', print_freq)

    #loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.BCELoss()

    if args.use_checkpoint:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_file))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        loss = checkpoint['loss']

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader)) as bar:
            for step, batch in enumerate(train_dataloader, start=1):
                model.train()
                optimizer.zero_grad()
                inputs_batch = tuple(t.to(device) for t in batch[:-2])
                labels_batch = batch[-2].to(device)
                output = model(*inputs_batch)
                loss = loss_function(output, labels_batch)
                tr_loss += loss.item()
                nb_tr_examples += inputs_batch[0].size(0)
                nb_tr_steps += 1


                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()

                global_step += 1

                if step % print_freq == 0:
                    bar.update(min(print_freq, step))
                    time.sleep(0.01)
                    print(global_step, tr_loss / nb_tr_steps)
                    log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
                log_wf.flush()
                pass

        # add a eval step after each epoch
        val_result = eval_running_model(val_dataloader)
        print('Epoch %d, Global Step %d VAL res:\n' % (epoch, global_step), val_result)
        log_wf.write('Global Step %d VAL res:\n' % global_step)
        log_wf.write(str(val_result) + '\n')

        if val_result['eval_loss'] <= best_eval_loss:
            best_eval_loss = val_result['eval_loss']
            val_result['best_eval_loss'] = best_eval_loss
            # save model
            print('[Saving model at]', state_save_path)
            log_wf.write('[Saving model at] %s\n' % state_save_path)
            torch.save(model.module.state_dict(), state_save_path)


        ## save after each epoch
        checkpoint_save_path = os.path.join(args.checkpoint_dir,
                                            '_epoch%d_chkpt.pt' % epoch)
        print('[Saving epoch checkpoint at]', checkpoint_save_path)
        log_wf.write('[Saving epoch checkpoint at] %s\n' % checkpoint_save_path)
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.module.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_save_path)

        print(global_step, tr_loss / nb_tr_steps)
        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
