# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import argparse
import glob
import logging
import os
import random
import timeit
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample

from torch.utils.tensorboard import SummaryWriter

from utils import logging_config

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
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

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        with tqdm(total=len(train_dataloader)) as bar:
            for step, batch in enumerate(train_dataloader):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                    del inputs["token_type_ids"]

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                    if args.version_2_with_negative:
                        inputs.update({"is_impossible": batch[7]})
                    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                        )

                outputs = model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if args.evaluate_during_training:
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        bar.update(min(args.logging_steps, step))
                        logger.info("lr: %f, global_step: %d", scheduler.get_lr()[0], global_step)
                        logger.info("loss: %f, global_step: %d", (tr_loss - logging_loss) / args.logging_steps,
                                    global_step)
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def test(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, "test", output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.test_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running testing {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.test_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    testTime = timeit.default_timer() - start_time
    logger.info("  Testing done in total %f secs (%f sec per example)", testTime, testTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    return predictions


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, "eval", output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def remove_unnecessary_columns(df):
    print("-----------------------------------------------------------------------------------")
    print("----- Remove all unnecessary columns and only keep passage, answer, and query -----")
    print("-----------------------------------------------------------------------------------")

    to_remove = list(df.columns)
    to_remove.remove("passage")
    to_remove.remove("query")
    to_remove.remove("answer")
    to_remove.remove("qid")

    for t in to_remove:
        del df[t]

    return df


def remove_urls(df):
    my_url_regex = r"(\(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]" \
                   r"[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\." \
                   r"[a-zA-Z0-9]+\.[^\s]{2,})"

    print("----------------------------------------------------------------------------------")
    print("----- Will remove all urls in passage and answer and replace them with (URL) -----")
    print("----------------------------------------------------------------------------------")

    for index, row in df.iterrows():
        df.at[index, "passage"] = re.sub(my_url_regex, "(URL)", row["passage"])
        df.at[index, "answer"] = re.sub(my_url_regex, "(URL)", row["answer"])

    return df


def clean_text(df):
    print("-----------------------------------------------------------------------------------")
    print("----- Clean the passage, answer, and question text. Remove newlines and such. -----")
    print("-----------------------------------------------------------------------------------")

    for index, row in df.iterrows():
        df.at[index, "passage"] = " ".join(str(row["passage"]).split()).replace("<", "").replace(">", "")
        df.at[index, "answer"] = " ".join(str(row["answer"]).split()).replace("<", "").replace(">", "")
        df.at[index, "query"] = " ".join(str(row["query"]).split()).replace("<", "").replace(">", "")

    return df


def split_passages(text, max_passage_length, stride):
    start = 0
    end = len(text.split())
    passages = []
    s = 0
    e = 0
    while s < end and e < end:
        if end - s + 1 >= max_passage_length:
            passages.append(' '.join(text.split()[s:s + max_passage_length]))
            s += stride
        else:
            passages.append(' '.join(text.split()[s:end]))
            e = end
    return passages


def get_passages(passage, answer, max_passage_length, stride):
    passages = []
    if len(passage.split()) <= max_passage_length:
        return [passage]
    try:
        answer_start_index = passage.index(answer)
    except:
        return [' '.join(passage.split()[i:i + max_passage_length]) for i in range(0, len(passage.split()), max_passage_length)]
    if max_passage_length > len(answer.split()):
        max_allowed_left = random.randint(0, max_passage_length - len(answer.split()))
    else:
        max_allowed_left = 0
    left_passage = ' '.join(passage[:answer_start_index].strip().split()[-max_allowed_left:])
    before_left_passage = ' '.join(passage[:answer_start_index].strip().split()[:-max_allowed_left])
    try:

        temp_start_idx = left_passage.index(".") + 1
        before_left_passage += ' ' + left_passage[:temp_start_idx]
    except ValueError:
        temp_start_idx = 0
        before_left_passage = ''

    left_passage = left_passage[temp_start_idx:].strip()
    left_passage += ' ' + answer
    if max_passage_length > len(left_passage.split()):
        max_allowed_right = random.randint(0, max_passage_length - len(left_passage.split()))
    else:
        max_allowed_right = 0
    right_passage_tmp = passage[answer_start_index + len(answer):].strip()
    right_passage = ' '.join(right_passage_tmp.split()[:max_allowed_right])
    right_passage_after = ' '.join(right_passage_tmp.split()[max_allowed_right:])
    try:
        temp_end_idx = len(right_passage) - right_passage[::-1].index(".")
        right_passage_after = right_passage[temp_end_idx:] + ' ' + right_passage_after
    except ValueError:
        temp_end_idx = 0
        right_passage_after = right_passage + ' ' + right_passage_after
    right_passage = right_passage[:temp_end_idx]
    final_passage = left_passage.strip() + ' ' + right_passage.strip()
    passage_large = before_left_passage + ' ' + right_passage_after.strip()
    if final_passage.strip() != '':
        passages.append(final_passage.strip())
    if passage_large.strip() != '':
        passage_splits = split_passages(passage_large.strip(), max_passage_length, stride)
        passages += passage_splits
    return passages


def create_examples(df_file: str, is_test: bool, max_passage_length: int = None, add_neg: bool = None, stride: int = None):
    df = pd.read_csv(df_file,sep='\t')
    examples = []
    if is_test:
        for index, row in tqdm(df.iterrows()):
            qas_id = row['qid']
            question_text = row['query']
            passage = row['passage']
            start_index = None
            is_impossible = False
            example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=passage,
                    answer_text=None,
                    start_position_character=start_index,
                    title=None,
                    is_impossible=is_impossible,
                    answers=[]
            )
            examples.append(example)
        logger.info('The total passage-question pairs=%d',len(examples))
        return examples
    else:
        count = 0
        for index, row in tqdm(df.iterrows()):
            qas_id = row['qid']
            question_text = row['query']
            passage = row['passage']
            answer = row['answer']
            passages = get_passages(passage, answer, max_passage_length, stride)
            if len(passages) > 0:
                for counter, i in enumerate(passages):
                    try:
                        start_index = i.index(answer)
                        is_impossible = False
                        if counter > 0:
                            continue
                        count += 1
                    except:
                        start_index = -1
                        answer=''
                        is_impossible = True
                    if add_neg:
                        example = SquadExample(
                            qas_id=((qas_id*len(df)) + counter) if counter > 0 else qas_id,
                            question_text=question_text,
                            context_text=i,
                            answer_text=answer,
                            start_position_character=start_index,
                            title=None,
                            is_impossible=is_impossible,
                            answers=[{"text":answer}] if not is_impossible else []
                        )
                        examples.append(example)
                        continue
                    else:
                        if not is_impossible:
                            example = SquadExample(
                                qas_id=((qas_id*len(df)) + counter) if counter > 0 else qas_id,
                                question_text=question_text,
                                context_text=i,
                                answer_text=answer,
                                start_position_character=start_index,
                                title=None,
                                is_impossible=is_impossible,
                                answers=[{"text":answer}]
                            )
                            examples.append(example)
                            continue

        logger.info('Total passage-question pairs=%d',len(df))
        logger.info('Total passage-question pairs after processing with positives=%d',count)
        logger.info('Total passage-question pairs after processing with negatives=%d',len(examples)-count)
        print('Total passage-question pairs=',len(df))
        print('Total passage-question pairs after processing with positives=',count)
        print('Total passage-question pairs after processing with negatives=',len(examples)-count)
        return examples


def create_train_test_split(path_to_json:str, max_answer_words:int = 100, train_test_split_ratio:float = 0.8):
    orig_df = pd.read_json(path_to_json, orient='columns', typ='frame')
    cleaner_df = remove_unnecessary_columns(orig_df)  # Remove unnecessary columns from dataframe
    clean_text_df = clean_text(cleaner_df)  # String Cleaning
    no_url_df = remove_urls(clean_text_df)  # URL Removal
    no_url_df = no_url_df.groupby(['qid', 'query', 'answer']).apply(lambda x: ' '.join(x['passage'])).to_frame(
        name='passage').reset_index()
    filtered = no_url_df.answer.apply(lambda x: True if len(x.split()) <= max_answer_words else False)
    shorted_ans_df = no_url_df[filtered].reset_index(drop=True)
    msk = np.random.rand(len(shorted_ans_df)) < train_test_split_ratio
    train = shorted_ans_df[msk].reset_index(drop=True)
    test = shorted_ans_df[~msk].reset_index(drop=True)
    path = '/'.join(path_to_json.split("/")[:-1])
    train.to_csv(os.path.join(path,'train_answerextraction_{}.tsv'.format(max_answer_words)), sep='\t', index=False)
    test.to_csv(os.path.join(path,'test_answerextraction_{}.tsv'.format(max_answer_words)), sep='\t', index=False)


def load_and_cache_examples(args, tokenizer, type, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else args.squad_data_dir if args.squad_data_dir else "."
    name = ''
    if args.data_dir:
        if type == "train":
            name += '-'.join(args.train_file.split(".")[:-1]) if args.train_file else ''
        elif type == "eval":
            name +=  '_'.join(args.predict_file.split(".")[:-1]) if args.predict_file else ''
        elif type == "test":
            name += '_'.join(args.test_file.split(".")[:-1]) if args.test_file else ''
    if args.squad_data_dir:
        if type == "train":
            name += '_'.join(args.squad_train_file.split(".")[:-1]) if args.squad_train_file else ''
        elif type=="eval":
            name += '_'.join(args.squad_predict_file.split(".")[:-1]) if args.squad_predict_file else ''

    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}.cache".format(type,name)
    )
    print(cached_features_file)
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        examples=[]
        if type == "eval":
            if args.data_dir and args.predict_file:
                predict_file = args.predict_file
                if '.json' in args.predict_file:
                    ## assume that test file is created from json file during training
                    predict_file = 'test_answerextraction_{}.tsv'.format(args.max_answer_words)
                examples += create_examples(os.path.join(args.data_dir,predict_file), False, max_passage_length=args.max_passage_length, add_neg=args.add_neg, stride=args.stride)
            if args.squad_data_dir and args.squad_predict_file:
                processor = SquadV2Processor()
                examples += processor.get_dev_examples(args.squad_data_dir, filename=args.squad_predict_file)
        elif type == "train":
            if args.data_dir and args.train_file:
                train_file = args.train_file
                if '.json' in args.train_file:
                    create_train_test_split(os.path.join(args.data_dir,args.train_file), max_answer_words=args.max_answer_words, train_test_split_ratio=args.train_test_split_ratio)
                    train_file = 'train_answerextraction_{}.tsv'.format(args.max_answer_words)
                examples += create_examples(os.path.join(args.data_dir,train_file), False, max_passage_length=args.max_passage_length, add_neg=args.add_neg, stride=args.stride)
            if args.squad_data_dir and args.squad_train_file:
                processor = SquadV2Processor()
                examples += processor.get_train_examples(args.squad_data_dir, filename=args.squad_train_file)
        elif type == "test":
            examples = create_examples(os.path.join(args.data_dir, args.test_file), True)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True if type=="train" else False,
            return_dataset="pt",
            threads=args.threads,
        )



        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gpu", type=str, default="6,7", help="gpus to run model on")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="The input test file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    # Other parameters
    parser.add_argument(
        "--squad_data_dir",
        default=None,
        type=str,
        help="The input data dir of squad. Should contain the .json files for the task.",
    )
    parser.add_argument(
        "--squad_train_file",
        default=None,
        type=str,
        help="The input training file for squad. If a data dir is specified, will look for the file there",
    )
    parser.add_argument(
        "--squad_predict_file",
        default=None,
        type=str,
        help="The input evaluation file for squad. If a data dir is specified, will look for the file there",
    )


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
        "--version_2_with_negative",
        action="store_true",
        help="If true, the train examples contain some that do not have an answer.",
    )

    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--train_test_split_ratio",
        type=float,
        default=0.8,
        help="If given data_all file as input use this ratio to split train and test for amazon data",
    )


    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )

    parser.add_argument(
        "--max_answer_words",
        default=100,
        type=int,
        help="The maximum number of words allowable in a passage answer during training. (Used for filtering long answers in amazon data)",
    )

    parser.add_argument(
        "--max_passage_length",
        default=350,
        type=int,
        help="The maximum number of words allowable in a passage for training. (Used for windowing passages for amazon data)",
    )

    parser.add_argument(
        "--stride",
        default=50,
        type=int,
        help="if add_neg is true. we stride to split the long passages and create negatives. (Used for striding on amazon data)",
    )

    parser.add_argument(
        "--add_neg",
        action="store_true",
        help="If true, the larger passage will be split and added as negatives for amazon data",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--per_gpu_test_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=100,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )


    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging_config(args.output_dir, name='train')

    # Set seed
    set_seed(args)



    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, "train", output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save the trained model and the tokenizer
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_train and args.do_eval:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    if not args.do_train and args.do_eval:
        # Evaluate
        result = evaluate(args, model, tokenizer, prefix="")
        results.update(result)

    if args.do_test:
        logger.info("Testing the model")
        # Evaluate
        test(args, model, tokenizer, prefix="")

    logger.info("Results: {}".format(results))
    return results


if __name__ == "__main__":
    main()
