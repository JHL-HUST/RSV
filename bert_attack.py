# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import re
import pickle

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from utils.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model.bert_model import BertForClassifier, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from model.bert_tokenizer import BertTokenizer
from model.bert_optimizer import BertAdam, warmup_linear

from utils.data_utils import *
from synonym_selector import EmbeddingSynonym,WordNetSynonym
from adversary.attacks import TextfoolerAdversary
from roberta_classifier import Config
from model.bert_wrapper import BertWrapper

class RobertBert(object):
    """The BERT model attacked by adversary."""

    def __init__(self, task_name):
        self.bertconfig= Config(task_name=task_name)
        self.bert_wrapper = BertWrapper(self.bertconfig.bert_max_len,self.bertconfig.num_classes)
        self.model = self.bert_wrapper.model
        checkpoint = torch.load(self.bertconfig.model_base_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.cuda()
        self.model.eval()

    def query(self, sentences, labels,usesoftmax =False):

        softmax = torch.nn.Softmax(dim=1)
        # if single:
        #     assert isinstance(inputs, str)
        #     inputs = [inputs]
        # else:
        assert isinstance(sentences, list)
        sentences = [x.split() for x in sentences]
        
        #inputs = [pad(config.bert_max_len, x, config.pad_token) for x in inputs]

        inputs, masks = [
            list(x) for x in zip(*[self.bert_wrapper.pre_pro(t) for t in sentences])
        ]
        inputs, masks = torch.tensor(inputs), torch.tensor(masks)
        masks = masks.cuda() 
    
        inputs = inputs.cuda() 

        with torch.no_grad():
            outputs = self.model(inputs, token_type_ids=None, attention_mask=masks)
            outputs = outputs.logits

        if usesoftmax:
            outputs = softmax(outputs)
        probs = outputs.cpu().detach().numpy()
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().detach().numpy()

        return  probs,preds

class TargetBert(object):
    """The BERT model attacked by adversary."""

    def __init__(self, args, num_labels, device):
        self.num_labels = num_labels
        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        self.device = device

        # Load a trained model and config that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, "epoch"+str(int(args.num_train_epochs))+"_"+WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)
        model = BertForClassifier(config, num_labels=num_labels)
        model.to(device)
        model.load_state_dict(torch.load(output_model_file))
        self.model = model
        self.model.eval()

    def query(self, sentences, labels,usesoftmax =False):
        examples = []
        for (i, sentence) in enumerate(sentences):
            guid = "%s-%s" % ("dev", i)
            examples.append(
                InputExample(guid=guid, text_a=sentence, text_b=None, label=0, flaw_labels=None))
                #InputExample(guid=guid, text_a=sentence, text_b=None, label=labels[i], flaw_labels=None))
        features = convert_examples_to_features(
            examples, self.max_seq_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(self.device)          

        with torch.no_grad():
            tmp_eval_loss, logits = self.model(input_ids, input_mask, label_ids, segment_ids)

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        if usesoftmax:
            logits =_softmax(logits)
        return logits, predictions

    def evaluate(self, args):
        eval_examples = get_examples_for_bert("%s/test" % args.task_name, args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, self.tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)


        eval_data = TensorDataset(all_input_ids, all_input_mask, \
                                   all_segment_ids, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
 
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)            

            with torch.no_grad():
                tmp_eval_loss, logits = self.model(input_ids, input_mask, label_ids, segment_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

def sample(examples, sample_num):
    """
    Use the Numpy library to randomly select the samples to be attacked. 
    Note that the seed used in our experiments is 0.
    """
    examples = np.array(examples)
    np.random.seed(0)
    shuffled_idx = np.arange(0, len(examples), 1)
    np.random.shuffle(shuffled_idx)
    sampled_idx = shuffled_idx[:sample_num]
    return list(examples[sampled_idx])

def sample_val(examples, sample_num):
    """
    Use the Numpy library to randomly select the samples to be attacked. 
    Note that the seed used in our experiments is 0.
    """
    examples = np.array(examples)
    np.random.seed(0)
    shuffled_idx = np.arange(0, len(examples), 1)
    np.random.shuffle(shuffled_idx)
    sampled_idx = shuffled_idx[sample_num:2*sample_num]
    return list(examples[sampled_idx])



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--attack",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the attack method.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--vGPU', type=str, default=None, help="Specify which GPUs to use.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--save_to_file",
                        default=None,
                        type=str,
                        help="Where do you want to store the generated adversarial examples")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vocab_size",
                        default=50000,
                        type=int)
    parser.add_argument("--max_candidates",
                        default=4,
                        type=int)
    parser.add_argument("--attack_batch",
                        default=1000,
                        type=int,)

    # The following settings are not important
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.vGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.vGPU

    if args.save_to_file:
        adv_pkl_path = args.save_to_file +".pkl"
        save_file = open(args.save_to_file, "a", encoding="utf-8")
        save_file.write(str(vars(args)) + '\n')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()
    num_labels = num_labels_task[task_name]

    vocab, inv_vocab = load_dictionary(task_name, args.vocab_size, data_dir=args.data_dir)
    dist_mat = load_dist_mat(task_name, args.vocab_size, data_dir=args.data_dir)
    for stop_word in stop_words:
        if stop_word in vocab:
            dist_mat[vocab[stop_word], :, :] = 0
    dist_mat = dist_mat[:, :args.max_candidates, :]
    if args.bert_model =="roberta":
        print("load roberta")
        model = RobertBert(args.task_name)
    else :
        print("base bert")
        model = TargetBert(args, num_labels, device)

    eval_accuracy = 0.0
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model.evaluate(args)

    eval_examples = get_examples_for_bert("%s/test" % args.task_name, args.data_dir)
    if args.attack == 'prioritized':
        sample_examples = sample_val(eval_examples, args.attack_batch)
    else :
        sample_examples = sample(eval_examples, args.attack_batch)
    substitution_ratio = []
    unchanged_sample_count = 0
    success_attack_count = 0
    fail_count = 0
    result_info = ""    
    adversarial_examples = []
    
    if args.attack in ['textfooler']:
        if args.attack == 'textfooler':
            synonym_selector = EmbeddingSynonym(args.max_candidates, vocab, inv_vocab, dist_mat, threshold=0.5)
            adversary = TextfoolerAdversary(synonym_selector, model)

        for i in tqdm(range(args.attack_batch), total=args.attack_batch):
            # if i <1604:
            #     continue
            sentence = sample_examples[i].text_a
            adv_sentence = sentence
            label = sample_examples[i].label
            adv_label = int(model.query([sentence], [label])[1][0])
            if adv_label == label:
                success, adv_sentence, adv_label = adversary.run(sentence, label)
                if success:
                    success_attack_count += 1
                else:
                    fail_count += 1
            else:
                unchanged_sample_count += 1
            log_info = (
                str(i)
                + "\noriginal text: "
                + sentence
                + "\noriginal label: "
                + str(label)
                + "\nperturbed text: "
                + adv_sentence
                + "\nperturbed label: "
                + str(adv_label)
                + "\n"
            )
            adversarial_examples.append({"clean_text": sentence,"perturbed_text": adv_sentence,"clean_label": label,"perturbed_lable": adv_label,})
            if args.save_to_file:
                save_file.write(log_info)     
    else:
        raise NotImplementedError
    with open(adv_pkl_path, "wb") as handle:
        pickle.dump(adversarial_examples, handle)
    model_acc_before_attack = 1.0 - unchanged_sample_count / args.attack_batch
    model_acc_after_attack = (
        1.0 - (unchanged_sample_count + success_attack_count) / args.attack_batch
    )

    if len(substitution_ratio) == 0:
        average_sub_ratio = 0.0
    else:
        average_sub_ratio = sum(substitution_ratio) / len(substitution_ratio)
    summary_table_rows = [
        ["ITEM", "VALUE"],
        # ["Total Time For Attack:", end_attack_time - start_attack_time],
        ["Model Accuracy of Test Set:", eval_accuracy],
        ["Model Accuracy Before Attack:", model_acc_before_attack,],
        [
            "Attack Success Rate:",
            success_attack_count / (args.attack_batch - unchanged_sample_count),
        ],
        ["Model Accuracy After Attack:", model_acc_after_attack,],
        ["Average Substitution Ratio:", average_sub_ratio,],
    ]
    for row in summary_table_rows:
        result_info += str(row[0]) + str(row[1]) + "\n"
    logger.info(result_info)
    if args.save_to_file:
        save_file.write(result_info)
        save_file.close()
    
if __name__ == "__main__":
    main()