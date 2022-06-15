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

from model.cnn_model import CNNModel
from utils.data_utils import *
from adversary.attacks import TextfoolerAdversary
from synonym_selector import EmbeddingSynonym

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
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

    # Other parameters
    parser.add_argument("--save_to_file",
                        default=None,
                        type=str,
                        help="Where do you want to store the generated adversarial examples")
    parser.add_argument("--attack_batch",
                        default=1000,
                        type=int,)
    parser.add_argument("--max_seq_length",
                        default=500,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--vocab_size",
                        default=50000,
                        type=int)
    parser.add_argument("--max_candidates",
                        default=4,
                        type=int)
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
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
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
    parser.add_argument('--vGPU', 
                        type=str, 
                        default=None, 
                        help="Specify which GPUs to use.")
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
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: Falses".format(
        device, n_gpu, bool(args.local_rank != -1)))

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
    
    model = CNNModel(num_labels, vocab, args.max_seq_length, device)
    output_model_file = os.path.join(args.output_dir, "epoch"+str(int(args.num_train_epochs)-1))
    model.to(device)
    model.load_state_dict(torch.load(output_model_file))
    
    eval_accuracy = 0.0
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_examples, test_labels = read_text("%s/test" % task_name, args.data_dir)
        test_seqs, test_seqs_mask = text_encoder(test_examples, vocab, args.max_seq_length)
        test_data = TensorDataset(torch.tensor(test_seqs, dtype=torch.long), \
                                    torch.tensor(test_seqs_mask, dtype=torch.long), \
                                    torch.tensor(test_labels, dtype=torch.long)) 
        logger.info("***** Running evaluation on dev set*****")
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predictions = []

        loss_fn = torch.nn.CrossEntropyLoss()

        for input_ids, input_mask, labels in tqdm(test_dataloader, desc="Evaluating"):
                
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)            

            with torch.no_grad():
                logits = model.input_to_logit(input_ids)
                tmp_eval_loss = loss_fn(logits, labels)

            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, labels)

            predictions.extend(list(np.argmax(logits, axis=1)))

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

    test_examples, test_labels = read_text("%s/test" % task_name, args.data_dir)
    sample_clean_texts, sample_labels = sample(test_examples, test_labels, args.attack_batch)
    sample_seqs, sample_seqs_mask = text_encoder(sample_clean_texts, vocab, args.max_seq_length)

    model.eval()
    substitution_ratio = []
    unchanged_sample_count = 0
    success_attack_count = 0
    fail_count = 0
    result_info = ""
    adversarial_examples = []
    
    # more attack method can be added here, we give textfooler as an example
    if args.attack in ['textfooler']:  
        if args.attack == 'textfooler':
            synonym_selector = EmbeddingSynonym(args.max_candidates, vocab, inv_vocab, dist_mat, threshold=0.5)
            adversary = TextfoolerAdversary(synonym_selector, model)
        
        for i in tqdm(range(args.attack_batch), total=args.attack_batch):
            sentence = sample_clean_texts[i]
            adv_sentence = sentence
            label = sample_labels[i]
    
            adv_label = int(model.query([sentence], [label])[1][0])
            if adv_label == label:
                success, adv_sentence, adv_label = adversary.run(sentence, label)
                if success:
                    success_attack_count += 1
                else:
                    fail_count += 1
            else:
                unchanged_sample_count += 1
            adversarial_examples.append({"clean_text": sentence,"perturbed_text": adv_sentence,"clean_label": label,"perturbed_lable": adv_label,})

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
