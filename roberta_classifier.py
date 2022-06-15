"""
Parts based on https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
"""
import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from copy import deepcopy
import random
from model.bert_wrapper import BertWrapper
import argparse
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.data_utils import *
def shuffle_lists(*args):
    """
    See https://stackoverflow.com/a/36695026
    """
    zipped = list(zip(*args))
    random.shuffle(zipped)
    return [list(x) for x in zip(*zipped)]

def save_model(config,epoch):
    save_path = config.save_model_path+str(int(epoch))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "{}/model.pth".format(save_path),
    )

def pad(max_len, seq, token):
    assert isinstance(seq, list)
    abs_len = len(seq)

    if abs_len > max_len:
        seq = seq[:max_len]
    else:
        seq += [token] * (max_len - abs_len)

    return seq

def inference(
    inputs,
    model,
    config,
    bert_wrapper=None,
    val=False,
    usesoftmax =False,
):
    softmax = nn.Softmax(dim=1)
    model.eval()

    assert isinstance(inputs, list)
    inputs = [x.split() for x in inputs]
    

    inputs, masks = [
        list(x) for x in zip(*[bert_wrapper.pre_pro(t) for t in inputs])
    ]
    inputs, masks = torch.tensor(inputs), torch.tensor(masks)
    masks = masks.cuda() 
 
    inputs = inputs.cuda() 

    with torch.no_grad():
        outputs = model(inputs, token_type_ids=None, attention_mask=masks)
        outputs = outputs.logits

    if usesoftmax:
        outputs = softmax(outputs)
    probs = outputs.cpu().detach().numpy().tolist()
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().detach().numpy().tolist()

    if val:
        return preds, outputs
    else:
        return preds, probs

def compute_accuracy(preds, labels):
    assert len(preds) == len(labels)
    return len([True for p, t in zip(preds, labels) if p == t]) / len(preds)

def eval_model(epoch,val_texts,val_pols,config,bert_wrapper=None):

    num_batches = int(math.ceil(len(val_texts) / config.batch_size_val))
    predictions = []
    total_loss = []

    for batch in range(num_batches):
        sentences = val_texts[
            batch * config.batch_size_val : (batch + 1) * config.batch_size_val
        ]
        labels = val_pols[
            batch * config.batch_size_val : (batch + 1) * config.batch_size_val
        ]

        labels = torch.tensor(labels, dtype=torch.int64)
        labels = labels.cuda() 

        preds, outputs = inference(
            sentences,
            model,
            config,
            bert_wrapper=bert_wrapper,
            val=True,
            usesoftmax =True
        )

        predictions += preds
        loss = criterion(outputs, labels)
        total_loss.append(loss.item())

    acc = compute_accuracy(predictions, val_pols)
    total_loss = np.mean(total_loss)

    
    print(
        "Val: epoch {}, loss {}, accuracy {}".format(epoch, total_loss, acc)
    )


def test_model(test_texts,test_pols,config,bert_wrapper=None):
 

    num_batches = int(math.ceil(len(test_texts) / config.batch_size_test))
    predictions = []

    for batch in range(num_batches):
        sentences = test_texts[
            batch * config.batch_size_test : (batch + 1) * config.batch_size_test
        ]
        labels = test_pols[
            batch * config.batch_size_test : (batch + 1) * config.batch_size_test
        ]

        preds, probs = inference(
            deepcopy(sentences),
            model,
            config,
            bert_wrapper=bert_wrapper,
            usesoftmax =True,
        )
        predictions += preds

        for idx in range(len(sentences)):
            sent = sentences[idx]

            print(
                "=============== {} ===============".format(
                    batch * config.batch_size_test + (idx + 1)
                )
            )
            print("Sentence: {}".format(sent))
            print("Label: {}".format(labels[idx]))
            print("Prediction: {}".format(preds[idx]))
            print("Confidence: {}".format(max(probs[idx])))

    print(
        "Test accuracy: {}".format(compute_accuracy(predictions, test_pols))
    )


def run_epoch(epoch, train_texts,train_pols,config,scheduler=None, bert_wrapper=None):
    num_batches = int(math.ceil(len(train_texts) / config.batch_size_train))
    total_loss = []

    model.train()

    train_texts, train_pols = shuffle_lists(
        train_texts, train_pols
    )

    for batch in range(num_batches):
        sentences = train_texts[
            batch * config.batch_size_train : (batch + 1) * config.batch_size_train
        ]
        labels = train_pols[
            batch * config.batch_size_train : (batch + 1) * config.batch_size_train
        ]

        sentences =[sentence.split() for sentence in sentences]
        inputs, masks = [
            list(x)
            for x in zip(
                *[bert_wrapper.pre_pro(sentence) for sentence in sentences]
            )
        ]

        inputs = torch.tensor(inputs, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        inputs, labels = inputs.cuda(), labels.cuda()

        model.zero_grad()


        masks = torch.tensor(masks)
        masks = masks.cuda() 

        outputs = model(
            inputs, token_type_ids=None, attention_mask=masks, labels=labels
        )

        loss = outputs.loss
        loss.backward()

        if config.clip_norm > 0:
            clip_grad_norm_(model.parameters(), config.clip_norm)

        optimizer.step()
        scheduler.step()
        

        total_loss.append(loss.item())

        
    print("Train: epoch {}, loss {}".format(epoch, np.mean(total_loss)))

    save_model(config,epoch)

    
class Config(object):
    def __init__(self, task_name='ag_news'):
        self.task_name =task_name
        self.bert_max_len = 256 if self.task_name == "imdb" else 128
        self.num_classes =-1
        if self.task_name == "imdb":
            self.num_classes = 2
        elif task_name == "ag_news":
            self.num_classes = 4
        elif self.task_name == "yahoo":
            self.num_classes = 10
    
        self.learning_rate = 1e-5
        self.adam_eps =  1e-6
        self.weight_decay = 0.1
        self.batch_size_train =  16
        self.num_epoch = 5
        self.warmup_percent = 0.06
        
        self.clip_norm = 0.0
        self.choose_epoch = 3
        self.save_model_path = "./output/model_file/"+str(self.task_name)+"/roberta/checkpoints/epoch_"
        self.model_base_path = "./output/model_file/"+str(self.task_name)+"/roberta/checkpoints/epoch_"+str(int(self.choose_epoch))+"/model.pth"
        self.pad_token = "<pad>"
        self.batch_size_test = 16
        self.batch_size_val = 16
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--mode",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    parser.add_argument('--vGPU', type=str, default=None, help="Specify which GPUs to use.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.vGPU
    
    bert_wrapper = None
    
    config= Config(task_name=args.task_name)
    if args.mode == "train":
        
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = None, None
       
        bert_wrapper = BertWrapper(config.bert_max_len,config.num_classes)
        model = bert_wrapper.model

        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )


        train_texts , train_pols = read_text("%s/train" % args.task_name, args.data_dir)
        test_texts, test_pols = read_text("%s/test" % args.task_name, args.data_dir)
        print(len(train_texts))
        print(len(test_texts))
        total_steps = config.num_epoch * int(
            math.ceil(len(train_texts) / config.batch_size_train)
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config.warmup_percent),
            num_training_steps=total_steps,
        )
        
        model.cuda()

        print("Start training")

        for epoch in range(1, config.num_epoch + 1):
            run_epoch(epoch,train_texts,train_pols,config, scheduler=scheduler, bert_wrapper=bert_wrapper)
            eval_model(epoch, test_texts, test_pols,config,bert_wrapper=bert_wrapper)
        print("Finished training")
        checkpoint = torch.load(config.model_base_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_model(test_texts, test_pols ,config,bert_wrapper=bert_wrapper)

    elif args.mode == "test":
        bert_wrapper = BertWrapper(config.bert_max_len,config.num_classes)
        model = bert_wrapper.model
        test_texts, test_pols = read_text("%s/test" % args.task_name, args.data_dir)
        checkpoint = torch.load(config.model_base_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.cuda()
        test_model(test_texts, test_pols ,config,bert_wrapper=bert_wrapper)
   
