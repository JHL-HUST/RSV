import math
import random
import numpy as np
import pickle
from synonym_selector import EmbeddingSynonym,WordNetSynonym
import argparse
from model.cnn_model import CNNModel
from utils.data_utils import *

import torch
from roberta_classifier import Config
from model.bert_wrapper import BertWrapper

from utils.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model.bert_model import BertForClassifier, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from model.bert_tokenizer import BertTokenizer
from model.bert_optimizer import BertAdam, warmup_linear


class EVAL_RDSU:
    def __init__(self,args):
        self.args = args
        self.model = self.load_model(args.modeltype)

    def load_model(self,modeltype):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        vocab, _ = load_dictionary(self.args.task_name, self.args.vocab_size, data_dir=self.args.data_dir)
        num_labels = num_labels_task[self.args.task_name]
        if modeltype == 'bert':
            model = TargetBert(args, num_labels, device)
        elif modeltype == 'roberta':
            model = RobertBert(args.task_name)
        elif modeltype == 'cnn':
            model = CNNModel(num_labels, vocab, self.args.max_seq_length, device)
            output_model_file = os.path.join(self.args.output_dir, "epoch"+str(int(self.args.num_train_epochs)-1))
            model.to(device)
            model.load_state_dict(torch.load(output_model_file))
            model.eval()
        return model
    
    def eval_all_examples(self,eval_file,overall = False):
        with open(eval_file, "rb") as handle:
            example_list = pickle.load(handle)

        votenum = len(example_list[0]["clean_transfer_list"])

        # load data 
        clean_text_list = []
        perturbed_text_list = []
        clean_label_list = []
        perturbed_lable_list = []
        clean_transfer_list_total = []
        perturbed_transfer_list_total = []
        for exp in example_list:
            clean_text_list.append(exp["clean_text"])
            perturbed_text_list.append(exp["perturbed_text"])
            clean_transfer_list_total += exp["clean_transfer_list"]
            perturbed_transfer_list_total += exp["perturbed_transfer_list"]
            clean_label_list.append(exp["clean_label"])
            perturbed_lable_list.append(exp["perturbed_lable"])
        
        # load data 
        split_batchnum = 100
        each_batctnum = len(clean_text_list)/split_batchnum

        
        ori_pre = []  # prediction on ori result
        adv_pre = []  # prediction on adv result

        # query 
        for batchnum in range(split_batchnum):
            _,temp_ori_pre = self.model.query(clean_text_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],0,usesoftmax =True)
            _,temp_adv_pre = self.model.query(perturbed_text_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],0,usesoftmax =True)
            ori_pre += temp_ori_pre.tolist()
            adv_pre += temp_adv_pre.tolist()

        adv_suc = 0
        ori_suc = 0

        right_after_attack = 0
        for i in range(len(example_list)):
            if ori_pre[i]==clean_label_list[i]:
                ori_suc +=1
                if adv_pre[i]!=clean_label_list[i]:
                    adv_suc += 1
                else :
                    right_after_attack += 1
           
        ori_acc = ori_suc/len(example_list)
        adv_acc = right_after_attack/len(example_list)
        pos = adv_suc
        print("acc on clean {}".format(ori_acc))
        print("acc on adv   {}".format(adv_acc))
        
     

        mul_transfer_ori_pre = []
        mul_transfer_adv_pre = []
        mul_transfer_ori_prob = []
        mul_transfer_adv_prob = []
        split_batchnum = 1000
        each_batctnum = len(clean_transfer_list_total)/split_batchnum
        
        for batchnum in range(split_batchnum):
            temp_mul_transfer_ori_prob,temp_mul_transfer_ori_pre=self.model.query(clean_transfer_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],0,usesoftmax =True)
            temp_mul_transfer_adv_prob,temp_mul_transfer_adv_pre=self.model.query(perturbed_transfer_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],0,usesoftmax =True)
            mul_transfer_ori_pre += temp_mul_transfer_ori_pre.tolist()
            mul_transfer_ori_prob += temp_mul_transfer_ori_prob.tolist()
            mul_transfer_adv_pre += temp_mul_transfer_adv_pre.tolist()
            mul_transfer_adv_prob += temp_mul_transfer_adv_prob.tolist()

        transfer_ori_pre_list = []
        transfer_adv_pre_list = []
        for i in range(len(example_list)):
            transfer_ori_prob=np.sum(mul_transfer_ori_prob[i*votenum:i*votenum+votenum],axis=0)
            transfer_ori_prob =transfer_ori_prob/votenum
            transfer_ori_pre = np.argmax(transfer_ori_prob)
            transfer_ori_pre_list.append(transfer_ori_pre)

            transfer_adv_prob=np.sum(mul_transfer_adv_prob[i*votenum:i*votenum+votenum],axis=0)
            transfer_adv_prob =transfer_adv_prob/votenum
            transfer_adv_pre = np.argmax(transfer_adv_prob)
            transfer_adv_pre_list.append(transfer_adv_pre)

        
        ori_right_trans_right = 0
        ori_right_trans_wrong = 0
        ori_wrong_trans_right = 0 
        ori_wrong_trans_wrong = 0 

        t_p = 0
        f_p = 0
        f_n = 0
        res_ori = 0
        for i in range(len(example_list)):
            # ori_pre = model(clean text)  clean_label_list ori_lable
            if ori_pre[i]==clean_label_list[i] and adv_pre[i]!=clean_label_list[i] and adv_pre[i]!= transfer_adv_pre_list[i]:
                t_p+=1
            if ori_pre[i]==clean_label_list[i] and adv_pre[i]==clean_label_list[i] and adv_pre[i]!= transfer_adv_pre_list[i]:
                f_p+=1
            if ori_pre[i]==clean_label_list[i] and adv_pre[i]!=clean_label_list[i] and adv_pre[i]== transfer_adv_pre_list[i]:
                f_n+=1
            if transfer_ori_pre_list[i]==clean_label_list[i] and ori_pre[i]==clean_label_list[i]:
                ori_right_trans_right +=1
            elif transfer_ori_pre_list[i]!=clean_label_list[i] and ori_pre[i]==clean_label_list[i]:
                ori_right_trans_wrong +=1
                temp_ori_pro,_ = self.model.query([clean_text_list[i]],0,usesoftmax =True)
            elif transfer_ori_pre_list[i]==clean_label_list[i] and ori_pre[i]!=clean_label_list[i]:
                ori_wrong_trans_right +=1
                temp_ori_pro,_ = self.model.query([clean_text_list[i]],0,usesoftmax =True)
            else:
                ori_wrong_trans_wrong +=1

            if transfer_ori_pre_list[i]==clean_label_list[i]:
                res_ori += 1

        print(res_ori)
        assert(f_n+t_p==pos)       

        f1 =  (2 * t_p) / (2 * t_p + f_p + f_n) if 2 * t_p + f_p + f_n > 0 else 0
        tpr=t_p / pos if pos > 0 else 0
        
        transfer_ori_acc = (ori_right_trans_right+ori_wrong_trans_right)/len(example_list)
        print("(RDSU) transfer acc on clean : {}".format(transfer_ori_acc))
        # print("(RDSU)  ori_right_trans_right : {}".format(ori_right_trans_right/len(example_list)))
        # print("(RDSU)  ori_right_trans_wrong : {}".format(ori_right_trans_wrong/len(example_list)))
        # print("(RDSU)  ori_wrong_trans_right : {}".format(ori_wrong_trans_right/len(example_list)))
        # print("(RDSU)  ori_wrong_trans_wrong : {}".format(ori_wrong_trans_wrong/len(example_list)))


        ori_wrong = 0  # ori predict fail  
        adv_to_ori = 0  #ori  attack fail num 
        adv_to_adv = 0  
        ori_to_adv = 0 
        ori_to_ori = 0
        res = 0
        for i in range(len(example_list)):
            if clean_label_list[i]!=ori_pre[i]:
                ori_wrong += 1
            elif adv_pre[i]==clean_label_list[i] and transfer_adv_pre_list[i]!=clean_label_list[i]:
                ori_to_adv +=1
            elif adv_pre[i]!=clean_label_list[i] and transfer_adv_pre_list[i]!=clean_label_list[i]:
                adv_to_adv +=1
            elif adv_pre[i]!=clean_label_list[i] and transfer_adv_pre_list[i]==clean_label_list[i]:
                adv_to_ori +=1
            elif adv_pre[i]==clean_label_list[i] and transfer_adv_pre_list[i]==clean_label_list[i]:
                ori_to_ori +=1

            if transfer_adv_pre_list[i]==clean_label_list[i]:
                res += 1

        transfer_adv_acc = (ori_to_ori+adv_to_ori)/len(example_list)
        # print("(RDSU)  transfer  acc on adv {}".format((ori_to_ori+adv_to_ori)/len(example_list)))
        # print("(RDSU)  ori_wrong : {}".format(ori_wrong/len(example_list)))
        # print("(RDSU)  adv_to_ori : {}".format(adv_to_ori/len(example_list)))
        # print("(RDSU)  adv_to_adv : {}".format(adv_to_adv/len(example_list)))
        # print("(RDSU)  ori_to_adv : {}".format(ori_to_adv/len(example_list)))
        # print("(RDSU)  ori_to_ori : {}".format(ori_to_ori/len(example_list)))
        print("(RDSU) restore acc on adv : {}".format(res/len(example_list)))
        print("(RDSU) f1 score : {}".format(f1))
        return transfer_ori_acc,transfer_adv_acc,t_p,f_p,f_n,f1,tpr


class TargetBert(object):
    """The BERT model attacked by adversary."""

    def __init__(self, args, num_labels, device):
        self.num_labels = num_labels
        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.device = device

        # Load a trained model and config that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, "epoch"+str(int(args.num_train_epochs)-1)+"_"+WEIGHTS_NAME)
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
       
        assert isinstance(sentences, list)
        sentences = [x.split() for x in sentences]
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
        probs = outputs.cpu().detach().numpy().tolist()
        _, preds = torch.max(outputs, 1)
        #preds = preds.cpu().detach().numpy().tolist()
        return  outputs,preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default='ag_news',
                        type=str,
                        help="The name of the task to eval.")
    parser.add_argument("--vocab_size",
                        default=50000,
                        type=int)
    parser.add_argument("--data_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument('--vGPU', type=str, default=None, help="Specify which GPUs to use.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--output_dir",
                        default="./output/model_file/ag_news/cnn",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
   
    parser.add_argument("--modeltype",
                        default="cnn",
                        type=str,
                        help="the model type")
    parser.add_argument("--eval_file",
                        default=" ",
                        type=str,
                        help="eval file path")
    args = parser.parse_args()
    if args.vGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.vGPU

    eval_main = EVAL_RDSU(args=args)
       
    transfer_ori_acc,transfer_adv_acc,t_p,f_p,f_n,f1,tpr =eval_main.eval_all_examples(args.eval_file)
           