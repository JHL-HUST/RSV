import math
import random
import numpy as np
from utils.data_utils import load_dictionary,load_dist_mat
import pickle
from synonym_selector import EmbeddingSynonym,WordNetSynonym
import argparse
class Detector_RDSU:
    def __init__(self, args,):
        self.randomrate = args.randomrate
        self.votenum = args.votenum
        self.fix_thr = args.fixrate
        self.n_neighbors = 6
        self.threshold = 0.5
        self.vocab, self.inv_vocab = load_dictionary(args.task_name, args.vocab_size, data_dir=args.data_dir)
        self.dist_mat = load_dist_mat(args.task_name, args.vocab_size, data_dir=args.data_dir)

        self.emb_synonym_selector = EmbeddingSynonym(self.n_neighbors, self.vocab, self.inv_vocab, self.dist_mat, threshold=self.threshold)
        self.wordnet_synonym_selector = WordNetSynonym(self.vocab, self.inv_vocab)
        with open(args.advfile, "rb") as handle:
            self.adv_examples = pickle.load(handle)
        self.neighborslist = {}

    def transfer(self, text):
        input_seq = text.split()
        masknum = int((len(input_seq)*self.randomrate)//1)
        N = range(len(input_seq))
        replace_idx = random.sample(N,masknum)
        replaced_idx = []
        replacenum = 0
        for idx in replace_idx:
            word = input_seq[idx]
            if not (word  in self.vocab) :
                continue
            if  self.vocab[word] <self.fix_thr*len(self.vocab) :
                continue
            if word in self.neighborslist:
                neighbors = self.neighborslist[word]
            else:
                neighbors = list(set(self.wordnet_synonym_selector.get_word_net_synonyms(word) + self.emb_synonym_selector.find_synonyms(word) ))
            filterneighbors =[]
            for w in neighbors:
                if w in self.vocab and self.vocab[w]<20000:
                    filterneighbors.append(w)
            neighbors =  filterneighbors       
            if len(neighbors) > 0:
                rep = random.choice(neighbors)
                input_seq[idx] = rep
                replaced_idx.append((word, rep, idx))
                replacenum += 1
        return " ".join(input_seq)


    def transfer_all_examples(self,save_path):
        transfer_examples = []
        for adv in self.adv_examples:
            clean_transfer_list = []
            perturbed_transfer_list = []
            clean_text = adv["clean_text"]
            perturbed_text = adv["perturbed_text"]
            for i in range (self.votenum):
                clean_transfer_list.append(self.transfer(clean_text))
                perturbed_transfer_list.append(self.transfer(perturbed_text))
            transfer_examples.append(
                {
                    "clean_text": clean_text,
                    "perturbed_text": perturbed_text,
                    "clean_transfer_list": clean_transfer_list,
                    "perturbed_transfer_list": perturbed_transfer_list,
                    "clean_label": adv["clean_label"],
                    "perturbed_lable": adv["perturbed_lable"],
                }

            )
        with open(save_path, "wb") as handle:
            pickle.dump(transfer_examples, handle)
        return transfer_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default="ag_news",
                        type=str,
                        help="task name, include 'imdb', 'ag_news' and 'yahoo'")
    parser.add_argument("--data_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_size",
                        default=50000,
                        type=int)
    parser.add_argument("--adv_file",
                        default="./output/adv_example/ag_news_cnn_textfooler.pkl",
                        type=str,
                        help="adversarial examples path")
    parser.add_argument("--votenum",
                        default=25,
                        type=int,
                        help="vote num for RS&V")
    parser.add_argument("--randomrate",
                        default=0.6,
                        type=float,
                        help="random rate for RS&V")
    parser.add_argument("--fixrate",
                        default=0.02,
                        type=float,
                        help="fix rate for RS&V")
    parser.add_argument("--advfile",
                        default="./output/adv_example/ag_news_cnn_textfooler.pkl",
                        type=str,
                        help="output file path")
    parser.add_argument("--out_file",
                        default="./output/transfer/transfer_ag_news_cnn_textfooler.pkl",
                        type=str,
                        help="output file path")
    args = parser.parse_args()

 
    RDSU = Detector_RDSU(args=args)
    transfer_examples = RDSU.transfer_all_examples(args.out_file)
    print(transfer_examples[0])
            
