import numpy as np
import random
import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import copy
from adversary.USE import USEmodel,get_pos,pos_filter
from collections import defaultdict
class Adversary(object):
    """An Adversary tries to fool a model on a given example."""

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25):
        self.synonym_selector = synonym_selector
        self.target_model = target_model
        self.max_perturbed_percent = max_perturbed_percent

    def run(self, model, dataset, device, opts=None):
        """Run adversary on a dataset.
        Args:
        model: a TextClassificationModel.
        dataset: a TextClassificationDataset.
        device: torch device.
        Returns: pair of
        - list of 0-1 adversarial loss of same length as |dataset|
        - list of list of adversarial examples (each is just a text string)
        """
        raise NotImplementedError

    def _softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            _c_matrix = np.max(x, axis=1)
            _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
            _diff = np.exp(x - _c_matrix)
            x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
        else:
            _c = np.max(x)
            _diff = np.exp(x - _c)
            x = _diff / np.sum(_diff)
        assert x.shape == orig_shape
        return x

    def check_diff(self, sentence, perturbed_sentence):
        words = sentence.split()
        perturbed_words = perturbed_sentence.split()
        diff_count = 0
        if len(words) != len(perturbed_words):
            raise RuntimeError("Length changed after attack.")
        for i in range(len(words)):
            if words[i] != perturbed_words[i]:
                diff_count += 1
        return diff_count


class TextfoolerAdversary(Adversary):
    """  Textfooler attack method.  """

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25):
        super(TextfoolerAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        stop_words_textfooler = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 
        'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 
        'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside',
        'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't",
        'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 
        'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter',
        'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't",
        'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 
        'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone',
        'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves',
        'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere',
        'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
        'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon',
        'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
        'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within',
        'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        self.stop_words = set(stop_words_textfooler)
        USE_cache_path = './output/model_file/USE/'
        self.USE = USEmodel(USE_cache_path)
    
    def run(self, sentence, ori_label,sim_score_window=15,import_score_threshold=-1., sim_score_threshold=0.5,synonym_num=50): 
        ori_probs,ori_pre = self.target_model.query([sentence],ori_label)
        ori_probs=self._softmax(ori_probs)
        #assert(ori_pre==ori_label)
        ori_prob=ori_probs[0][ori_pre[0]]
        text_ls = sentence.split()
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        # get the pos and verb tense info
        pos_ls = get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['[UNK]'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        query_texts = [" ".join(leave_1_texts[ii]) for ii in range(len_text)]
        leave_1_probs = []
        leave_1_pre = []
        if len(query_texts) >80:
            leave_1_probs,leave_1_pre = self.target_model.query(query_texts[0:len_text//2],[ori_label]*(len_text//2))
            # leave_1_probs=np.concatenate((leave_1_probs,t1_leave_1_probs),axis=0)
            # leave_1_pre=np.concatenate((leave_1_pre,t1_leave_1_pre),axis=0)
            t2_leave_1_probs,t2_leave_1_pre = self.target_model.query(query_texts[len_text//2:],[ori_label]*(len_text-len_text//2))
            
            leave_1_probs=np.concatenate((leave_1_probs,t2_leave_1_probs),axis=0)
            leave_1_pre=np.concatenate((leave_1_pre,t2_leave_1_pre),axis=0)
            
        else :
            leave_1_probs,leave_1_pre = self.target_model.query(query_texts,[ori_label]*len_text)

        leave_1_probs=self._softmax(leave_1_probs)
        
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = np.argmax(leave_1_probs,axis = 1 )
        
        FY2X1 =[ori_probs[0][leave_1_probs_argmax[i]] for i in range(len_text) ]
        FY2X2 = [leave_1_probs[i][leave_1_probs_argmax[i]] for i in range(len_text)]
     
        addpart = [FY2X1[i]-FY2X2[i] for i in range(len_text)]
        import_scores = (ori_prob - leave_1_probs[:, ori_label] + (leave_1_probs_argmax != ori_label) *addpart )

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in self.stop_words :
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))
        words_perturb2 =[words_perturb[i][1] for i in range(len(words_perturb))]

        synonym_words=self.synonym_selector.find_synonyms_for_tokens(words_perturb2,syn_num=synonym_num)
        synonyms_all = []
        for idx, word in words_perturb:   
            synonyms = synonym_words.pop(0)
            if len(synonyms) !=0 :
                synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0

        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            query_new_texts = [" ".join(new_texts[ii]) for ii in range(len(new_texts))]
            new_probs,_ = self.target_model.query(query_new_texts,None)
            new_probs=self._softmax(new_probs)
            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text

            semantic_sims = \
            self.USE.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                      list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            
            new_probs_mask = (ori_label != np.argmax(new_probs, axis=-1))

            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)

            # prevent incompatible pos
            synonyms_pos_ls = [get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                
                new_label_probs = new_probs[:, ori_label] + (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)
                new_label_prob_min = np.min(new_label_probs, axis=-1)
                new_label_prob_argmin =  np.argmin(new_label_probs, axis=-1)
            
                if new_label_prob_min < ori_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]

        adv_probs,adv_pre = self.target_model.query([' '.join(text_prime)],ori_label)
        adv_probs=self._softmax(adv_probs)
        success = False
        if adv_pre[0]!=ori_label:
            success = True
           
        return success, ' '.join(text_prime), adv_pre[0]

