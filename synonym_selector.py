import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.tag import StanfordPOSTagger
from functools import partial

class SynonymSelector(object):
    """An class tries to find synonyms for a given word."""

    def __init__(self, vocab, inv_vocab):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.stop_words = ['the', 'a', 'an', 'to', 'of', 'and', 'with', 'as', 'at', 'by', 'is', 'was', 'are', 'were', 'be', 'he', 'she', 'they', 'their', 'this', 'that']

    def find_synonyms(self, word):
        """Return the `num` nearest synonyms of word."""
        raise NotImplementedError


class EmbeddingSynonym(SynonymSelector):
    """Selecting syonyms by GLove word embeddings distance."""

    def __init__(self, max_candidates, vocab, inv_vocab, synonym_matrix, threshold=None):
        super(EmbeddingSynonym, self).__init__(vocab, inv_vocab)
        self.max_candidates = max_candidates
        self.synonym_matrix = synonym_matrix
        self.threshold = threshold

    def find_synonyms(self, word, syn_num=None):
        if word in self.stop_words or word not in self.vocab:
            return []
        word_id = self.vocab[word]
        dist_order = self.synonym_matrix[word_id, :, 0]
        dist_list = self.synonym_matrix[word_id, :, 1]
        if syn_num:
            n_return = np.min([np.sum(dist_order > 0), syn_num])
        else:
            n_return = np.min([np.sum(dist_order > 0), self.max_candidates])
        dist_order, dist_list = dist_order[:n_return], dist_list[:n_return]
        if self.threshold is not None:
            mask_thres = np.where(dist_list < self.threshold)
            dist_order, dist_list = dist_order[mask_thres], dist_list[mask_thres]
        synonyms = []
        for word_id in dist_order:
            synonyms.append(self.inv_vocab[word_id])
        return synonyms

    def find_synonyms_for_tokens(self, words,syn_num=None):
        synsets = []
        for w in words:
            synsets.append(self.find_synonyms(w,syn_num=syn_num))
        return synsets

    def find_synonyms_id(self, word_id, syn_num=None):
        if word_id <= 0 or word_id > 50000:
            return []
        dist_order = self.synonym_matrix[word_id, :, 0]
        dist_list = self.synonym_matrix[word_id, :, 1]
        if syn_num:
            n_return = np.min([np.sum(dist_order > 0), syn_num])
        else:
            n_return = np.min([np.sum(dist_order > 0), self.max_candidates])
        dist_order, dist_list = dist_order[:n_return], dist_list[:n_return]
        if self.threshold is not None:
            mask_thres = np.where(dist_list < self.threshold)
            dist_order, dist_list = dist_order[mask_thres], dist_list[mask_thres]
        return dist_order



class WordNetSynonym(SynonymSelector):
    """Selecting syonyms by GLove word embeddings distance."""

    def __init__(self, vocab, inv_vocab):
        super(WordNetSynonym, self).__init__(vocab, inv_vocab)
        self.supported_nltk_pos = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR','JJS','RB','RBR','RBS']
        jar = './data/aux_files/stanford-postagger-2018-10-16/stanford-postagger.jar'
        model = './data/aux_files/stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
        self.pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')


    def _nltk_pos_to_wordnet(self, nltk_pos):
        if nltk_pos[:2] == 'NN':
            wn_pos = 'n'
        elif nltk_pos[:2] == 'VB':
            wn_pos = 'v'
        elif nltk_pos[:2] == 'JJ':
            wn_pos = 'a'
        else:
            wn_pos = 'r'
        return wn_pos
        

    def find_synonyms(self, word, nltk_pos):
        if (word in self.stop_words) or (
            word not in self.vocab) or (
            nltk_pos not in self.supported_nltk_pos):
            return []
        candidates = []
        wordnet_pos = self._nltk_pos_to_wordnet(nltk_pos)
        synsets = wn.synsets(word, pos=wordnet_pos)
        for synset in synsets:
            candidates.extend(synset.lemmas())
        candidates = [s.name() for s in candidates]
        synonyms = []
        for c in candidates:
            if ('_' in c) or (
                c == word) or (
                c in self.stop_words) or (
                c not in self.vocab):
                continue
            else:
                synonyms.append(c)
        return synonyms

    def find_synonyms_for_tokens(self, words):
        synsets = []
        pos_tags = self.pos_tagger.tag(words)
        for i, w in enumerate(words):
            synsets.append(self.find_synonyms(w, pos_tags[i][1]))
        return synsets

    def get_word_net_synonyms(self,word):
        """
        Code refers to https://github.com/JHL-HUST/PWWS/blob/master/paraphrase.py
        """
        if (word in self.stop_words) or (
            word not in self.vocab) :
            return []
        synonyms = []

        for synset in wn.synsets(word):
            for w in synset.lemmas():
                synonyms.append(w.name().replace("_", " "))

        synonyms = sorted(
            list(set([x.lower() for x in synonyms if len(x.split()) == 1]) - {word})
        )
        synonymsout =[]
        for c in synonyms:
            if ('_' in c) or (
                c == word) or (
                c in self.stop_words) or (
                c not in self.vocab):
                continue
            else:
                synonymsout.append(c)
        return synonymsout    