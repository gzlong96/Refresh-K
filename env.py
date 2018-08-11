import numpy as np
import os
import time
import config
import random

class ENV:

    metadata = {'render.modes': ['human']}

    def __init__(self, data_loader):
        np.random.seed(1234567)

        self.gamestep = 0
        self.episode_count = 0

        self.data_loader = data_loader
        self.data_loader.read_data(split='test')

        self.doc = None
        self.embedded_doc = None
        self.highlight = None
        self.mask = np.ones((config.doc_max_len,), dtype=np.int32)

    def reset(self):
        self.gamestep = 0
        self.episode_count += 1

        self.mask = np.ones((config.doc_max_len,), dtype=np.int32)

        self.embedded_doc, self.doc, self.highlight = self.data_loader.random_sample_one()

        return [self.embedded_doc, self.mask]

    def step(self, action):
        if action != config.doc_max_len:
            self.mask[action] = 0
            self.gamestep += 1

        if self.gamestep == config.max_step or action == config.doc_max_len:
            done = True
        else:
            done = False

        reward = self.get_reward()

        return [self.embedded_doc, self.mask], reward, done, {}

    def get_reward(self):

        return self.rouge1()

    def rouge1(self):
        label_dic = {}
        answer_dic = {}

        total_label_words = 0
        total_answer_words = 0

        for sentence in self.highlight:
            for word in sentence:
                total_label_words += 1
                if word not in label_dic:
                    label_dic[word] = 1
                else:
                    label_dic[word] += 1

        summary = []
        for i in range(config.doc_max_len if config.doc_max_len<len(self.doc) else len(self.doc)):
            if self.mask[i]==0:
                summary.append(self.doc[i])

        for sentence in summary:
            for word in sentence:
                if word not in answer_dic:
                    answer_dic[word] = 1
                else:
                    answer_dic[word] += 1

        for word in list(answer_dic.keys()):
            if word in label_dic:
                total_answer_words += min([label_dic[word], answer_dic[word]])

        return total_answer_words/total_label_words
