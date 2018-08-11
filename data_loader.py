import time
import os
import numpy as np
import random

import config

class DataLoader():
    def __init__(self):
        self.word_embed = None
        self.word_index_map = {}
        self.load_word_embedding()

    def load_word_embedding(self):
        time1 = time.time()
        with open('data/word2vec.vec', encoding='UTF-8') as word_embed_file:
            nb_word, embedding_len = word_embed_file.readline().split()
            nb_word = eval(nb_word)
            embedding_len = eval(embedding_len)

            self.word_embed = np.zeros((nb_word+1, embedding_len), dtype=np.float32)

            for i in range(nb_word):
                line = word_embed_file.readline().split()

                self.word_index_map[line[0]] = i+1
                self.word_embed[i+1] = list(map(float, line[1:]))

        print('finish load embedding: '+str(time.time()-time1))

    def read_data(self, source='cnn', split='training'):
        self.doc_list = self.read_one(source=source, split=split,part='doc')
        self.image_list = self.read_one(source=source, split=split, part='image')
        self.title_list = self.read_one(source=source, split=split, part='title')
        self.highlight_list = self.read_one(source=source, split=split, part='highlights')

        return self.doc_list, self.image_list, self.title_list, self.highlight_list

    def read_one(self, source='cnn', split='training', part='doc'):
        max_sentence_len = 0
        max_doc_len = 0

        doc_list = []
        with open('data/preprocessed-input-directory/'+source+'.'+split+'.'+part, encoding='UTF-8') as doc:
            doc_count = 0
            line = doc.readline()
            while line:
                if '0'<line[0]<='9':
                    line = line.split()
                    doc_count += 1
                    if doc_count > max_doc_len:
                        max_doc_len = doc_count
                    if len(line)>max_sentence_len:
                        max_sentence_len = len(line)
                    doc_list[-1].append(list(map(int, line)))
                elif 'a'<=line[0]<='z':
                    doc_list.append([])
                    doc_count = 0
                else:
                    pass
                line = doc.readline()
        print(source,split,part, max_sentence_len, max_doc_len)
        return doc_list

    def to_embedding(self, doc):
        embedded = np.zeros([config.doc_max_len, config.sentence_max_len, 200])
        for i in range(min([config.doc_max_len, len(doc)])):
            for j in range(min([config.sentence_max_len, len(doc[i])])):
                embedded[i][j] = self.word_embed[doc[i][j]]
        return embedded

    def random_sample_one(self):
        random_sample_index = random.randint(0,len(self.doc_list)-1)
        doc = self.to_embedding(self.doc_list[random_sample_index])
        return doc, self.doc_list[random_sample_index], self.highlight_list[random_sample_index]


if __name__ == '__main__':
    DL = DataLoader()
    DL.read_data(split='test')
