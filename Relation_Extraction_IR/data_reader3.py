from collections import Counter


import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from utils import *
from sklearn import preprocessing
import os
import csv
import os
from metadata import *
from glob import glob
from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm")

class PaperReader(object):
    def __init__(self,corpus='NLP-TDMS'):
        self.corpus = corpus
        self.paper_text,self.paper_table = self.read_papers()

    def get_paperIDs(self):
        paper_fnames = []
        with open(os.path.join(NLP_TDM_path,'downloader','paper_links.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                paper_fnames.append(line[0])
        return paper_fnames

    def get_PaperTxt_files(self):
        return glob(os.path.join(NLP_TDM_path,'pdfFile_txt','*'))

    def get_PaperTable_files(self):
        return glob(os.path.join(NLP_TDM_path, 'pdfFile_table', '*'))

    def read_papers(self):
        pids = self.get_paperIDs()
        text_fnames = self.get_PaperTxt_files()
        text_dict = dict()
        for t_fname in text_fnames:
            text_dict[os.path.basename(t_fname).split('.txt')[0]] = t_fname

        table_fnames = self.get_PaperTable_files()
        table_dict = dict()
        for t_fname in table_fnames:
            table_dict[os.path.basename(t_fname).split('.json')[0]] = t_fname

        paper_text = dict()
        paper_table = dict()
        for pid in pids:
            pid = pid.split('.pdf')[0]
            if pid in text_dict:
                f = open(text_dict[pid],'r')
                paper_text[pid] = f.read()
                f.close()
            if pid in table_dict:
                f = open(table_dict[pid],'r')
                paper_table[pid] = json.loads(f.readline())
                f.close()
        return paper_text,paper_table

    def get_paper_datasets(self):
        paper_dataset = dict()
        with open(os.path.join(NLP_TDM_path, 'annotations', 'datasetAnnotation.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                paper_dataset[line[0].split('.pdf')[0]] = line[1]
        return paper_dataset

    def get_paper_task(self):
        paper_task = dict()
        with open(os.path.join(NLP_TDM_path, 'annotations', 'taskAnnotation.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                paper_task[line[0].split('.pdf')[0]] = line[1]
        return paper_task

    def get_paper_result(self):
        #each paper entry has the format:
        #task1#dataset1#metric1#score1$task2#dataset2#metric2#score2$...
        paper_result = dict()
        with open(os.path.join(NLP_TDM_path, 'annotations', 'resultsAnnotation.tsv')) as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                paper_result[line[0].split('.pdf')[0]] = line[1]
        return paper_result

def load_pretrained_wv(path):
    wv = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.split(' ')
            wv[items[0]] = torch.DoubleTensor([float(a) for a in items[1:]])
    return wv

def pad_or_crop(field,max_tokens,to_add):
    if len(field)>max_tokens:
        field=field[0:max_tokens]
    if len(field)<max_tokens:
        for i in range(max_tokens-len(field)):
            field+=to_add
    return field


def pad_or_crop_with_rep(field,max_tokens,dictt,field_type):

    final=[]

    for f in field:
        if f in dictt.keys():
            final.append(f)
        else:
            final.append('unk')

    if len(final)>max_tokens:
        final=final[0:max_tokens]
    if len(final)<max_tokens:
        inter=final.copy()
        if len(inter)==0:
            #print('here empty')
            if field_type in ['description','attributes']:
                inter=[',']
            else:
                inter = ['.']
        j=0
        for i in range(max_tokens-len(final)):
            final.append(inter[j%len(inter)])
            j+=1

    return final

def encode_field(field,dictt,field_type):
    vect_ = []
    for qu in field:
        try:
            vect_ += [dictt[qu]]
        except:

            if field_type in ['description','attributes']:
                vect_ += [dictt[',']]
            else:
                vect_ += [dictt['.']]

    return vect_



class DataAndQuery(Dataset):
    def __init__(self,wv=None,word_to_index=None,index_to_word=None,train=True,train_set=None,labels_set=None,use_same_collection=False,nb_all=0,nb_train=0):

        if wv and word_to_index and index_to_word:
            self.wv=wv
            self.word_to_index=word_to_index
            self.index_to_word=index_to_word
            self.train_set=train_set
            self.labels_set=labels_set
            self.nb_all=nb_all
            self.nb_train = nb_train

        else:
            self.word_to_index = {}
            self.index_to_word = []
            self.train_set=set()
            self.labels_set=set()

            pretrained_wv = '/home/mohamed/PycharmProjects/glove.6B/glove.6B.50d.txt'
            #pretrained_wv = '/home/mohamed/PycharmProjects/glove.6B/glove.6B.300d.txt'
            self.wv = load_pretrained_wv(pretrained_wv)
            for i,key in enumerate(self.wv.keys()):
                self.word_to_index[key]=i
                self.index_to_word.append(key)

        preader = PaperReader()
        paper_result = preader.get_paper_result()
        paper_text, paper_table = preader.read_papers()
        if train:
            nb_all=0
            for pid in paper_result:
                #print(pid)
                if pid not in paper_text:
                    continue
                results = paper_result[pid]
                results = results.split('$')
                nb_all+=1

                for result in results:
                    seps = result.split('#')
                    task, dataset, metric, score = seps[:4]
                    label=task+' '+dataset+' '+metric
                    query_tokens = preprocess(label, 'description')
                    query_tokens=' '.join(query_tokens)
                    self.labels_set.add(query_tokens)
            nb_train = int(0.8 * nb_all)
            self.nb_all=nb_all
            self.nb_train=nb_train

        if train:
            if use_same_collection:
                paper_result=np.load('train.npy',allow_pickle=True)
                paper_result=paper_result[()]
        else:
            if use_same_collection:
                paper_result=np.load('test.npy',allow_pickle=True)
                paper_result=paper_result[()]

        self.test_samples= {}
        self.train_samples = {}
        max_tokens_desc = 200
        max_tokens_query = 15
        nb_tokens_per_line=20

        all_desc = []
        all_query = []
        labels = []
        all_query_labels=[]

        nb_negative_samples=5

        if use_same_collection:
            nb_samples=len(paper_result)
        else:
            if train:
                nb_samples=self.nb_train
            else:
                nb_samples=self.nb_all-self.nb_train

        #nb_samples = 5
        actual_number_of_train=0
        with tqdm(total=nb_samples) as pbar:
            for index,pid in enumerate(paper_result):

                #
                # if not train and index>2*nb_samples:
                #     break
                #print(pid)

                if pid not in paper_text:
                    continue
                actual_number_of_train+=1
                if train and actual_number_of_train>nb_samples:
                    break
                if train:
                    self.train_set.add(pid)
                    self.train_samples[pid]=paper_result[pid]
                if not train:
                    if pid in self.train_set:
                        continue

                if not train:
                    self.test_samples[pid]=paper_result[pid]


                text = paper_text[pid]
                lines = text.split('\n')
                i=0
                text_paper_to_use=''
                while i<len(lines):
                    if len(lines[i]) == 0:
                        i+=1
                        continue
                    doc = nlp(lines[i])
                    while doc[0].text == 'section' and doc[1].text == ':' and i<len(lines):
                        i+=1
                        while i<len(lines) and len(lines[i]) == 0:
                            i+=1
                        if i<len(lines):
                            doc = nlp(lines[i])
                        else:
                            break

                    if i<len(lines):
                        text_paper_to_use+=doc[:nb_tokens_per_line].text+' '
                        i += 1


                description = preprocess(text_paper_to_use, 'description')
                description = pad_or_crop_with_rep(description, max_tokens_desc, self.word_to_index, 'description')
                vector_desc = [encode_field(description, self.word_to_index, 'attributes')]


                results = paper_result[pid]
                results = results.split('$')
                paper_labels=set()

                for result in results:
                    seps = result.split('#')
                    task, dataset, metric, score = seps[:4]
                    label = task + ' ' + dataset + ' ' + metric
                    query_tokens = preprocess(label, 'description')
                    query_ = pad_or_crop_with_rep(query_tokens, max_tokens_query, self.word_to_index, 'query')
                    vector_query = [encode_field(query_, self.word_to_index, 'query')]

                    all_desc.append([vector_desc])
                    all_query.append([vector_query])
                    labels.append(1)
                    query_tokens = ' '.join(query_tokens)
                    paper_labels.add(query_tokens)

                ns=0

                for lb in self.labels_set:
                    if lb not in paper_labels:
                        ns+=1
                        query_tokens=lb.split(' ')
                        query_ = pad_or_crop_with_rep(query_tokens, max_tokens_query, self.word_to_index, 'query')
                        vector_query = [encode_field(query_, self.word_to_index, 'query')]
                        all_desc.append([vector_desc])
                        all_query.append([vector_query])
                        labels.append(0)
                    if ns==nb_negative_samples:
                        break

                pbar.update(1)



        self.all_desc = all_desc
        self.all_desc = torch.tensor(self.all_desc)
        self.all_query = all_query
        self.all_query = torch.tensor(self.all_query)

        self.all_query_labels=all_query_labels
        self.labels = labels

        if not use_same_collection:
            if train:
                np.save('train',self.train_samples)
            else:
                np.save('test', self.test_samples)






    def __getitem__(self, t):
        """
            return: the t-th (center, context) word pair and their co-occurrence frequency.
        """
        ## Your codes go here
        return self.all_desc[t],self.all_query[t],self.labels[t]

    def __len__(self):
        """
            return: the total number of (center, context) word pairs.
        """
        ## Your codes go here
        return len(self.all_desc)