import collections
import csv
import os
from metadata import *
from glob import glob
import json
from fuzzywuzzy import fuzz
import spacy
nlp = spacy.load("en_core_web_sm")

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class SciProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def tokenizer(self,sentence):
        return sentence.split()

    def get_test_result(self, testdata_dir, output_dir):
        testlines = self._read_tsv(os.path.join(testdata_dir, "test.tsv"))
        testresult = self._read_tsv(os.path.join(output_dir, "test_results.tsv"))
        prediction_mention = {}
        prediction_value = {}
        prediction_ground = {}
        for (i, line) in enumerate(testlines):
            line = line + testresult[i - 1]
            if line[1] in prediction_mention:
                if (float(line[6]) > prediction_value.get(line[1])):
                    prediction_mention[line[1]] = line[2]
                    prediction_value[line[1]] = float(line[6])
                    prediction_ground[line[1]] = line[0]
            else:
                prediction_mention[line[1]] = line[2]
                prediction_value[line[1]] = float(line[6])
                prediction_ground[line[1]] = line[0]
        for i, j in prediction_mention.items():
            print(i, j)
        tp = 0
        for i in prediction_ground.values():
            if i == 1:
                tp = tp + 1
        print(prediction_ground.keys().__len__(), tp)
        print("evaluation finished")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = self.tokenizer(line[2])
            text_b = self.tokenizer(line[3])
            if set_type == "test":
                label = "true"
            else:
                label = self.tokenizer(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


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


'''
for each paper id:
    get paper result
    get task, dataset, metric, score from result
    matching those entities in text and record the token
    scores should be matched only in tables
'''
sim_thred = 50
output_mode = 'key_sent' # all_sec / key_sec
preader = PaperReader()
paper_result = preader.get_paper_result()
paper_text, paper_table = preader.read_papers()
f_out = open('./data/all.data','w')
for pid in paper_result:
    print(pid)
    if pid not in paper_text:
        continue
    text = paper_text[pid]
    lines = text.split('\n')
    data_pairs = []
    for line in lines:
        if len(line) == 0:
            continue
        doc = nlp(line)
        # first check Sections
        if doc[0].text == 'section' and doc[1].text == ':':
            for sec_idx in range(2,len(doc)):
                if sec_idx == 2:
                    if len(doc) > 3:
                        data_pairs.append([doc[sec_idx].text,'SECTITLE_START'])
                    else:
                        data_pairs.append([doc[sec_idx].text, 'SECTITLE_END'])
                elif sec_idx + 1< len(doc):
                    data_pairs.append([doc[sec_idx].text, 'SECTITLE_CONTENT'])
                elif sec_idx + 1 == len(doc):
                    data_pairs.append([doc[sec_idx].text, 'SECTITLE_END'])
            continue

        entity_entries = []  # (start,end,entity type)
        results = paper_result[pid]
        results = results.split('$')
        has_dataset = False
        for result in results:
            seps = result.split('#')
            task,dataset,metric,score = seps[:4]
            #print(seps)
            # fuzzy matching for entities
            for np in doc.noun_chunks:
                sim = fuzz.ratio(np.text.lower(),task.replace('_',' ').lower())
                #sim = textdistance.jaccard(np.text.lower(),task.replace('_',' ').lower())
                if sim > sim_thred:
                    entity_entries.append([np.start,np.end,'task'])
                sim = fuzz.ratio(np.text.lower(), dataset.lower())
                #sim = textdistance.jaccard(np.text.lower(), dataset.replace('_',' ').lower())
                if sim == 100:
                    entity_entries.append([np.start, np.end, 'dataset'])
                    has_dataset= True
                    #print("{0}:{1}:{2}".format(np.text,dataset,sim ))
                sim = fuzz.ratio(np.text.lower(), metric.replace('_',' ').lower())
                #sim = textdistance.jaccard(np.text.lower(), metric.replace('_',' ').lower())
                if sim > sim_thred:
                    entity_entries.append([np.start, np.end, 'metric'])
                    #print("{0}:{1}:{2}".format(np.text, metric, sim))

        entity_entries.sort(key=lambda x: x[0])

        #begin to output every token
        if len(entity_entries) == 0:
            current_entity = [len(doc)+1,len(doc)+1]
        else:
            current_entity = entity_entries.pop(0)

        if output_mode == 'all_sec':
            idx = 0
            while idx < len(doc):
                if idx == 0:
                    data_pairs.append([doc[idx].text,'SEC_START'])
                elif idx + 1 < len(doc):
                    #current token belongs to an entity
                    if current_entity[0] == idx:
                        data_pairs.append([doc[idx].text, current_entity[2]+"START"])
                    elif idx ==  current_entity[1]-1:
                        data_pairs.append([doc[idx].text, current_entity[2] + "END"])
                    elif  current_entity[0] < idx+1 <  current_entity[1]:
                        data_pairs.append([doc[idx].text,current_entity[2]])
                    else:
                        data_pairs.append([doc[idx].text, 'SEC_CONTENT'])
                    #finish tag the entity
                    if idx+1 == current_entity[1] and len(entity_entries) != 0:
                        current_entity = entity_entries.pop(0)

                elif idx + 1 == len(doc):
                    data_pairs.append([doc[idx].text, 'SEC_END'])
                idx += 1
        if output_mode == 'key_sent':
            sents = list(doc.sents)
            for sent in sents:
                has_ent = False
                if len(sent) < 3:
                    continue
                #if the sentence does not have an entity, skip
                if sent.start <= current_entity[0] and sent.end >=  current_entity[1]:
                    idx = sent.start

                    while idx < sent.end:
                        if idx == sent.start and idx != current_entity[0]:
                            data_pairs.append([sent[idx-sent.start].text, 'SENT_START'])
                        elif idx + 1 < sent.end:
                            # current token belongs to an entity
                            if current_entity[0] == idx:
                                has_ent = True
                                data_pairs.append([sent[idx-sent.start].text, current_entity[2] + "START"])
                                #print("{0}:{1}".format(current_entity[2],sent[idx-sent.start].text))
                            elif idx == current_entity[1] - 1:
                                data_pairs.append([sent[idx-sent.start].text, current_entity[2] + "END"])
                            elif current_entity[0] < idx + 1 < current_entity[1]:
                                data_pairs.append([sent[idx-sent.start].text, current_entity[2]])
                            else:
                                data_pairs.append([sent[idx-sent.start].text, 'SENT_CONTENT'])
                            # finish tag the entity
                            if idx + 1 == current_entity[1] and len(entity_entries) != 0:
                                current_entity = entity_entries.pop(0)

                        elif idx + 1 ==  sent.end and idx+1 != current_entity[1]:
                            data_pairs.append([sent[idx-sent.start].text, 'SENT_END'])
                        idx += 1


    #write into files
    f_out.write(pid + '\t')
    for pair in data_pairs:
        f_out.write(pair[0] + '\\tag' + pair[1] + '\t')
        #print(pair[0] + '\t' + pair[1])
    f_out.write('\n')


f_out.close()











