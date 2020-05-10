'''
1. from each paper, find the voted entities
2. building only one tuple from voted entities
3. save the processed data for other methods

for voting method, calculate MRR for each entity type
'''

from metadata import *
from reader import PaperReader,KBReader
from glob import glob
import json
import pickle
import numpy as np
from fuzzywuzzy import fuzz
from collections import defaultdict,Counter
import spacy
nlp = spacy.load("en_core_web_sm")


sim_thred = 50
data_thred = 60
preader = PaperReader()
kbreader = KBReader()
paper_result = preader.get_paper_result()
paper_text, paper_table = preader.read_papers()

#data to save
paper_title = dict()
paper_task = defaultdict(list)
paper_dataset = defaultdict(list)
paper_metric = defaultdict(list)
paper_scores = defaultdict(list)
#top-n entities in a paper
paper_entities = dict()

#input for KB construction
f_rs = open(vote_rs_path,'w')
#f_rs = open('debug','w')

#for evaluation
task_MRRs = []
dataset_MRRs = []
metric_MRRs = []
score_MRRs = []
tuples = [] # (title, task,dataset,metric,score)
for pid in paper_result:
    print(pid)
    if pid not in paper_text:
        continue
    text = paper_text[pid]
    lines = text.split('\n')
    next_is_title = False
    for line in lines:
        if len(line) == 0:
            continue
        doc = nlp(line)
        #print(line)
        # first check Sections
        if doc[0].text == 'section' and doc[1].text == ':':
            #check paper title
            if doc[2].text == 'title':
                next_is_title = True
            continue

        if next_is_title:
            next_is_title = False
            title = doc.text
            paper_title[pid] = title
            continue

        for sent in doc.sents:
            #entity matching with lexical method
            for nph in sent.noun_chunks:
                #compare with all task candidates
                task_sims = [fuzz.ratio(nph.text.lower(),task.replace('_',' ').lower()) for task in kbreader.task_dict.values()]
                data_sims = [fuzz.ratio(nph.text.lower(), dataset.lower()) for dataset in kbreader.dataset_dict.values()]
                metric_sims = [fuzz.ratio(nph.text.lower(), metric.replace('_',' ').lower()) for metric in kbreader.metric_dict.values()]
                task_best = max(task_sims)
                data_best = max(data_sims)
                metric_best = max(metric_sims)
                if task_best == max([task_best,data_best,metric_best]):
                    if task_best > sim_thred:
                        task_label = list(kbreader.task_dict.values())[np.argmax(task_sims)]
                        paper_task[pid].append([task_label,sent.text])
                elif data_best == max([task_best,data_best,metric_best]):
                    #compare with all dataset candidates
                    if max(data_sims) > data_thred:
                        #print(list(kbreader.dataset_dict.values())[np.argmax(data_sims)])
                        dataset_label = list(kbreader.dataset_dict.values())[np.argmax(data_sims)]
                        paper_dataset[pid].append([dataset_label, sent.text])
                elif metric_best == max([task_best,data_best,metric_best]):
                    # compare with all metric candidates
                    if max(metric_sims) > sim_thred:
                        metric_label = list(kbreader.metric_dict.values())[np.argmax(metric_sims)]
                        paper_metric[pid].append([metric_label,sent.text])


    #process paper tables
    #currently only keep bolded numbers
    for table in paper_table[pid]:
        for cell in table['numberCells']:
            #if cell['isBolded']:
            number = cell['number']
            try:
                paper_scores[pid].append([table['caption'],float(cell['number']),cell['associatedRows'],cell['associatedColumns'],cell['associatedMergedColumns']])
            except:
                continue
    best_score = max([each[1] for each in paper_scores[pid]]) if len(paper_scores[pid]) != 0 else -1
    # print(best_score)

    #begin voting for this paper
    sorted_tasks = [each[0] for each in Counter([each[0] for each in paper_task[pid]]).most_common()]
    sorted_datasets = [each[0] for each in Counter([each[0] for each in paper_dataset[pid]]).most_common()]
    sorted_metrics = [each[0] for each in Counter([each[0] for each in paper_metric[pid]]).most_common()]
    sorted_scores = sorted([each[1] for each in paper_scores[pid]],reverse=True) if len(paper_scores[pid]) != 0 else  [-1]
    voted_task = sorted_tasks[0] if len(sorted_tasks) != 0 else 'null'
    voted_dataset = sorted_datasets[0] if len(sorted_datasets) != 0 else 'null'
    voted_metric = sorted_metrics[0] if len(sorted_metrics) != 0 else 'null'
    # print(voted_task,voted_dataset,voted_metric,best_score)
    tuples.append([paper_title[pid],voted_task,voted_dataset,voted_metric,best_score])
    f_rs.write('$'.join([paper_title[pid],voted_task,voted_dataset,voted_metric,str(best_score)]) + '\n' )

    #save top-n entities
    paper_entities[pid] = [sorted_tasks[:5],sorted_datasets[:5],sorted_metrics[:5],sorted_scores[:5]]

    #EVALUATION
    #entity MRR/top-n Acc
    results = paper_result[pid]
    results = results.split('$')
    for result in results:
        seps = result.split('#')
        task, dataset, metric, score = seps[:4]
        try:
            score = float(score)
        except:
            score = -1
        #task
        if task in sorted_tasks:
            rank = sorted_tasks.index(task)
            rank = rank + 1
            task_MRRs.append(1.0/rank)
        else:
            task_MRRs.append(0)
        #dataset
        if dataset in sorted_datasets:
            rank = sorted_datasets.index(dataset)
            rank = rank + 1
            dataset_MRRs.append(1.0/rank)
        else:
            dataset_MRRs.append(0)
        # metric
        if metric in sorted_metrics:
            rank = sorted_metrics.index(metric)
            rank = rank + 1
            metric_MRRs.append(1.0 / rank)
        else:
            metric_MRRs.append(0)
        # scores
        if score in sorted_scores:
            rank = sorted_scores.index(score)
            rank = rank + 1
            score_MRRs.append(1.0 / rank)
        else:
            score_MRRs.append(0)


f_rs.close()
#save the processed data for other methods
#pickle.dump([paper_title,paper_task,paper_dataset,paper_metric,paper_scores,paper_entities],open('debug_rs','wb'))
pickle.dump([paper_title,paper_task,paper_dataset,paper_metric,paper_scores,paper_entities],open(paper_entity_path,'wb'))
print("Task MMR:{}".format(np.mean(task_MRRs)))
print("Dataset MMR:{}".format(np.mean(dataset_MRRs)))
print("Metric MMR:{}".format(np.mean(metric_MRRs)))
print("Score MMR:{}".format(np.mean(score_MRRs)))


'''
Task MMR:0.4775281993333612
Dataset MMR:0.19581078457261428
Metric MMR:0.13147365216880882
Score MMR:0.065931711 45872177
'''

