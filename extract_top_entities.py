from model import *
from utils import *
import itertools
import argparse
import os
import sys
from collections import Counter
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LSTM-CRF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--CharToIdx", type=str, default='/home/mohamedt/scientific_data/data/train.char_to_idx',
                    help="char_to_idx")
parser.add_argument("--WordToIdx", type=str, default='/home/mohamedt/scientific_data/data/train.word_to_idx',
                    help="word_to_idx")
parser.add_argument("--TagToIdx", type=str, default='/home/mohamedt/scientific_data/data/train.tag_to_idx',
                    help="tag_to_idx")

parser.add_argument("--test", type=str, default='/home/mohamedt/scientific_data/data/data',
                     help="dataset to use")
parser.add_argument("--checkpoint", type=str, default='/home/mohamedt/scientific_data/data/phase1_model.epoch70',
                     help="checkpoint")

parser.add_argument('--device', type=int, default=0)


args = parser.parse_args()


def load_model():
    cti = load_tkn_to_idx(args.CharToIdx) # char_to_idx
    wti = load_tkn_to_idx(args.WordToIdx) # word_to_idx
    itt = load_idx_to_tkn(args.TagToIdx) # idx_to_tag
    model = rnn_crf(len(cti), len(wti), len(itt))
    #print(model)
    load_checkpoint(args.checkpoint, model)
    return model, cti, wti, itt

def run_model(model, itt, data):
    data.sort()
    for batch in data.split():
        xc, xw = data.tensor(batch.xc, batch.xw, batch.lens)
        y1 = model.decode(xc, xw, batch.lens)
        data.y1.extend([[itt[i] for i in x] for x in y1])
    data.unsort()
    for x0, y0, y1 in zip(data.x0, data.y0, data.y1):
        if HRE:
            for x0, y0, y1 in zip(x0, y0, y1):
                yield x0, y0, y1
        else:
            yield x0[0], y0, y1

def predict(filename, model, cti, wti, itt):


    all_papers_data={}

    lines=[]
    max_length=0
    with open(filename) as fo:
        for line in fo:
            line = line.rstrip().replace("\\tag", "/").replace('\t', ' ')

            line=line.split(' ',1)
            paper_id = line[0]
            line=line[1]
            line = re.sub(' +', ' ', line)
            lines.append(line)
            all_papers_data[paper_id]=line

    with tqdm(total=len(all_papers_data)) as bar1:
        for key in all_papers_data:
            text = [all_papers_data[key]]
            data = dataloader()
            for block in text:
                for x0 in block.split("\n"):


                    res=[]
                    y0=[]

                    for num, w in enumerate(x0.split(" ")):

                        if num==0 and not re.match("\S+/\S+( \S+/\S+)*$", w):
                            res=x0.split(" ")
                            y0 = []
                            break

                        w, tag = (w, None) if HRE else re.split("/(?=[^/]+$)", w)

                        w0 = normalize(w)  # for character embedding
                        if len(w0) == 1:
                            w0 += '<punct>'
                        if len(w0) == 0:
                            w0 = '<punct><punct>'
                        w1 = w0.lower()  # for word embedding
                        res.append(w1)
                        y0.append(tag)

                    res=' '.join(res)
                    res=res.split(' ')
                    seq_x=[]
                    seq_y=[]

                    for index,rel in enumerate(y0):
                        seq_x.append(res[index])
                        seq_y.append(rel)

                        if rel=='SENT_END':
                            if len(seq_x)>max_length:
                                max_length=len(seq_x)
                            seq_x = ' '.join(seq_x)
                            x1 = tokenize(seq_x)
                            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
                            xw = [wti[w] if w in wti else UNK_IDX for w in x1]
                            data.append_item(x0=[seq_x], xc=[xc], xw=[xw], y0=seq_y)
                            data.append_row()
                            seq_x=[]
                            seq_y=[]
                    if len(seq_x)>0:
                        if len(seq_x)>max_length:
                            max_length=len(seq_x)
                        seq_x = ' '.join(seq_x)
                        x1 = tokenize(seq_x)
                        xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
                        xw = [wti[w] if w in wti else UNK_IDX for w in x1]
                        data.append_item(x0=[seq_x], xc=[xc], xw=[xw], y0=seq_y)
                        data.append_row()

            data.strip()
            all_papers_data[key]=data
            bar1.update(1)
    #print(max_length)
    with torch.no_grad():
        model.eval()
        for key in all_papers_data:
            data=all_papers_data[key]
            all_papers_data[key]=run_model(model, itt, data)

    return all_papers_data

def extract_TDM_triples(section,predicted_tags):
    i=0
    tasks=[]
    datasets=[]
    metrics=[]

    while i<len(section):
        task=[]
        dataset=[]
        metric=[]
        if predicted_tags[i]=="task":
            task.append(section[i].replace("<punct>",''))
            while i<len(section)-1 and predicted_tags[i+1]=="task":
                i+=1
                task.append(section[i].replace("<punct>",''))

            task=" ".join(task)
            tasks.append(task)

        #if i==len(section):
        #    break

        if predicted_tags[i]=="dataset":
            dataset.append(section[i].replace("<punct>",''))
            while i<len(section)-1 and predicted_tags[i+1]=="dataset":
                i+=1
                dataset.append(section[i].replace("<punct>",''))
            dataset=" ".join(dataset)
            datasets.append(dataset)

        #if i==len(section):
        #    break

        if predicted_tags[i]=="metric":
            metric.append(section[i].replace("<punct>",''))
            while i<len(section)-1 and predicted_tags[i+1]=="metric":
                i+=1
                metric.append(section[i].replace("<punct>",''))
            metric=" ".join(metric)
            metrics.append(metric)
        i+=1

    return tasks,datasets,metrics

if __name__ == "__main__":


    top_entities={}
    with open('top_tags.pickle', 'rb') as handle:
        b = pickle.load(handle)

    papers_tags=predict(args.test, *load_model())
    with tqdm(total=len(papers_tags)) as pbar:
        for key in papers_tags:
            tasks = []
            datasets = []
            metrics = []

            g_tasks=[]
            g_datasets=[]
            g_metrics=[]

            for x0, y0, y1 in papers_tags[key]:

                #print(x0.split(' '))
                #print(y0)
                #print(y1)
                #print('\n')
                sec_tasks,sec_datasets,sec_metrics=extract_TDM_triples(x0.split(' '), y1)
                tasks += sec_tasks
                datasets += sec_datasets
                metrics += sec_metrics
                sec_tasks,sec_datasets,sec_metrics=extract_TDM_triples(x0.split(' '), y0)
                g_tasks += sec_tasks
                g_datasets += sec_datasets
                g_metrics += sec_metrics


            tasks=Counter(tasks)
            metrics = Counter(metrics)
            datasets = Counter(datasets)
            tasks_ordered=sorted(tasks, key=tasks.get, reverse=True)
            metrics_ordered = sorted(metrics, key=metrics.get, reverse=True)
            datasets_ordered = sorted(datasets, key=datasets.get, reverse=True)

            g_tasks=Counter(g_tasks)
            g_metrics = Counter(g_metrics)
            g_datasets = Counter(g_datasets)
            g_tasks_ordered=sorted(g_tasks, key=g_tasks.get, reverse=True)
            g_metrics_ordered = sorted(g_metrics, key=g_metrics.get, reverse=True)
            g_datasets_ordered = sorted(g_datasets, key=g_datasets.get, reverse=True)


            top_entities[key]={}
            top_entities[key]['task']=tasks_ordered[:5]
            top_entities[key]['dataset'] = datasets_ordered[:5]
            top_entities[key]['metric'] = metrics_ordered[:5]

            top_entities[key]['g_task']=g_tasks_ordered
            top_entities[key]['g_dataset'] = g_datasets_ordered
            top_entities[key]['g_metric'] = g_metrics_ordered


            pbar.update(1)

    with open('top_tags.pickle', 'wb') as handle:
        pickle.dump(top_entities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')

