import sys
sys.path.append('/home/mohamedt/scientific_data')
from model import *
from utils import *
import itertools
import argparse
from PrepareIRInput import DataAndQuery
from torch.utils.data import Dataset,DataLoader
import os
#sys.path.append('/home/mohamedt/scientific_data/Relation_Extraction_IR')
from convknrm_model import CONVKNRM

parser = argparse.ArgumentParser(description='LSTM-CRF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--CharToIdx", type=str, default='/home/mohamedt/scientific_data/data/train.char_to_idx',
                    help="char_to_idx")
parser.add_argument("--WordToIdx", type=str, default='/home/mohamedt/scientific_data/data/train.word_to_idx',
                    help="word_to_idx")
parser.add_argument("--TagToIdx", type=str, default='/home/mohamedt/scientific_data/data/train.tag_to_idx',
                    help="tag_to_idx")

parser.add_argument("--test", type=str, default='/home/mohamedt/scientific_data/data/demo.input1',
                     help="dataset to use")
parser.add_argument("--checkpoint", type=str, default='/home/mohamedt/scientific_data/data/phase1_model.epoch70',
                     help="phase1 model")

parser.add_argument("--pretrained_embedding", type=str, default='/home/mohamedt/ARCI/glove.6B.50d.txt',
                     help="pretrained embedding")
parser.add_argument("--phase2_model", type=str, default='/home/mohamedt/scientific_data/Relation_Extraction_IR/stage2_model.pt',
                     help="checkpoint")


parser.add_argument('--emsize', type=int, default=50)
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--nbins', type=int, default=5)

args = parser.parse_args()

def kernal_mus(n_kernels):
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

args.mu = kernal_mus(args.nbins)
args.sigma = kernel_sigmas(args.nbins)

def load_pretrained_wv(path):
    wv = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.split(' ')
            wv[items[0]] = torch.DoubleTensor([float(a) for a in items[1:]])
    return wv

pretrained_wv = args.pretrained_embedding
#pretrained_wv = '/home/mohamed/PycharmProjects/glove.6B/glove.6B.300d.txt'
wv = load_pretrained_wv(pretrained_wv)
word_to_index={}
index_to_word=[]
for i,key in enumerate(wv.keys()):
    word_to_index[key]=i
    index_to_word.append(key)

def load_checkpoint_for_eval(model, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


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
    data = dataloader()

    lines=[]
    max_length=0
    with open(filename) as fo:
        for line in fo:
            line = line.rstrip().replace("\\tag", "/").replace('\t', ' ')
            global paper_id

            line=line.split(' ',1)
            paper_id = line[0]
            line=line[1]
            line = re.sub(' +', ' ', line)
            lines.append(line)

    text=lines
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
    #print(max_length)
    with torch.no_grad():
        model.eval()
        return run_model(model, itt, data)


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
            while i<len(section) and predicted_tags[i+1]=="task":
                i+=1
                task.append(section[i].replace("<punct>",''))

            task=" ".join(task)
            tasks.append(task)

        if i==len(section):
            break

        if predicted_tags[i]=="dataset":
            dataset.append(section[i].replace("<punct>",''))
            while i<len(section) and predicted_tags[i+1]=="dataset":
                i+=1
                dataset.append(section[i].replace("<punct>",''))
            dataset=" ".join(dataset)
            datasets.append(dataset)

        if i==len(section):
            break

        if predicted_tags[i]=="metric":
            metric.append(section[i].replace("<punct>",''))
            while i<len(section) and predicted_tags[i+1]=="metric":
                i+=1
                metric.append(section[i].replace("<punct>",''))
            metric=" ".join(metric)
            metrics.append(metric)
        i+=1

    return tasks,datasets,metrics


def extract_TDM_triples2(section,predicted_tags):
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

    tasks=[]
    datasets=[]
    metrics=[]

    for x0, y0, y1 in predict(args.test, *load_model()):

        #print(x0.split(' '))
        #print(y0)
        #print(y1)
        #print('\n')
        sec_tasks,sec_datasets,sec_metrics=extract_TDM_triples2(x0.split(' '), y0)
        tasks+=sec_tasks
        datasets+=sec_datasets
        metrics+=sec_metrics

    tasks=list(set(tasks))
    datasets=list(set(datasets))
    metrics=list(set(metrics))

    print(">>>>task candidates ",tasks)
    print(">>>>datasets candidates ",datasets)
    print(">>>>metrics candidates",metrics)

    #print(paper_id)

    candidates=[tasks,datasets,metrics]
    TDM_candidates=list(itertools.product(*candidates))
    print(">>>> TDM candidates ")
    for cand in TDM_candidates:
        print(cand)
    only_one_candidate=False
    if len(TDM_candidates)==1:
        only_one_candidate=True
        TDM_candidates.append(TDM_candidates[0])


    TDM_candidates=[[' '.join(candidate),candidate] for candidate in TDM_candidates]

    test_dataset = DataAndQuery(wv, word_to_index,
                                index_to_word, paper_id, TDM_candidates)
    print(len(test_dataset))
    batch_size=32
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    args.index_to_word = test_dataset.index_to_word
    args.wv = test_dataset.wv

    model = CONVKNRM(args).to(args.device)
    save_path = args.phase2_model

    model= load_checkpoint_for_eval(model,  save_path)

    m = nn.Sigmoid()

    all_outputs=[]
    all_labels=[]
    all_triples=[]

    for batch_desc,batch_query, labels,triples in test_iter:
        batch_desc, batch_query = batch_desc.to(
            args.device), batch_query.to(args.device)

        batch_query = torch.squeeze(batch_query, 1)
        batch_query = torch.squeeze(batch_query, 1)
        batch_desc = torch.squeeze(batch_desc, 1)
        batch_desc = torch.squeeze(batch_desc, 1)


        outputs = model(batch_query, batch_desc).to(args.device)
        outputs=m(outputs)
        all_outputs += outputs.tolist()
        all_labels += labels
        all_triples+=triples

    predicted_triples=[[all_triples[i],all_outputs[i]] for i in range(len(all_triples)) if all_outputs[i]>0.1]
    predicted_triples.sort(key=lambda x: x[1],reverse=True)
    #formated_for_next_stage='$'.join([label[0] for label in predicted_triples])
    if only_one_candidate and len(predicted_triples)>0:
        predicted_triples=[predicted_triples[0]]
    print('>>>>Extracted TDS ')
    for trip in predicted_triples:
        print(trip)
    print(len(predicted_triples))

