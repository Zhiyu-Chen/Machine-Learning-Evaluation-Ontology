from model import *
from utils import *

import argparse
parser = argparse.ArgumentParser(description='LSTM-CRF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--CharToIdx", type=str, default='data/train.char_to_idx',
                    help="char_to_idx")
parser.add_argument("--WordToIdx", type=str, default='data/train.word_to_idx',
                    help="word_to_idx")
parser.add_argument("--TagToIdx", type=str, default='data/train.tag_to_idx',
                    help="tag_to_idx")

parser.add_argument("--test", type=str, default='data/test',
                     help="dataset to use")
parser.add_argument("--checkpoint", type=str, default='data/model.epoch100',
                     help="checkpoint")

args = parser.parse_args()

def load_model():
    cti = load_tkn_to_idx(args.CharToIdx) # char_to_idx
    wti = load_tkn_to_idx(args.WordToIdx) # word_to_idx
    itt = load_idx_to_tkn(args.TagToIdx) # idx_to_tag
    model = rnn_crf(len(cti), len(wti), len(itt))
    print(model)
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
    # with open(filename) as fo:
    #     text = fo.read().strip().split("\n" * (HRE + 1))
    lines=[]
    with open(filename) as fo:
        for line in fo:

            line=line.rstrip().replace("\\tag", "/").replace('\t',' ').split(' ',1)[1]
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

            # x1 = tokenize(res)
            # xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
            # xw = [wti[w] if w in wti else UNK_IDX for w in x1]
            # data.append_item(x0 = [res], xc = [xc], xw = [xw], y0 = y0)

            res=res.split(' ')
            seq_x=[]
            seq_y=[]
            for index,rel in enumerate(y0):
                if rel!='SEC_START':
                    seq_x.append(res[index])
                    seq_y.append(rel)
                else:
                    seq_x = ' '.join(seq_x)
                    x1 = tokenize(seq_x)
                    xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
                    xw = [wti[w] if w in wti else UNK_IDX for w in x1]
                    data.append_item(x0=[seq_x], xc=[xc], xw=[xw], y0=seq_y)
                    data.append_row()
                    seq_x=[res[index]]
                    seq_y=[rel]
            seq_x = ' '.join(seq_x)
            x1 = tokenize(seq_x)
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
            xw = [wti[w] if w in wti else UNK_IDX for w in x1]
            data.append_item(x0=[seq_x], xc=[xc], xw=[xw], y0=seq_y)
            data.append_row()



        #data.append_row()
    data.strip()
    with torch.no_grad():
        model.eval()
        return run_model(model, itt, data)

if __name__ == "__main__":

    for x0, y0, y1 in predict(args.test, *load_model()):
        if not TASK:
            #print((x0, y0, y1) if y0 else (x0, y1))
            print(x0.split(' '))
            print(y0)
            print(y1)
            print('\n')
        else: # word/sentence segmentation
            if y0:
                print(iob_to_txt(x0, y0))
            print(iob_to_txt(x0, y1))
            print()
