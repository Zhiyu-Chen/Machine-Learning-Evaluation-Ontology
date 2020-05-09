from utils import *
import argparse
parser = argparse.ArgumentParser(description='LSTM-CRF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data", type=str, default='data/train',
                    help="dataset to use")
args = parser.parse_args()
def load_data():
    data = []
    max_length=0
    if KEEP_IDX:
        cti = load_tkn_to_idx(args.data + ".char_to_idx")
        wti = load_tkn_to_idx(args.data + ".word_to_idx")
        tti = load_tkn_to_idx(args.data + ".tag_to_idx")
    else:
        cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}

    fo = open(args.data)
    if HRE:
        tmp = []
        txt = fo.read().strip().split("\n\n")
        for doc in txt:
            data.append([])
            for line in doc.split("\n"):
                x, y = load_line(line, cti, wti, tti)
                data[-1].append((x, y))
        for doc in sorted(data, key = lambda x: -len(x)):
            tmp.extend(doc)
            tmp.append(None)
        data = tmp[:-1]
    else:
        for line in fo:
            line = line.strip()
            x, y = load_line(line, cti, wti, tti)
            seq_x=[]
            seq_y=[]

            for index,rel in enumerate(y):
                seq_x.append(x[index])
                seq_y.append(rel)
                if rel==str(tti['SENT_END']):
                    if len(seq_x)>max_length:
                        max_length=len(seq_x)
                    data.append((seq_x, seq_y))
                    seq_x=[]
                    seq_y=[]
            if len(seq_x)>0:
                if len(seq_x)>max_length:
                    max_length=len(seq_x)
                data.append((seq_x, seq_y))

        data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    fo.close()
    print(max_length)
    return data, cti, wti, tti

def load_line(line, cti, wti, tti):
    x, y = [], []
    if HRE:
        line, y = line.split("\t")
        if y not in tti:
            tti[y] = len(tti)
        y = [str(tti[y])]
    for num,w in enumerate(line.split("\t")):
        if num==0:
            continue
        w=w.replace("\\tag", "/")
        w, tag = (w, None) if HRE else re.split("/(?=[^/]+$)", w)

        w0 = normalize(w) # for character embedding
        if len(w0)==1:
            w0+='<punct>'
        if len(w0)==0:
            w0='<punct><punct>'
        w1 = w0.lower() # for word embedding

        if not KEEP_IDX:
            for c in w0:
                if c not in cti:
                    cti[c] = len(cti)
            if w1 not in wti:
                wti[w1] = len(wti)
            if tag and tag not in tti:
                tti[tag] = len(tti)
        x.append("+".join(str(cti[c]) for c in w0) + ":%d" % wti[w1])
        if tag:
            y.append(str(tti[tag]))
    return x, y

if __name__ == "__main__":

    # with open(args.data) as f:
    #     for line in f:
    #         line=line.rstrip()


    data, cti, wti, tti = load_data()
    save_data(args.data + ".csv", data)
    if not KEEP_IDX:
        save_tkn_to_idx(args.data + ".char_to_idx", cti)
        save_tkn_to_idx(args.data + ".word_to_idx", wti)
        save_tkn_to_idx(args.data + ".tag_to_idx", tti)
