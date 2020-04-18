from predict import *

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

def evaluate(result, summary = False):
    avg = defaultdict(float) # average
    tp = defaultdict(int) # true positives
    tpfn = defaultdict(int) # true positives + false negatives
    tpfp = defaultdict(int) # true positives + false positives
    for _, y0, y1 in result: # actual value, prediction
        if HRE:
            tp[y0] += (y0 == y1)
            tpfn[y0] += 1
            tpfp[y1] += 1
            continue
        for y0, y1 in zip(y0, y1):
            tp[y0] += (y0 == y1)
            tpfn[y0] += 1
            tpfp[y1] += 1
    print()
    for y in sorted(tpfn.keys()):
        pr = (tp[y] / tpfp[y]) if tpfp[y] else 0
        rc = (tp[y] / tpfn[y]) if tpfn[y] else 0
        avg["macro_pr"] += pr
        avg["macro_rc"] += rc
        if not summary:
            print("label = %s" % y)
            print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
            print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
            print("f1 = %f\n" % f1(pr, rc))
    avg["macro_pr"] /= len(tpfn)
    avg["macro_rc"] /= len(tpfn)
    avg["micro_f1"] = sum(tp.values()) / sum(tpfn.values())
    print("macro precision = %f" % avg["macro_pr"])
    print("macro recall = %f" % avg["macro_rc"])
    print("macro f1 = %f" % f1(avg["macro_pr"], avg["macro_rc"]))
    print("micro f1 = %f" % avg["micro_f1"])

if __name__ == "__main__":

    evaluate(predict(args.test, *load_model()))
