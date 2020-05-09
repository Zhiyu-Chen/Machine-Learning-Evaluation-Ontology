from model import *
from utils import *
from evaluate import *
import argparse
parser = argparse.ArgumentParser(description='LSTM-CRF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data", type=str, default='data/train.csv',
                    help="dataset to use")
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument("--CharToIdx", type=str, default='data/train.char_to_idx',
                    help="char_to_idx")
parser.add_argument("--WordToIdx", type=str, default='data/train.word_to_idx',
                    help="word_to_idx")
parser.add_argument("--TagToIdx", type=str, default='data/train.tag_to_idx',
                    help="tag_to_idx")
parser.add_argument("--model_file", type=str, default='data/phase1_model',
                     help="dataset to use")
parser.add_argument("--valid", type=str, default='data/valid',
                     help="dataset to use")

args = parser.parse_args()

def load_data(args):
    data = dataloader()
    batch = []
    cti = load_tkn_to_idx(args.CharToIdx) # char_to_idx
    wti = load_tkn_to_idx(args.WordToIdx) # word_to_idx
    itt = load_idx_to_tkn(args.TagToIdx) # idx_to_tkn
    print("loading %s..." % args.data)
    with open(args.data, "r") as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))


    for block in text:
        for line in block.split("\n"):
            x, y = line.split("\t")
            x = [x.split(":") for x in x.split(" ")]
            y = [int(y)] if HRE else [int(x) for x in y.split(" ")]
            for xc,xw in x:
                if len(xc.split("+"))==1:
                    print('here')
            xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
            data.append_item(xc = [xc], xw = [xw], y0 = y)
        data.append_row()
    data.strip()
    for _batch in data.split():
        xc, xw = data.tensor(_batch.xc, _batch.xw, _batch.lens)
        _, y0 = data.tensor(None, _batch.y0, sos = True)
        batch.append((xc, xw, y0))
    print("data size: %d" % len(data.y0))
    print("batch size: %d" % BATCH_SIZE)
    return batch, cti, wti, itt

def train(args):
    num_epochs = int(args.epochs)
    batch, cti, wti, itt = load_data(args)
    model = rnn_crf(len(cti), len(wti), len(itt))
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint(args.model_file, model) if isfile(args.model_file) else 0
    filename = re.sub("\.epoch[0-9]+$", "", args.model_file)
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time()
        for xc, xw, y0 in batch:
            loss = model(xc, xw, y0) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(batch)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            argss = [model, cti, wti, itt]
            evaluate(predict(args.valid, *argss), True)
            model.train()
            print()

train(args)
