'''
1. construct the following relation pairs from each paper:
    task-dataset (TD)
    dataset-metric (DM)
2. from those pairs, build pos/neg pairs with gold standard
3. build and train models
4. evaluation
'''

import pickle
from metadata import *
from reader import PaperReader,KBReader
import random
import torch
import numpy as np
import pandas as pd
import matchzoo as mz
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import classification_report

def set_seed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)

def get_paired_data(pids,context_sent_num = 1):
    #return TD pairs, DM pairs
    #e.g. [pid,T_sentences, D_sentences,t_label,d_label, rel_label] -> p,t,d
    td_pairs = []
    dm_pairs = []
    acc_TD_label = 0
    acc_DM_label = 0
    total_pos = preader.get_num_pos()
    for pid in pids:
        num_neg = 0
        sorted_tasks, sorted_datasets, sorted_metrics = paper_tags[pid]['task'], paper_tags[pid]['dataset'], \
                                                        paper_tags[pid]['metric']
        g_tasks, g_datasets, g_metrics = paper_tags[pid]['g_task'], paper_tags[pid]['g_dataset'], paper_tags[pid][
            'g_metric']
        results = paper_result[pid]
        results = results.split('$')
        gold_results = [result.split('#') for result in results]
        #TD pairs
        for t_index in range(len(sorted_tasks)):
            for d_index in range(len(sorted_datasets)):
                T_sentences = ' '.join([each[1] for each in paper_task[pid][:context_sent_num]])
                D_sentences = ' '.join([each[1] for each in paper_dataset[pid][:context_sent_num]])
                t_label = sorted_tasks[t_index]
                d_label = sorted_datasets[d_index]
                #print(t_label,d_label,gold_results)
                # check the relation
                is_true = 0
                for result in gold_results:
                    if t_label == result[0].lower() and d_label == result[1].lower():
                        is_true = 1
                        acc_TD_label += 1
                if is_true == 1 :
                    td_pairs.append([pid,T_sentences,D_sentences,t_label,d_label,is_true])
                else:
                    num_neg += 1
                    if num_neg < total_pos:
                        td_pairs.append([pid, T_sentences, D_sentences, t_label, d_label, is_true])
        #DM pairs
        num_neg = 0
        for d_index in range(len(sorted_datasets)):
            for m_index in range(len(sorted_metrics)):
                D_sentences = ' '.join([each[1] for each in paper_dataset[pid]])
                M_sentences = ' '.join([each[1] for each in paper_metric[pid]])
                d_label = sorted_datasets[d_index]
                m_label = sorted_metrics[m_index]
                # check the relation
                is_true = 0
                for result in gold_results:
                    if m_label == result[2].lower() and d_label == result[1].lower():
                        is_true = 1
                        acc_DM_label += 1
                if is_true == 1 :
                    dm_pairs.append([pid, D_sentences, M_sentences, d_label, m_label, is_true])
                else:
                    num_neg += 1
                    if num_neg < total_pos:
                        dm_pairs.append([pid, D_sentences, M_sentences, d_label, m_label, is_true])
    #print(acc_TD_label,acc_DM_label)
    return td_pairs,dm_pairs



def pairs_to_datapack(pairs,task='classification'):
    #text id: pid + entity label
    text_left = []
    text_right = []
    id_left = []
    id_right = []
    label = []
    for pair in pairs:
        text_left.append(pair[1])
        id_left.append(pair[0]+'_'+pair[3])
        text_right.append(pair[2])
        id_right.append(pair[0] + '_' + pair[4])
        label.append(pair[5])
    df = pd.DataFrame({
        'text_left': text_left,
        'text_right': text_right,
        'id_left': id_left,
        'id_right': id_right,
        'label': label
    })
    return mz.pack(df, task)

def resample_pairs(pairs,ratio=2,sample_type='down'):
    #balance pos/neg pairs
    #check pair stats
    pos_idx = []
    neg_idx = []
    for idx in range(len(pairs)):
        if pairs[idx][-1] == 1:
            pos_idx.append(idx)
        else:
            neg_idx.append(idx)
    if sample_type == 'down':
        neg_idx = random.sample(neg_idx,int(ratio*len(pos_idx)))
    elif sample_type == 'up':
        pos_idx = pos_idx*ratio
    pos_idx.extend(neg_idx)
    random.shuffle(pos_idx)
    return [pairs[each] for each in pos_idx]

set_seed(666)
relation_type = "DM" # TD or DM
task_type = 'classification'
sample_type = 'up'
neg_ratio = 20 # number of negative samples divided by number of positive samples
context_sents = 1
model_type = "BERT" # ESIM or BERT
overwrite_pack = True
train_ratio = 0.8
num_epochs = 3
paper_title,paper_task,paper_dataset,paper_metric,paper_scores,paper_entities = pickle.load(open(paper_entity_path,'rb'))
paper_tags = pickle.load(open(top_tag_path,'rb'))
preader = PaperReader()
paper_result = preader.get_paper_result()


all_pids = list(paper_title.keys())


# random.shuffle(all_pids)
# train_pids = all_pids[:int(len(all_pids)*train_ratio)]
# test_pids = all_pids[int(len(all_pids)*train_ratio):]
#
# td_train,dm_train = get_paired_data(train_pids,context_sent_num=context_sents)
# td_train = resample_pairs(td_train,ratio=neg_ratio)
# dm_train = resample_pairs(dm_train,ratio=neg_ratio)
# td_test,dm_test = get_paired_data(test_pids,context_sent_num=context_sents)
#


td_pairs,dm_pairs = get_paired_data(all_pids,context_sent_num=context_sents)


random.shuffle(td_pairs)
random.shuffle(dm_pairs)

td_train = td_pairs[:int(len(td_pairs)*train_ratio)]
td_test = td_pairs[int(len(td_pairs)*train_ratio):]
td_train = resample_pairs(td_train,ratio=neg_ratio,sample_type=sample_type)


dm_train = dm_pairs[:int(len(dm_pairs)*train_ratio)]

dm_test = dm_pairs[int(len(dm_pairs)*train_ratio):]
dm_train = resample_pairs(dm_train,ratio=neg_ratio,sample_type=sample_type)
task_data = {"TD" : [td_train,td_test],
             "DM" : [dm_train,dm_test]
             }
# for rel_type in  task_data:

train_pairs,test_pairs = task_data[relation_type]
if task_type == 'ranking':
    sci_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss(num_neg=5))
    sci_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.MeanAveragePrecision()
    ]
else:
    sci_task = mz.tasks.Classification(num_classes=2)
    sci_task.metrics = ['acc']

print('Task-Dataset classification')
print('data loading ...')
train_pack_raw = pairs_to_datapack(train_pairs)
dev_pack_raw = pairs_to_datapack(test_pairs)

if model_type == 'ESIM':
    preprocessor = mz.models.ESIM.get_default_preprocessor( filter_low_freq=3,
                                                            truncated_length_left=400,
                                                            truncated_length_right=400)
    train_pack_processed = preprocessor.fit_transform(train_pack_raw)
    dev_pack_processed = preprocessor.transform(dev_pack_raw)
    print(preprocessor.context)

    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = glove_embedding.build_matrix(term_index)
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

    if task_type == 'ranking':
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='pair',
            num_dup=1,
            num_neg=5,
            sort=False,
            resample=True,
            batch_size=20
        )
        devset = mz.dataloader.Dataset(
            data_pack=dev_pack_processed,
            batch_size=20,
            sort=False,
            shuffle=False
        )
    else:
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            batch_size=3,
            mode='point'
        )
        devset = mz.dataloader.Dataset(
            data_pack=dev_pack_processed,
            batch_size=3,
            mode='point'
        )

    padding_callback = mz.models.ESIM.get_default_padding_callback()


    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        stage='train',
        callback=padding_callback
    )
    devloader = mz.dataloader.DataLoader(
        dataset=devset,
        stage='dev',
        callback=padding_callback
    )

    model = mz.models.ESIM()
    model.params['task'] = sci_task
    model.params['embedding'] = embedding_matrix
    model.params['mask_value'] = 0
    model.params['dropout'] = 0.2
    model.params['hidden_size'] = 200
    model.params['lstm_layer'] = 1

    model.build()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=devloader,
        validate_interval=None,
        epochs=20
    )
    trainer.run()

    # Evaluation / Prediction
    y_preds = trainer.predict(devloader)
    y_preds = np.argmax(y_preds, axis=1)
    y_true = devloader.label
    print(classification_report(y_true, y_preds, labels=[0, 1]))

    y_preds = trainer.predict(trainloader)
    y_preds = np.argmax(y_preds, axis=1)
    y_true = trainloader.label
    print(classification_report(y_true, y_preds, labels=[0, 1]))

if model_type == "BERT":
    if not os.path.exists(BERT_TD_path) or overwrite_pack:
        preprocessor = mz.models.Bert.get_default_preprocessor()
        train_pack_processed = preprocessor.transform(train_pack_raw)
        dev_pack_processed = preprocessor.transform(dev_pack_raw)
        pickle.dump([train_pack_processed,dev_pack_processed],open(BERT_TD_path,'wb'))
    else:
        train_pack_processed,dev_pack_processed = pickle.load(open(BERT_TD_path,'rb'))

    if task_type == 'ranking':
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='pair',
            num_dup=1,
            num_neg=5,
            sort=False,
            resample=True,
            batch_size=20
        )
        devset = mz.dataloader.Dataset(
            data_pack=dev_pack_processed,
            batch_size=20,
            sort=False,
            shuffle=False
        )
    else:
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            batch_size=20,
            mode='point'
        )

        devset = mz.dataloader.Dataset(
            data_pack=dev_pack_processed,
            batch_size=20,
            mode='point'
        )

    padding_callback = mz.models.Bert.get_default_padding_callback(fixed_length_left=100,fixed_length_right=100)
    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        stage='train',
        callback=padding_callback
    )
    testloader = mz.dataloader.DataLoader(
        dataset=devset,
        stage='dev',
        callback=padding_callback
    )

    model = mz.models.Bert()

    model.params['task'] = sci_task
    model.params['mode'] = 'bert-base-uncased'
    model.params['dropout_rate'] = 0.2
    model.build()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, betas=(0.9, 0.98), eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=6, t_total=-1)

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=trainloader,
        validloader=testloader,
        validate_interval=None,
        epochs=num_epochs
    )

    trainer.run()

    #Evaluation / Prediction
    print("=========Training Results=========")
    y_preds = trainer.predict(trainloader)
    y_preds = np.argmax(y_preds, axis=1)
    y_true = trainloader.label
    print(classification_report(y_true, y_preds, labels=[0, 1]))

    print("=========Testing Results=========")
    y_preds = trainer.predict(testloader)
    y_preds = np.argmax(y_preds, axis=1)
    y_true = testloader.label
    print(classification_report(y_true, y_preds, labels=[0, 1]))



