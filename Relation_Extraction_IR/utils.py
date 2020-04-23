import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import tensorflow as tf
import os
import re


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def preprocess(input,type):

    if type=='attribute':
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')
        #w = input.replace('_', ' ')

        tokens = word_tokenize(w)

        camel_tokens=[]

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens=camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter
        # remove punctuation from each word
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in inter_words2 if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # final_words = []
        # for w in words:
        #     inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words += inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok)]

    elif type=='value':
        #w = input.replace('_', ' ').replace(',', ' ')
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')
        #w=input

        tokens = word_tokenize(w)

        camel_tokens = []

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens = camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter

        # remove punctuation from each word
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic

        numerical_values=[]
        string_values=[]
        for word in inter_words2:
            try:
                float(word)
                numerical_values.append(word)

            except ValueError:
                string_values.append(word)


        string_values_final=[]
        for w in string_values:
            inter=re.split(r'(\d+)', w)

            for word in inter:
                if len(word)>0:
                    try:
                        float(word)
                        numerical_values.append(word)

                    except ValueError:
                        string_values_final.append(word)

        #keep 0 digits
        #numerical_values = [re.sub('\d', '#', s) for s in numerical_values]

        #keep 1 digit
        numerical_values_inter=[]
        for s in numerical_values:
            if s[0]=='-':
                ss=s[2::]
                ss=re.sub('\d', '#', ss)
                ss=s[0:2]+ss


            else:
                ss = s[1::]
                ss = re.sub('\d', '#', ss)
                ss = s[0] + ss

            numerical_values_inter += [ss]

        #keep 2 digits

        # for s in numerical_values:
        #     ss=s[2::]
        #     ss=re.sub('\d', '#', ss)
        #     ss=s[0:2]+ss
        #     numerical_values_inter+=[ss]

        numerical_values=numerical_values_inter
        inter_words2 = string_values_final

        words = [word for word in inter_words2 if word.isalpha() or word in['$','@','%','£','€','°']]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        stop_words.remove('d')
        stop_words.remove('m')
        stop_words.remove('s')

        words = [w for w in words if not w in stop_words]

        # final_words = []
        # for w in words:
        #     inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words += inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok) or tok in['$','@','%','£','€','°']]

        final_words=final_words+numerical_values



    elif type=='value2':

        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')

        tokens = word_tokenize(w)

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter


        numerical_values=[]
        string_values=[]
        for word in inter_words2:
            try:
                float(word)
                numerical_values.append(word)

            except ValueError:
                string_values.append(word)


        string_values_final=[]
        for w in string_values:
            inter=re.split(r'(\d+)', w)

            for word in inter:
                if len(word)>0:
                    try:
                        float(word)
                        numerical_values.append(word)

                    except ValueError:
                        string_values_final.append(word)



        inter_words2 = string_values_final

        words = [word for word in inter_words2 if word.isalpha() or word in['$','@','%','£','€','°']]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        stop_words.remove('d')
        stop_words.remove('m')
        stop_words.remove('s')

        words = [w for w in words if not w in stop_words]

        final_words = []
        for w in words:
            inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
            final_words += inter

        final_words = [tok for tok in final_words if isEnglish(tok) or tok in['$','@','%','£','€','°']]

        final_words=final_words+numerical_values






    elif type == 'description':
        #w = input.replace('_', ' ').replace(',', ' ').replace('-', " ").replace('.', ' ')
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')


        tokens = word_tokenize(w)

        camel_tokens = []

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens = camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter
        # remove punctuation from each word
        #table = str.maketrans('', '', string.punctuation)
        #stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in inter_words2 if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # final_words=[]
        # for w in words:
        #     inter=re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words+=inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok)]

        #not_to_use=['com','u','comma','separated','values','csv','data','dataset','https','api','www','http','non','gov','rows','p','download','downloads','file','files','p']

        #final_words=[tok for tok in final_words if tok not in not_to_use]

    return final_words



def find_chunk_and_sample(dir,batch_size,nb_steps_per_epoch):
    CHECKPOINT_DIR = dir + '/skip_gram_model_with_context'
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    total_size=0

    with tf.Session() as sess:


        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)
        # aa=[n.name for n in tf.get_default_graph().as_graph_def().node]
        bb = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

        glob_step = bb[4]
        glob_step = sess.run(glob_step)

        nb_steps_last_epoch =glob_step%nb_steps_per_epoch

        nb_samples_processed=int(nb_steps_last_epoch*batch_size)



        input_folder = dir + '/X'

        nb_chunks = len(os.listdir(input_folder))

        for chunk in range(nb_chunks):
            train_x_file = 'x_train_' + str(chunk + 1) + '.npy'

            train_x_file = os.path.join(input_folder, train_x_file)

            x_train = np.load(train_x_file)
            total_size+=len(x_train)

            if total_size>nb_samples_processed:
                break


        current_chunk=chunk+1

        current_sample=len(x_train)-(total_size-nb_samples_processed)

    return current_chunk,current_sample


def steps_per_epoch(dir,batch_size):
    total_size=0
    input_folder = dir + '/X'

    nb_chunks = len(os.listdir(input_folder))

    for chunk in range(nb_chunks):
        train_x_file = 'x_train_' + str(chunk + 1) + '.npy'

        train_x_file = os.path.join(input_folder, train_x_file)

        x_train = np.load(train_x_file)
        total_size += len(x_train)


    nb_steps_per_epoch=total_size//batch_size

    rest_of_samples=total_size%batch_size

    if rest_of_samples>0:
        nb_steps_per_epoch+=1

    return nb_steps_per_epoch



def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0