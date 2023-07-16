"""
CARENET
@author: aaq109
"""

import pickle
from tqdm import tqdm
import json
import numpy as np
from collections import Counter
import torch
import gensim
import itertools
import logging
from sklearn.utils.class_weight import compute_class_weight
from data_extractor import DataReaderRHIP

def flatten(l):
    return [item for sublist in l for item in sublist]
    
    
class DataProcessorRHIP:

    def __init__(self, args, fold, pretrain=True):
        
        self.word_map = dict()
        self.id2label = dict()
        self.min_count = args.min_count
        self.n_classes = args.n_classes
        self.out_folder = args.fname_output
        self.fold=fold
        self.data_raw = DataReaderRHIP(self.out_folder + '/' + args.fname)
        self.hdf_len = len(self.data_raw.data_train[0]['hdf'][0])
        self.onsite_len = len(self.data_raw.data_train[0]['onsite'])
        y = [i['catgy'][0] for i in self.data_raw.data_train]
        self.weights = compute_class_weight('balanced', classes=np.array([i for i in range(args.n_classes)]), y=y)
 
        self.prep_Data(self.data_raw.data_train, 'train', args.sentence_limit, args.word_limit,w2v=True)
        self.prep_Data(self.data_raw.data_val, 'val', args.sentence_limit, args.word_limit,w2v=False)
        self.prep_Data(self.data_raw.data_test, 'test', args.sentence_limit, args.word_limit,w2v=False)
        if pretrain:
            self.train_word2vec_model(args.emb_size, algorithm="skipgram")

    def prep_Data(self, data, tp, sentence_limit, word_limit, w2v):
    # Read training data
        print('\nReading and preprocessing {0} data...\n'.format(tp))
  
        docs = []
        labels = []
        patients = []
        hdf = []
        onsite = []
        word_counter = Counter()
    
        for i in tqdm(data):
            paragraphs = i['text']
            para = list()
            for sentences in paragraphs:
                words = list()
                for s in sentences:
                    w = s[-word_limit:]
                    if len(w) == 0:
                        continue
                    words.append(w)
                    word_counter.update(w)
                para.append(words)

            docs.append(para)
            
            x = torch.nn.functional.one_hot(torch.tensor(i['Label_multi']), self.n_classes).sum(0).numpy().tolist()        
            labels.append(x)
            patients.append(i['Id'])
            hdf.append(i['hdf'])
            onsite.append(i['onsite'])
            #labels.append(i['labels'])

    # Save text data for word2vec
        if w2v:
            temp = [k for i in docs for k in i]
            temp = [[flatten(i)] for i in temp]
            torch.save(temp, self.out_folder + '/word2vec_data.pth.tar')
            print('\nText data for word2vec saved')

    # Create word map
            self.word_map['<pad>'] = 0
            for word, count in word_counter.items():
                if count >= self.min_count:
                    self.word_map[word] = len(self.word_map)
            self.word_map['<unk>'] = len(self.word_map)
            print('\nDiscarding words with counts less than %d, the size of the vocabulary is %d.\n' % (
                self.min_count, len(self.word_map)))

            with open(self.out_folder+'/Fold{0}'.format(self.fold)+'/word_map.json', 'w') as j:
                json.dump(self.word_map, j)
            print('Word map saved')
            self.word_counter = word_counter

    # Encode and pad
        print('Encoding and padding {0} data...\n'.format(tp))
        encoded_train_docs = list(map(lambda doc: list(map(lambda p: list(
            map(lambda s: list(map(lambda w: self.word_map.get(w, self.word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
                p)), doc)), docs))
        par_per_train_document = list(map(lambda doc: len(doc), docs))
        sentences_per_train_para = list(
            map(lambda doc: list(map(lambda p: len(p), doc)), docs))
        words_per_train_sentence = list(
            map(lambda doc: list(map(lambda p: list(map(lambda s: len(s), p)), doc)), docs))
        temp = [hdf[i][-j:] for i,j in enumerate(par_per_train_document)]
        encoded_train_hdf = list(
            map(lambda doc: list(map(lambda s: s, doc)), temp))

    # Save
        print('Saving...\n')
        assert len(encoded_train_docs) == len(labels) == len(sentences_per_train_para) == len(
            words_per_train_sentence) == len(par_per_train_document)
    # Because of the large data, saving as a JSON can be very slow
        torch.save({'docs': encoded_train_docs,
                    'para_per_document': par_per_train_document,
                    'sentences_per_para': sentences_per_train_para,
                    'words_per_sentence': words_per_train_sentence,
                    'hdf': encoded_train_hdf,
                    'onsite': onsite,
                    'labels': labels,
                    'patients': patients},
                   self.out_folder+'/{0}_data.pth.tar'.format(tp))
        print('Encoded, padded {0} data'.format(tp))


    def train_word2vec_model(self, emb_size, algorithm='skipgram'):
        assert algorithm in ['skipgram', 'cbow']
        sg = 1 if algorithm is 'skipgram' else 0

        # Read data
        sentences = torch.load(self.out_folder+'/word2vec_data.pth.tar')
        sentences = list(itertools.chain.from_iterable(sentences))

        # Activate logging for verbose training
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # Initialize and train the model (this will take some time)
        model = gensim.models.word2vec.Word2Vec(sentences=sentences, vector_size=emb_size, workers=8, window=20, min_count=self.min_count,
                                            sg=sg)

        # Normalize vectors and save model
        model.init_sims(True)
        model.wv.save(self.out_folder+'/Fold{0}'.format(self.fold)+'/word2vec_model')



