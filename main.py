"""
CARENET
@author: aaq109
"""

import argparse
import torch
import pickle
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from trainers import CARENET_trainer

parser = argparse.ArgumentParser(description='CARENET arguments')

parser.add_argument('--fname', type=str, default='ehr_input',
                    help='Input pkl file name')
parser.add_argument('--fname_output', type=str, default='Outputs',
                    help='Output folder name')
parser.add_argument('--n_classes', type=int, default=4,
                    help='N classes')
parser.add_argument('--iterations', type=int, default=25,
                    help='Training iterations')
parser.add_argument('--edl', default=True,
                    help='Use EDL loss; else use BCE loss')
parser.add_argument('--batch_size', type=int, default=64,
                    help='No. of visits per batch')
parser.add_argument('--min_count', type=int, default=50,
                    help='Minimum code count')
parser.add_argument('--sentence_limit', type=int, default=6,
                    help='6 by default one sentence per care unit (IP,OP,PC,ED,AMB,Others)')
parser.add_argument('--word_limit', type=int, default=20,
                    help='Maximum number of codes/words allowed per sentence')
parser.add_argument('--emb_size', type=int, default=100,
                    help='Word embedding dimension')
parser.add_argument('--word_rnn_size', type=int, default=50,
                    help='Hidden layer size - word level')
parser.add_argument('--word_rnn_layers', type=int, default=1,
                    help='No. of hidden layers - word level')
parser.add_argument('--sentence_rnn_size', type=int, default=37,
                    help='Hidden layer size - sentence level')
parser.add_argument('--para_rnn_size', type=int, default=25,
                    help='Hidden layer size - paragraph level')
parser.add_argument('--sentence_rnn_layers', type=int, default=1,
                    help='No. of hidden layers -  sentence level')
parser.add_argument('--para_rnn_layers', type=int, default=1,
                    help='No. of hidden layers -  paragraph level')
parser.add_argument('--word_att_size', type=int, default=50,
                    help='Attention size - word level')
parser.add_argument('--sentence_att_size', type=int, default=37,
                    help='Attention size -  sentence level')
parser.add_argument('--para_att_size', type=int, default=25,
                    help='Attention size -  paragraph level')
parser.add_argument('--dropout', type=int, default=0.5,
                    help='Drop out level')

args = parser.parse_args(args=[])
args.tied = True

with open(args.fname + '.pkl', 'rb') as handle:
    ehr_input = pickle.load(handle)


skf = StratifiedKFold(n_splits=10)
X = [i['Id'] for i in ehr_input]
y = [i['catgy'][0] for i in ehr_input]
fold = 0
output = dict()


if not os.path.exists(args.fname_output):
    os.makedirs(args.fname_output)
    for i in range(10):
        os.makedirs(args.fname_output + '/Fold{0}'.format(i))
        os.makedirs(args.fname_output + '/Fold{0}/AttentionPlots'.format(i))
    os.makedirs(args.fname_output + '/Models')
    os.makedirs(args.fname_output + '/Checkpoints')

for train_val_index, test_index in skf.split(X, y):
    print('Now at Fold {}'.format(fold))
    train_index, val_index = train_test_split(train_val_index, test_size=0.2)
    X_train = [ehr_input[i] for i in train_index]
    X_val = [ehr_input[i] for i in val_index]
    X_test = [ehr_input[i] for i in test_index]
    with open(args.fname_output+'/ehr_input_train.pkl', 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.fname_output+'/ehr_input_val.pkl', 'wb') as handle:
        pickle.dump(X_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.fname_output+'/ehr_input_test.pkl', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dt = CARENET_trainer(args, fold)


    with open(args.fname_output+'/Fold{0}/dicts.pkl'.format(fold), 'wb') as handle:
        pickle.dump((dt.word_map, dt.data_prosd.word_counter), handle, protocol=pickle.HIGHEST_PROTOCOL)  

    dt.train_classifier(args)
    dt.test_classifier(fold)
    output[fold] = dt.output
    dt.plot_attn(fold)
    dt.model.cpu()
    torch.save(dt.model, args.fname_output + '/Models/model_classifier_{0}.pth'.format(fold))
    fold = fold + 1 
  

