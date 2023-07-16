"""
CARENET
@author: aaq109
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import gensim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle


def compute_auc(Y_pred_proba,y_test,n_classes,classes_key, fold,out_folder, lw=1):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Y_pred_proba[:, i],pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc_micro = roc_auc["micro"] 

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    
    print("******* Plot AUC roc ********")

    # Plot all ROC curves
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12,12))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
        )

    colors = cycle(["aqua", "darkorange", "cornflowerblue",'blue'])
    for i, color in zip(range(4), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(classes_key[i], roc_auc[i]),
            )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(out_folder + '/Fold{0}/ROC_curves.pdf'.format(fold),format = 'pdf', dpi = 500, bbox_inches='tight')
    plt.savefig(out_folder + '/Fold{0}/ROC_curves.png'.format(fold),format = 'png', dpi = 500, bbox_inches='tight')
    plt.close()

    print("auc_micro = ", roc_auc_micro)

    print("********************************")

    return (roc_auc_micro) # Acc, F1, ovr_auc_macro, ovo_auc_macro, ovr_auc_weighted, ovo_auc_weighted


def edl_loss(output, target,weights, device=None):
    fn = torch.digamma
    evidence = torch.exp(output) + 1
    a=target*weights
    a_split = torch.cat(torch.split(a,1,dim=1)) 
    evidence_split = torch.cat(torch.split(evidence,2,dim=1))
    target_split = torch.cat(torch.split(target,1,dim=1)) 
    target = 1 - F.one_hot(torch.flatten(target_split).long(), num_classes=2)
    S = torch.sum(evidence_split, dim=1, keepdim=True)
    A = torch.sum(target* (fn(S) - fn(evidence_split)), dim=1, keepdim=True)
    x = target_split + 0.1
    A = A*x
    #A = A*(a_split + 0.05)
    return A.sum()/output.shape[0]


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_word2vec_embeddings(word2vec_file, word_map):
    """
    Load pre-trained embeddings for words in the word map.

    :param word2vec_file: location of the trained word2vec model
    :param word_map: word map
    :return: embeddings for words in the word map, embedding size
    """
    # Load word2vec model into memory
    w2v = gensim.models.KeyedVectors.load(word2vec_file, mmap='r')

    print("\nEmbedding length is %d.\n" % w2v.vector_size)

    # Create tensor to hold embeddings for words that are in-corpus
    embeddings = torch.FloatTensor(len(word_map), w2v.vector_size)
    init_embedding(embeddings)

    # Read embedding file
    print("Loading embeddings...")
    for word in word_map:
        if word in w2v.key_to_index:
            embeddings[word_map[word]] = torch.FloatTensor(w2v[word])

    print("Done.\n Embedding vocabulary: %d.\n" % len(word_map))

    return embeddings, w2v.vector_size


def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class HANDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        #assert split in {'train', 'test'}
        self.split = split

            
        # Load data
        self.data = torch.load(split+'_data.pth.tar')


    def __getitem__(self, i):
        
        return torch.LongTensor(self.data['docs'][i]), \
                    torch.LongTensor([self.data['para_per_document'][i]]), \
                    torch.LongTensor([self.data['sentences_per_para'][i]]), \
                    torch.LongTensor(self.data['words_per_sentence'][i]), \
                    torch.FloatTensor([self.data['labels'][i]]), \
                    torch.FloatTensor([self.data['hdf'][i]]), \
                    torch.FloatTensor([self.data['onsite'][i]]), \
                    torch.LongTensor([self.data['patients'][i]])


    def __len__(self):
        return len(self.data['docs'])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



