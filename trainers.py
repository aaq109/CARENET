"""
CARENET
@author: aaq109
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import json
import pickle
from matplotlib import pyplot as plt
from data_processor import DataProcessorRHIP
from model import HierarchialAttentionNetwork
from utils import EarlyStopping, compute_auc, HANDataset, load_word2vec_embeddings, edl_loss, adjust_learning_rate
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class CARENET_trainer:

    def __init__(self, args, fold):

        self.edl = args.edl
        self.out_folder = args.fname_output
        self.n_classes = args.n_classes
        self.classes_key = {0:'NA',1:'KOL',2:'AHS',3:'Pneumonia'}
        self.data_prosd = DataProcessorRHIP(args, fold)
        
        with open(self.out_folder+'/Fold{0}'.format(fold)+'/word_map.json', 'r') as j: 
            self.word_map = json.load(j)
        
        self.id2label = {j:i for i,j in self.word_map.items()}
        #self.device = "cpu"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.train_loader = torch.utils.data.DataLoader(HANDataset(self.out_folder+'/train'), batch_size=args.batch_size, shuffle=True,
                                              num_workers=1, pin_memory=True) 
        self.val_loader = torch.utils.data.DataLoader(HANDataset(self.out_folder+'/val'), batch_size=100, shuffle=True,
                                              num_workers=1, pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(HANDataset(self.out_folder+'/test'), batch_size=100, shuffle=True,
                                              num_workers=1, pin_memory=True) 
         
        self.model = HierarchialAttentionNetwork(n_classes=self.n_classes,
                                        vocab_size=len(self.word_map),
                                        emb_size=args.emb_size,
                                        word_rnn_size=args.word_rnn_size,
                                        sentence_rnn_size=args.sentence_rnn_size,
                                        para_rnn_size=args.para_rnn_size,
                                        word_rnn_layers=args.word_rnn_layers,
                                        sentence_rnn_layers=args.sentence_rnn_layers,
                                        para_rnn_layers=args.para_rnn_layers,
                                        word_att_size=args.word_att_size,
                                        sentence_att_size=args.sentence_att_size,
                                        para_att_size=args.para_att_size,
                                        edl=self.edl,
                                        hdf_len=self.data_prosd.hdf_len,
                                        onsite_len=self.data_prosd.onsite_len,
                                        dropout=args.dropout
                                        )
        embeddings, emb_size = load_word2vec_embeddings(self.out_folder+'/Fold{0}'.format(fold)+'/word2vec_model', self.word_map)
        self.model.para_attention.sentence_attention.word_attention.init_embeddings(embeddings) 
        #self.model.para_attention.sentence_attention.word_attention.fine_tune_embeddings(fine_tune=True)
        
    def train_classifier(self, args):
        optimizer = optim.Adam(self.model.parameters(),lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.data_prosd.weights).to(self.device)) #weight=torch.LongTensor(self.weights).to(self.device)
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(self.weights).to(self.device))
        self.model.to(self.device)
        self.weights = torch.tensor(self.data_prosd.weights).to(self.device)
        early_stopping = EarlyStopping(self.out_folder + '/Checkpoints/checkpoint.pt',patience=5, verbose=True)
        
        for iteration in range(1,args.iterations):
            losses = []
            self.model.train()
            for i, (documents, para_per_document, sentences_per_para, words_per_sentence, labels, hdf,onsite, patients) in enumerate(self.train_loader):
                #i, (documents, para_per_document, sentences_per_para, words_per_sentence, labels, hdf, onsite, patients) = next(iter(enumerate(self.train_loader)))
                documents = documents.to(self.device)  # (batch_size, sentence_limit, word_limit)
                para_per_document = para_per_document.to(self.device)
                sentences_per_para = sentences_per_para.squeeze(1).to(self.device)  # (batch_size)
                words_per_sentence = words_per_sentence.to(self.device)  # (batch_size, sentence_limit)
                #labels = torch.argmax(labels.squeeze(1), 1).to(self.device)  # (batch_size)
                hdf = hdf.squeeze().to(self.device)  # (batch_size, sentence_limit, 6)  
                onsite = onsite.squeeze(1).to(self.device) 
                X = (documents, para_per_document, sentences_per_para,words_per_sentence, hdf, onsite)
                scores, word_alphas, sentence_alphas, para_alphas, document_embeddings = self.model(X)  
                optimizer.zero_grad()
                if self.edl:
                    labels = labels.squeeze(1).to(self.device)  # (batch_size)
                    loss = edl_loss(scores, labels,self.weights, device=self.device)
                else:
                    labels = torch.argmax(labels.squeeze(1), 1).to(self.device)
                    loss = self.criterion(scores, labels)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            print('Train loss:', np.mean(losses), 'Iteration',iteration, ' of ', args.iterations)
            adjust_learning_rate(optimizer, 0.99)
            self.validate_model()

            early_stopping(self.valid_loss, self.model)
    
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
            # load the last checkpoint with the best model
            self.model.load_state_dict(torch.load(self.out_folder + '/Checkpoints/checkpoint.pt'))
    
    def validate_model(self):
        self.model.eval()

        losses = []

        #all_scores, all_labels = [], []
        for i, (documents, para_per_document, sentences_per_para, words_per_sentence, labels, hdf, onsite, patients) in enumerate(self.val_loader):
            documents = documents.to(self.device)  # (batch_size, sentence_limit, word_limit)
            para_per_document = para_per_document.to(self.device)
            sentences_per_para = sentences_per_para.squeeze(1).to(self.device)  # (batch_size)
        #words_per_sentence = words_per_sentence.to(self.device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(self.device)  # (batch_size)
            hdf = hdf.squeeze().to(self.device)  # (batch_size, sentence_limit, 6)  
            onsite = onsite.squeeze(1).to(self.device) 
            X = (documents, para_per_document, sentences_per_para,words_per_sentence, hdf, onsite)
            scores, word_alphas, sentence_alphas, para_alphas, document_embeddings = self.model(X) 
            if self.edl:
                labels = labels.squeeze(1).to(self.device)  # (batch_size)
                loss = edl_loss(scores, labels,self.weights, device=self.device)
            else:
                labels = torch.argmax(labels.squeeze(1), 1).to(self.device)
                loss = self.criterion(scores, labels)
            losses.append(loss.item())
        self.valid_loss = np.mean(losses)
        print('Validation loss = ', self.valid_loss)
            

    def test_classifier(self, fold):
        m = nn.Softmax(dim=1) #nn.Sigmoid()
        
        self.model.eval()
        all_scores, all_labels, all_uncertainties, all_patients, all_data_patient = [], [], [], [], []
        
        self.model.to(self.device)

        for i, (documents, para_per_document, sentences_per_para, words_per_sentence, labels, hdf, onsite, patients) in enumerate(self.test_loader):
                #i, (documents, para_per_document, sentences_per_para, words_per_sentence, labels, hdf,onsite, patients) = next(iter(enumerate(self.test_loader)))
            documents = documents.to(self.device)  # (batch_size, sentence_limit, word_limit)
            para_per_document = para_per_document.to(self.device)
            sentences_per_para = sentences_per_para.squeeze(1).to(self.device)  # (batch_size)
            #words_per_sentence = words_per_sentence.to(self.device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(self.device)  # (batch_size)
            hdf = hdf.squeeze().to(self.device)  # (batch_size, sentence_limit, 6)  
            onsite = onsite.squeeze(1).to(self.device) 
            X = (documents, para_per_document, sentences_per_para,words_per_sentence, hdf, onsite)
            scores, word_alphas, sentence_alphas, para_alphas, document_embeddings = self.model(X)  
            data_patient = []
            for ii,jj in enumerate(patients):
                data_patient.append(words_per_sentence[ii].sum().item())
            words_per_sentence = words_per_sentence.to(self.device)  # (batch_size, sentence_limit)

            if self.edl:
                
                alpha = torch.exp(scores) + 1
                n = torch.arange(0,alpha.shape[1],2)
                m = torch.arange(1,alpha.shape[1],2)            
                scores = alpha[:,n] / (alpha[:,n] + alpha[:,m])            
                S = alpha[:,n] + alpha[:,m]
                u = (alpha[:,n]* alpha[:,m]) / ((S +1)*(S*S))
                #u_norm = u / u.max(1, keepdim=True)[0]                 
                all_uncertainties.extend(u.data.cpu().numpy().tolist())
                
            else:
                scores = m(scores)
            
            all_scores.extend(scores.data.cpu().numpy().tolist())
            all_labels.extend(labels.data.cpu().numpy().tolist())
            all_patients.extend(patients.data.cpu().numpy().tolist())
            all_data_patient.extend(data_patient)
        self.output = compute_auc(np.array(all_scores), np.array(all_labels), self.n_classes, self.classes_key, fold,self.out_folder)


    def plot_attn(self,fold,lmt=50):
        
        mm = nn.Softmax(dim=1)
        id2site = {0:'Amb',1:'Others',2:'PC',3:'OP',4:'ED',5:'IP'}
        i, (documents, para_per_document, sentences_per_para, words_per_sentence, labels, hdf, onsite, patients) = next(iter(enumerate(self.train_loader)))
        documents = documents.to(self.device)  # (batch_size, sentence_limit, word_limit)
        para_per_document = para_per_document.to(self.device)
        sentences_per_para = sentences_per_para.squeeze(1).to(self.device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(self.device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1)
        hdf = hdf.squeeze().to(self.device)  # (batch_size, sentence_limit, 6)  
        onsite = onsite.squeeze(1).to(self.device) 
        X = (documents, para_per_document, sentences_per_para,words_per_sentence, hdf, onsite)
        scores, word_alphas, sentence_alphas, para_alphas, document_embeddings = self.model(X)  
        cnt = 0
        if self.edl:

            alpha = torch.exp(scores) + 1
            n = torch.arange(0,alpha.shape[1],2)
            m = torch.arange(1,alpha.shape[1],2)
            
            scores = alpha[:,n] / (alpha[:,n] + alpha[:,m])
            #scores = mm(scores)
            S = alpha[:,n] + alpha[:,m]
            u = (alpha[:,n]* alpha[:,m]) / ((S +1)*(S*S))
            u_norm = u / u.max(0, keepdim=True)[0] 
                            
        else:
            scores = mm(scores)
                
        for ind,pat in enumerate(patients):
          
            if cnt<lmt:

                temp = para_alphas[ind].data.cpu().numpy()
                para_attn = {str(i):temp[i] for i in range(para_per_document[ind].item())}
                x = temp.argsort()[-1:][::-1]
                MIV = str(x[0])

                temp = sentence_alphas[ind][x[0]].data.cpu().numpy()
                sent_attn = {id2site[i]:temp[i] for i in range(sentences_per_para[ind][x[0]].item())}
                y = temp.argsort()[-1:][::-1]
                SIV = str(y[0])
        
                temp = word_alphas[ind][x[0]][y[0]].data.cpu().numpy()        
                code_attn = {self.id2label[documents[ind][x[0]][y[0]][k].item()]:temp[k] for k in range(words_per_sentence[ind][x[0]][y[0]].item())}
                #plot_attn(para_attn,sent_attn,code_attn,MIV,SIV)
        
                plt.rcParams.update({'font.size': 18})
                fig = plt.figure(figsize=(25,20))

                gs = fig.add_gridspec(2,2)
                ax1 = fig.add_subplot(gs[0, :])
                ax2 = fig.add_subplot(gs[1, 0])
                ax3 = fig.add_subplot(gs[1, 1])

                ax1.bar(para_attn.keys(), para_attn.values(), 0.3, edgecolor='black',color='royalblue')
                ax2.bar(sent_attn.keys(), sent_attn.values(), 0.3, edgecolor='black',color='grey')

                ax3.bar(code_attn.keys(), code_attn.values(), 0.3, edgecolor='black',color='grey')
                ax1.set_title('Time attention',fontsize=20)
                ax1.set_xlabel('Time window',fontsize=20)
                ax1.set_ylabel('Attention score')
                ax2.set_title('Site attention during time period {}'.format(MIV),fontsize=20)
                ax3.set_title('Code attention for Site {}'.format(id2site[int(SIV)]),fontsize=20)
                ax3.set_xticklabels(code_attn.keys(),rotation=10, fontsize=12)
                ax1.set_xticklabels(para_attn.keys(),rotation=10, fontsize=14)
                ax2.set_xticklabels(sent_attn.keys(),rotation=10, fontsize=16)

                fig.suptitle("Attention scores for Visit {0} with target label {1} and prediction {2}  "
                             .format(patients[ind].item(),labels[ind].cpu().tolist(),list(np.round(scores[ind].detach().cpu().tolist(),2))), fontsize=22)
 
                plt.show()
                plt.savefig(self.out_folder +'/Fold{0}/AttentionPlots/Attn_{1}_ind_{2}.pdf'.format(fold,patients[ind].item(),ind), format='pdf', dpi=500,bbox_inches='tight')
                plt.close()
                cnt = cnt + 1
            
