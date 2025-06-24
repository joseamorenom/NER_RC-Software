# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:46:45 2022

@author: Santiago Moreno
"""
from upsampling import upsampling_ner
from flair.datasets import ColumnCorpus
from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from torch.optim.lr_scheduler import OneCycleLR
from flair.data import Sentence
from sklearn.model_selection import StratifiedGroupKFold
from distutils.dir_util import copy_tree
import numpy as np
import torch
import pandas as pd
import json
import os
import operator
import flair
import argparse
import shutil

default_path = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')
tagger_document = 0
tagger_sentence = 0

def check_create(path):
    import os
    
    if not (os.path.isdir(path)):
        os.makedirs(path)
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True','true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''
def copy_data(original_path):
    data_folder  = default_path + '/../../data/NER/train'
    copy_tree(original_path, data_folder)
'''

def copy_data(original_path):
    # copiar uno a uno sin intentar borrar nada
    data_folder = os.path.abspath(os.path.join(default_path, '../../data/NER/train'))
    os.makedirs(data_folder, exist_ok=True)
    for fname in os.listdir(original_path):
        src = os.path.abspath(os.path.join(original_path, fname))
        dst = os.path.abspath(os.path.join(data_folder,    fname))
         # si es el mismo archivo, lo saltamos
        if src == dst:
            continue
        try:
            shutil.copy2(src, dst)
        except PermissionError:
            # está en uso por otro proceso: no lo copiamos
            print(f"Warning: no pude copiar {src} porque está en uso.")
            continue
    
def characterize_data():
    data_folder  = default_path + '/../../data/NER/train'
    columns = {0: 'text', 1:'ner'}
    
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(
            data_folder, columns,
            train_file='train.txt',
            test_file='test.txt'
    )
    try:
        corpus: Corpus = ColumnCorpus(
            data_folder, columns,
            train_file='train.txt',
            test_file='test.txt'
    )
    except Exception as e:
        print('Error cargando corpus:', e)
        return 8


    '''
    except: 
        print('Invalid input document in training')
        return 8
    '''

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    #tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
    tag_dictionary = corpus.get_label_distribution()
    return tag_dictionary
    #return corpus
    

def upsampling_data(entities_to_upsample, probability,  entities):
    print('-'*20,'upsampling','-'*20)
    data_folder  = default_path + '/../../data/NER/train'
    columns = {'text':0, 'ner':1}
    for m in ["SiS","LwTR","MR","SR", "MBT"]:
        upsampler = upsampling_ner(data_folder+'/train.txt', entities+['O'], columns)
        data, data_labels = upsampler.get_dataset()
        new_samples, new_labels = upsampler.upsampling(entities_to_upsample,probability,[m])
        data += new_samples
        data_labels += new_labels

        with open(data_folder+'/train.txt', mode='w', encoding='utf-8') as f:
            for l,sentence in enumerate(data):
                for j,word in enumerate(sentence):
                    f.write(word+' '+ data_labels[l][j])
                    f.write('\n')
    
                if l < (len(data)-1):
                    f.write('\n')

    print('-'*20,'upsampling complete','-'*20)
    
    
def usage_cuda(cuda):
    if cuda:
        flair.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if flair.device == torch.device('cpu'): return 'Error handling GPU, CPU will be used'
        elif flair.device == torch.device('cuda:0'): return 'GPU detected, GPU will be used'
    else:
        flair.device = torch.device('cpu')
        return 'CPU will be used'


def training_model(name, epochs=20):
    data_folder = os.path.join(default_path, '..', '..', 'data', 'NER', 'train')
    path_model = os.path.join(default_path, '..', '..', 'models', 'NER', name)

    if os.path.isdir(path_model):
        print(f'ADVERTENCIA: El modelo {name} ya existe y será sobreescrito.')
        
    columns = {0: 'text', 1: 'ner'}

    try:
        corpus: Corpus = ColumnCorpus(data_folder, columns,
                                      train_file='train.txt',
                                      test_file='test.txt')
    except Exception as e:
        print(f"Error fatal: No se pudo cargar el corpus desde {data_folder}. Causa: {e}")
        return 8

    tag_type = 'ner'
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    try:
        embeddings = TransformerWordEmbeddings(
            model='xlm-roberta-large',
            layers="-1",
            subtoken_pooling="first",
            fine_tune=True,
            use_context=True,
        )
    except Exception as e:
        print(f"Error fatal: No se pudieron cargar los embeddings de Transformer. Causa: {e}")
        return 5

    try:
        tagger_train = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type='ner',
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
        )
    except Exception as e:
        print(f"Error fatal: No se pudo inicializar el SequenceTagger. Causa: {e}")
        return 6

    trainer = ModelTrainer(tagger_train, corpus)

    try:
        print(f"--- Iniciando entrenamiento del modelo {name} ---")
        trainer.train(path_model,
                      learning_rate=5.0e-6,
                      mini_batch_size=1,
                      mini_batch_chunk_size=1,  # Considera aumentarlo si tienes GPU con buena memoria
                      max_epochs=epochs,
                      scheduler=OneCycleLR,
                      embeddings_storage_mode='cpu',
                      optimizer=torch.optim.AdamW
                      )
    except Exception as e:
        import traceback
        print(f"Error fatal durante el entrenamiento. Causa: {e}")
        traceback.print_exc() # Imprime el traceback completo para un diagnóstico detallado
        return 7

    print(f"Modelo {name} entrenado y guardado exitosamente en {path_model}")
    return 0 # Devuelve 0 para indicar éxito
    
    
def tag_sentence(sentence, name):
    
    results={'Sentence_tagged':'', 'Highligth':{}}
    Highligth_dict={"text": "", "entities": []}
    
    
    #--------------Load the trained model-------------------------
    path_model = default_path + '/../../models/NER/{}'.format(name)
    #print(path_model)
    global tagger_sentence
    
    #tagger_sentence = SequenceTagger.load(path_model+'/best-model.pt')

    if (not tagger_sentence):
        
        try:
            tagger_sentence = SequenceTagger.load(path_model+'/best-model.pt')
        except:
            try:
                tagger_sentence = SequenceTagger.load(path_model+'/final-model.pt')
            except: 
                print('Invalid model')
                return 1
        
    #------------------Tagged sentence---------------------
    print('-'*20,'Tagging','-'*20)
    sentence_f = Sentence(sentence)
    tagger_sentence.predict(sentence_f)
    sentence_tokenized = []
    Highligth_dict['text'] = sentence_f.to_plain_string()
    
    for indx,token in enumerate(sentence_f.tokens):
        
        t = token.get_label()
        if t.value == 'O':
            sentence_tokenized += [token.text]
        else: 
            sentence_tokenized += [t.shortstring]
            token_info={
                'entity': t.value ,
                'index' : indx,
                'word' : token.text,
                'start': token.start_position,
                'end' : token.end_position
                
                }
            Highligth_dict["entities"].append(token_info)
    sen_tagged = ' ' .join(sentence_tokenized)
    results['Highligth'] = Highligth_dict
    results['Sentence_tagged'] = sen_tagged
    print('-'*20,'Tagged complete','-'*20)
    return results
    
    
def use_model(name, path_data, output_dir):
    
    #--------------Load the trained model-------------------------
    path_model = default_path + '/../../models/NER/{}'.format(name)
    
    if not (os.path.isdir(path_model)): 
        print('Model does not exists')
        return 10
        
    if not os.path.isfile(path_data): 
        print('Input file is not a file')
        return 9 
    
    global tagger_document
    
    if (not tagger_document):
        
        try:
            tagger_document = SequenceTagger.load(path_model+'/best-model.pt')
        except:
            try:
                tagger_document = SequenceTagger.load(path_model+'/final-model.pt')
            except: 
                print('Invalid model')
                return 1
    
    #-----------------Load the document-------------------------
    try:
        data = pd.read_json(path_data, orient ='index', encoding='utf-8')[0]
    except: 
        print('Can\'t open the input file')
        return 2
    
    if len(data) <= 0:
        print(f"length of document greater than 0 expected, got: {len(data)}")
        return 2
    
    try:
        sentences=data['sentences']
        t = sentences[0]['text']
    except: 
        print('Invalid JSON format in document {}'.format(path_data))
        return 3
    print('-'*20,'Tagging','-'*20)
    
    
    
    #-----------------Tagged the document-------------------------
    results = {'text':"", 'text_labeled':"",'sentences':[], 'entities': []}
    indx_prev = 0
    pos_prev = 0
    for s in sentences:
        sentence = Sentence(s['text'])
        tagger_document.predict(sentence, mini_batch_size = 1)
        sen_dict_temp = {'text':sentence.to_plain_string(), 'text_labeled':'', 'tokens':[]}
        #return sentence
        sentence_tokenized = []
        for indx,token in enumerate(sentence.tokens):
            token_dict = {'text':token.text, 'label':token.get_label('ner').value}
            sen_dict_temp['tokens'].append(token_dict)
            
            t = token.get_label('ner')
            if t.value == 'O':
                sentence_tokenized += [token.text]
            else: 
                sentence_tokenized += [t.shortstring]
                token_info={
                    'entity': t.value ,
                    'index' : indx + indx_prev,
                    'word' : token.text,
                    'start': token.start_position + pos_prev,
                    'end' : token.end_position +pos_prev
                    
                    }
                results["entities"].append(token_info)
        indx_prev += len(sentence.tokens)
        pos_prev += len(sentence.to_plain_string())
        sen_tagged = ' ' .join(sentence_tokenized)
        sen_dict_temp['text_labeled'] = sen_tagged
        results['sentences'].append(sen_dict_temp)
        results['text'] += sentence.to_plain_string() 
        #return sentence
        results['text_labeled'] += sen_tagged
        
    #-----------------Save the results-------------------------
    try:
        with open(output_dir, "w", encoding='utf-8') as write_file:
            json.dump(results, write_file)
    
        print('-'*20,'Tagged complete','-'*20)
        print('Document tagged saved in {}'.format(output_dir))
    except:
        print('Error in output file')
        return 11
    
    return results

def json_to_txt(path_data_documents):
    #-------------List the documents in the path------------
    documents=os.listdir(path_data_documents)
    if len(documents) <= 0:
        print('There are not documents in the folder')
        return 4
    
    data_from_documents={'id':[],'document':[],'sentence':[],'word':[],'tag':[]}
    
    #--------------Verify each documment-------------
    for num,doc in enumerate(documents):
        data=path_data_documents+'/'+doc
        df = pd.read_json(data, orient ='index')[0]
        try:
            sentences = df['sentences']
            t = sentences[0]['text']
            t = sentences[0]['id']
            t = sentences[0]['tokens']
            j = t[0]['text']
            j = t[0]['begin']
            j = t[0]['end']
            tags = df['mentions']
            if tags:
                tg = tags[0]['id']
                tg = tags[0]['begin']
                tg = tags[0]['end']
                tg = tags[0]['type']
        except: 
            print('Invalid JSON input format in document {}'.format(doc))
            return 3
            
       
        #-----------------Organize the data----------------
        for s in sentences:
            id_senten=s['id']
            for tk in s['tokens']:
                if len(tk['text'])==1:
                    #if ord(tk['text'])>=48 and ord(tk['text'])<=57 and ord(tk['text'])>=65 and ord(tk['text'])<=90 and ord(tk['text'])>=97 and ord(tk['text'])<=122:
                    tk_beg=tk['begin']
                    tk_end=tk['end']
                    data_from_documents['id'].append('d'+str(num)+'_'+id_senten)
                    data_from_documents['document'].append(doc)
                    data_from_documents['word'].append(tk['text'])
                    data_from_documents['sentence'].append(s['text'])
                    data_from_documents['tag'].append('O')
                    for tg in tags:
                        if id_senten == tg['id'].split('-')[0] and tk['begin']>=tg['begin'] and tk['begin']<tg['end']:
                            data_from_documents['tag'][-1]=tg['type']
                            break
                        
                else:
                    tk_beg=tk['begin']
                    tk_end=tk['end']
                    data_from_documents['id'].append('d'+str(num)+'_'+id_senten)
                    data_from_documents['document'].append(doc)
                    data_from_documents['word'].append(tk['text'])
                    data_from_documents['sentence'].append(s['text'])
                    data_from_documents['tag'].append('O')
                    for tg in tags:
                        if id_senten == tg['id'].split('-')[0] and tk['begin']>=tg['begin'] and tk['begin']<tg['end']:
                            data_from_documents['tag'][-1]=tg['type']
                            break

    X=np.array(data_from_documents['word'])
    y=np.array(data_from_documents['tag'])     
    groups=np.array(data_from_documents['id'])  
    
    
    #-------------------Save the data in CONLL format--------------
    group_kfold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    group_kfold.get_n_splits(X, y, groups)
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train, groups_test = groups[train_index], groups[test_index]
        break


                    

    X_write=[X_train,X_test]
    y_write=[y_train,y_test]
    groups_write=[groups_train, groups_test]
    archivos=['train','test']
    
    
    for k in range(2):
        X_temp = X_write[k]
        y_temp = y_write[k]
        groups_temp = groups_write[k]
        arch=archivos[k]
        id_in=groups_temp[0]
        
            
        data_folder  = default_path + '/../../data/NER/train'
        check_create(data_folder)
        count = 0
        with open(data_folder + '/{}.txt'.format(arch), mode='w', encoding='utf-8') as f:
            for i in range(len(X_temp)):
                if groups_temp[i] != id_in:
                    id_in=groups_temp[i]
                    f.write('\n')
                    count = 0

                count += 1
                f.write(X_temp[i]+' '+ y_temp[i])
                f.write('\n')
                
                if count >= 150: 
                    count = 0
                    f.write('\n')

            

