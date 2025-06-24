# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:31:58 2022

@author: gita
"""
import random
import numpy as np
import copy

class upsampling_ner:


    
    def __init__(self, path_data, entities, pos_labels):
        """
        

        Parameters
        ----------
        path_data : str
            Path of the dataset in format CONLL.
        entities : List
            List of the senten.
        pos_labels : Dict
            Dictionary where the keys are the kind of labels, and the values 
            are the position of the labels in one line

        Returns
        -------
        None.

        """
        self.__path_data = path_data
        self.__entities = entities
        self.__search_factor = 1000
        self.__pos_labels = pos_labels
        self.__get_data_variables()
        
    def __get_data_variables(self):
        """
        Takes the data path and turn the senteces into a matrix of shape
        (Sentences, tokens of each sentence).
        Also executes the __get_total_mentions. 

        Returns
        -------
        None.

        """
        col  = self.__pos_labels['ner']
        self.__dataset = []
        self.__data_labels = []
        data_temp = []
        labels_temp = []
        with open(self.__path_data, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                if line != '\n':
                    data_temp.append(line.split(' ')[0])
                    labels_temp.append(line.split(' ')[col][:-1])
                    #print('si')
                else:
                    self.__dataset.append(data_temp)
                    self.__data_labels.append(labels_temp)
                    data_temp = []
                    labels_temp = []
        self.__get_total_mentions_and_tokens()

        
    def get_mentions(self, sentence, labels):
        """
        Divide sentence to a dictionary of mentions and a  dictionary of labels
        of the mentions
        

        Parameters
        ----------
        sentence : List
            List of the tokens of the sentence.
        labels : List
            List of the labels of each token.

        Returns
        -------
        dict_mentions : Dictionary
            sentece divided by its entities mentions key=number of mention, 
            value= set of tokens in the mention.
        dict_label_mentions : Dictionary
            labels corresponding of the mentions in the same order as token 
            mentions. key= number of mention, value= label of the mention.

        """

        dict_mentions = {}
        dict_label_mentions = {}
        mention = 0
        #print(sentence)
        dict_mentions[mention] = [sentence[0]]
        
        dict_label_mentions[mention] = labels[0]
        for i,label in enumerate(labels[1:]):
            if label == labels[i]:
                dict_mentions[mention].append(sentence[i+1])
            else: 
                mention += 1
                dict_mentions[mention] = [sentence[i+1]]
                dict_label_mentions[mention] = labels[i+1]
            
        return dict_mentions, dict_label_mentions
        
        
    def __get_total_mentions_and_tokens(self):
        """
        Takes the dataset and divide ach sentence in mentions and it store it 
        in __all_mentions

        Returns
        -------
        None.

        """
        
        self.__all_mentions = {}
        self.__tokens_per_entity = {}
        
        for key in self.__entities:
            self.__all_mentions[key] = []
            self.__tokens_per_entity[key] = []
            
        for i,sentence in enumerate(self.__dataset):
            if sentence:
                for j,word in enumerate(sentence):
                    self.__tokens_per_entity[self.__data_labels[i][j]].append(word) 
                    
                mentions,label_mentions = self.get_mentions(sentence, self.__data_labels[i])
                for n,label in enumerate(label_mentions.values()):
                    if mentions[n] not in self.__all_mentions[label]: self.__all_mentions[label].append(mentions[n]);
    
    
    def get_mentions_dict(self):
        "Return all the mentions in the dataset"
        return self.__all_mentions
    
    
    def get_dataset(self):
        "Return the dataset"
        return self.__dataset, self.__data_labels
    

    def Label_wise_token_replacement(self, token_mentions, label_mentions, labels, p):
        """
        Do the Label wise token replacement to a sentence divided in mentions
        

        Parameters
        ----------
        token_mentions : Dictionary
            sentece divided by its entities mentions key=number of mention, 
            value= set of tokens in the mention.
        label_mentions : Dictionary
            labels corresponding of the mentions in the same order as token 
            mentions. key= number of mention, value= label of the mention
        labels : List
            list of entities to be upsampled.
        p : float
            probability upsampled a mention selected.

        Returns
        -------
        token_mentions : Dictionary
            token mentions but with mention replacement.

        """

        p = 1-p 
        for i in token_mentions.keys():
            if label_mentions[i] in labels:
                for j,token in enumerate(token_mentions[i]):
                    umbral=np.random.uniform(0,1)
                    if umbral>=p:
                        token_selected = random.choice(self.__tokens_per_entity[label_mentions[i]])
                        search = 0
                        while token_selected == token and search <= self.__search_factor:
                            token_selected = random.choice(self.__tokens_per_entity[label_mentions[i]])
                            search += 1
                        token_mentions[i][j] = token_selected
                        
        return token_mentions

    def synonym_replacement(self, token_mentions, label_mentions, labels, p): 
        
        """
        Do the synonym_replacement to a sentence divided in mentions
    
    
        Parameters
        ----------
        token_mentions : Dictionary
            sentece divided by its entities mentions key=number of mention, 
            value= set of tokens in the mention.
        label_mentions : Dictionary
            labels corresponding of the mentions in the same order as token 
            mentions. key= number of mention, value= label of the mention
        labels : List
            list of entities to be upsampled.
        p : float
            probability upsampled a mention selected.
    
        Returns
        -------
        token_mentions : Dictionary
            token mentions but with shuffled.
    
        """
        
        import requests
        from bs4 import BeautifulSoup
        url='http://www.wordreference.com/sinonimos/'
    
        p = 1-p        
        
        for i in token_mentions.keys():
            if label_mentions[i] in labels:
                for j,token in enumerate(token_mentions[i]):
                    umbral=np.random.uniform(0,1)
                    if umbral>=p:
                        
                        buscar=url+token
                        resp=requests.get(buscar)
                        bs=BeautifulSoup(resp.text,'lxml')
                        try:
                            lista=bs.find(class_='trans clickable')
                            sino=lista.find('li')
                            list_synonyms = sino.next_element.split(',  ')
                        except:
                            list_synonyms = False
                        if list_synonyms:
                            synonym_selected = random.choice(list_synonyms)
                            search = 0
                            while synonym_selected == token_mentions[i][j] and search <= self.__search_factor:
                                synonym_selected = random.choice(list_synonyms)
                                search += 1
                            token_mentions[i][j] = synonym_selected
                        
        return token_mentions



    def mention_replacement(self, token_mentions, label_mentions, labels, p):
        """
        Do the mentions replacement to a sentence divided in mentions
        

        Parameters
        ----------
        token_mentions : Dictionary
            sentece divided by its entities mentions key=number of mention, 
            value= set of tokens in the mention.
        label_mentions : Dictionary
            labels corresponding of the mentions in the same order as token 
            mentions. key= number of mention, value= label of the mention
        labels : List
            list of entities to be upsampled.
        p : float
            probability upsampled a mention selected.

        Returns
        -------
        token_mentions : Dictionary
            token mentions but with mention replacement.

        """

        p = 1-p 
        for i in token_mentions.keys():
            if label_mentions[i] in labels:
                umbral=np.random.uniform(0,1)
                if umbral>=p: 
                    set_of_mentions = self.__all_mentions[label_mentions[i]]
                    mention_selected = random.choice(set_of_mentions)
                    search = 0
                    while token_mentions[i] == mention_selected and search <= self.__search_factor:
                        mention_selected = random.choice(set_of_mentions)
                        search += 1
                    token_mentions[i] = mention_selected
        return token_mentions
    

        
    def shuffle_within_segments(self, token_mentions, label_mentions, labels, p): 
        """
        Do the shuffle within segments to a sentence divided in mentions


        Parameters
        ----------
        token_mentions : Dictionary
            sentece divided by its entities mentions key=number of mention, 
            value= set of tokens in the mention.
        label_mentions : Dictionary
            labels corresponding of the mentions in the same order as token 
            mentions. key= number of mention, value= label of the mention
        labels : List
            list of entities to be upsampled.
        p : float
            probability upsampled a mention selected.

        Returns
        -------
        token_mentions : Dictionary
            token mentions but with shuffled.

        """

        p = 1-p        
        for i in token_mentions.keys():
            if label_mentions[i] in labels:
                umbral=np.random.uniform(0,1)
                if umbral>=p: random.shuffle(token_mentions[i])
        return token_mentions
    
    def mention_back_traslation(self, token_mentions, label_mentions, labels, p):
        """
        Do the back traslation to each mention in a sentence divided in mentions
        

        Parameters
        ----------
        token_mentions : Dictionary
            sentece divided by its entities mentions key=number of mention, 
            value= set of tokens in the mention.
        label_mentions : Dictionary
            labels corresponding of the mentions in the same order as token 
            mentions. key= number of mention, value= label of the mention
        labels : List
            list of entities to be upsampled.
        p : float
            probability upsampled a mention selected.

        Returns
        -------
        token_mentions : Dictionary
            token mentions but with mention brack traslation.
}
        """

        from deep_translator import GoogleTranslator
        from nltk.tokenize import word_tokenize


        p = 1-p 
        for i in token_mentions.keys():
            if label_mentions[i] in labels:
                umbral=np.random.uniform(0,1)
                if umbral>=p: 
                    try: 
                        language = random.choice(['en', 'sv', 'fr', 'ja', 'ko', 'af', 'sq', 'cs', 'es', 'el', 'ga'])
                        to_translate = " ".join(token_mentions[i])
                        
                        #print("to_trans: ", to_translate[:20])
                        
                        translateden = GoogleTranslator(source='auto', target=language).translate(to_translate)
    
                        #print("Trans: ",translateden[:20])
    
                        translatedes = GoogleTranslator(source='auto', target='de').translate(translateden)
    
                        #print("back Trans: ",translatedes[:20])
                    
                        mention_selected = word_tokenize(translatedes)
                        token_mentions[i] = mention_selected
                    except: 
                        pass                 
        return token_mentions
    

    def upsampling(self, labels, p, methods=None):
        
        if methods is None: 
            print("Not upsampling required")
        else: 
            new_mentions = []
            new_labels = []
            for i,sentence in enumerate(self.__dataset):
                if sentence:
                    sentence_mentions,label_mentions = self.get_mentions(sentence, self.__data_labels[i])
                    
                    
                    if "SiS" in methods:
                        new_mentions_temp = self.shuffle_within_segments(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                        if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                            new_mentions.append(new_mentions_temp)
                            new_labels.append(label_mentions)
                            
                    
                    if "LwTR" in methods:
                        new_mentions_temp = self.Label_wise_token_replacement(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                        if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                            new_mentions.append(new_mentions_temp)
                            new_labels.append(label_mentions)
                            
                        
                    
                        
                    if "MR" in methods:
                        new_mentions_temp = self.mention_replacement(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                        if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                            new_mentions.append(new_mentions_temp)
                            new_labels.append(label_mentions)
                            
    
                    
                    if "SR" in methods:
                        new_mentions_temp = self.synonym_replacement(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                        if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                            new_mentions.append(new_mentions_temp)
                            new_labels.append(label_mentions)
                            
                            
                            
                    if "MBT" in methods:
                        new_mentions_temp = self.mention_back_traslation(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                        if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                            new_mentions.append(new_mentions_temp)
                            new_labels.append(label_mentions)
                    
                    
            #Turn the mentions into sentences
            new_samples_generated = []
            new_labels_generated = []
            
            for i,mentions in enumerate(new_mentions):
                new_labels_temp = new_labels[i]
                sample_temp = []
                labels_temp = []
                for key in mentions.keys():
                    sample_temp += mentions[key]
                    labels_temp += [new_labels_temp[key]]*len(mentions[key])
                new_samples_generated.append(sample_temp)
                new_labels_generated.append(labels_temp)
            return new_samples_generated, new_labels_generated
        
        
        
    def mention_to_sentence(self, mentions, labels):
        sample_temp = []
        labels_temp = []
        for key in mentions.keys():
            sample_temp += mentions[key]
            labels_temp += [labels[key]]*len(mentions[key])

        return sample_temp, labels_temp



    def upsampling_by_sentence(self, labels, p, methods=None):
        
        if methods is None: 
            print("Not upsampling required")
        else: 
            new_mentions = []
            new_labels = []
            map_sentences = []
            map_labels = []
            sentences_upsampled = []
            labels_upsampled = []
            
            for i,sentence in enumerate(self.__dataset):
                sentences_upsampled_temp  = {}
                labels_upsampled_temp  = {}
                
                sentences_upsampled_temp["Original"] = sentence
                labels_upsampled_temp["Original"] = self.__data_labels[i]
                
                sentence_mentions,label_mentions = self.get_mentions(sentence, self.__data_labels[i])
                
                
                if "SiS" in methods:
                    new_mentions_temp = self.shuffle_within_segments(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                    if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                        sentences_upsampled_temp["SiS"], labels_upsampled_temp["SiS"] = self.mention_to_sentence(new_mentions_temp, label_mentions)


                if "LwTR" in methods:
                    new_mentions_temp = self.Label_wise_token_replacement(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                    if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                        sentences_upsampled_temp["LwTR"], labels_upsampled_temp["LwTR"] = self.mention_to_sentence(new_mentions_temp, label_mentions)
                        
                    
                
                    
                if "MR" in methods:
                    new_mentions_temp = self.mention_replacement(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                    if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                        sentences_upsampled_temp["MR"], labels_upsampled_temp["MR"] = self.mention_to_sentence(new_mentions_temp, label_mentions)

                
                if "SR" in methods:
                    new_mentions_temp = self.synonym_replacement(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                    if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                        sentences_upsampled_temp["SR"], labels_upsampled_temp["SR"] = self.mention_to_sentence(new_mentions_temp, label_mentions)
                        
                        
                        
                if "MBT" in methods:
                    new_mentions_temp = self.mention_back_traslation(copy.deepcopy(sentence_mentions), label_mentions,labels ,p)
                    if new_mentions_temp not in new_mentions and new_mentions_temp != sentence_mentions: 
                        sentences_upsampled_temp["MBT"], labels_upsampled_temp["MBT"] = self.mention_to_sentence(new_mentions_temp, label_mentions)
                
                if len(sentences_upsampled_temp)>1:
                    print(len(sentences_upsampled_temp))
                    sentences_upsampled.append(sentences_upsampled_temp)
                    labels_upsampled.append(labels_upsampled_temp)
                    
            return sentences_upsampled, labels_upsampled


