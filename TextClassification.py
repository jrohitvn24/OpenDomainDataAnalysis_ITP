import pandas as pd
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import os
from sklearn.model_selection import train_test_split

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
    
raw_data = pd.read_csv('England.csv')
raw_data = raw_data.dropna()

class TopicClassification:
    
    def __init__(self, num_topics=10, passes = 2,data = None,  weighting_scheme = 'tfidf',technique = 'LDA', keep_topN = 100000, min_df = 15, max_df = 0.5):
        self.num_topics = num_topics
        self.passes = passes
        self.data = data
        self.weighting_scheme = weighting_scheme       
        self.keep_topN = keep_topN
        self.min_df = min_df
        self.max_df = max_df
        self.technique = technique
    
    def lemmatize_stemming(self,text):
        stemmer = PorterStemmer()
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
    def preprocess(self,text):    
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result
    
    def process_data(self):
                
        processed_docs = self.data['content'].map(self.preprocess)        
        dictionary = gensim.corpora.Dictionary(processed_docs)            
        dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df, keep_n=self.keep_topN)    
        return processed_docs,dictionary
    
    def train_model(self):        
        
        processed_docs,dictionary = self.process_data()
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        
        if self.weighting_scheme == 'tf':                
            if self.technique == 'LDA':
                model = gensim.models.LdaModel(bow_corpus, num_topics=self.num_topics, id2word=dictionary, passes=self.passes)
            else:
                model = gensim.models.LsiModel(bow_corpus, num_topics=self.num_topics, id2word=dictionary, onepass = False, power_iters = self.passes)
        else:
            tfidf = models.TfidfModel(bow_corpus)
            corpus_tfidf = tfidf[bow_corpus]  
            if self.technique == 'LDA':
                model = gensim.models.LdaModel(corpus_tfidf, num_topics=self.num_topics, id2word=dictionary, passes=self.passes)        
            else:
                model = gensim.models.LsiModel(corpus_tfidf, num_topics=self.num_topics, id2word=dictionary, onepass = False, power_iters = self.passes)
        return model


ldatextclassifier = TopicClassification(num_topics=5,passes=2,data = raw_data,weighting_scheme = 'tfidf',technique = 'LDA')

lda_model = ldatextclassifier.train_model()


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

lsitextclassifier = TopicClassification(num_topics=5,passes=2,data = raw_data,weighting_scheme = 'tfidf',technique = 'LSI')

lsi_model = lsitextclassifier.train_model()


for idx, topic in lsi_model.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
