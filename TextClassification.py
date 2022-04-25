import pandas as pd
import gensim
import pyLDAvis
from pyLDAvis import gensim_models
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

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
        
    raw_data_eng = pd.read_csv('England.csv')
    raw_data_ni = pd.read_csv('NI.csv')
    raw_data_scot = pd.read_csv('Scotland.csv')
    raw_data_wales = pd.read_csv('Wales.csv')
    raw_data = pd.concat([raw_data_eng, raw_data_ni, raw_data_scot, raw_data_wales])
    raw_data = raw_data.dropna()    
    class TopicClassification:
        
        def __init__(self):            
            self.num_topics = None
            self.passes = None
            self.alpha = None
            self.beta = None
            self.data = None
            self.weighting_scheme = None
            self.keep_topN = None
            self.min_df = None
            self.max_df = None            
            self.corpus = []
            self.processed_docs = []
            self.dictionary = []
            self.model = None
            
        
        def lemmatize_stemming(self,text):
            stemmer = PorterStemmer()
            return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
        
        def preprocess(self,text):    
            result = []
            for token in gensim.utils.simple_preprocess(text):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    result.append(self.lemmatize_stemming(token))
            return result
        
        def process_data(self,data = None,  weighting_scheme = 'tfidf',keep_topN = 10000, min_df = 15, max_df = 0.5):
            self.data = data            
            self.weighting_scheme = weighting_scheme       
            self.keep_topN = keep_topN
            self.min_df = min_df
            self.max_df = max_df            
            data_words = self.data['content'].map(self.preprocess) 
            
            bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=50)
            #trigrams_phrases = gensim.models.Phrases(bigrams_phrases[data_words], threshold=50)
            bigrams = gensim.models.phrases.Phraser(bigrams_phrases)
            #trigrams = gensim.models.phrases.Phraser(trigrams_phrases)            
            data_bigrams = [bigrams[doc] for doc in data_words]
            #data_bigrams_trigrams = [trigrams[bigrams[doc]] for doc in data_bigrams]
            self.processed_docs = data_bigrams
            self.dictionary = gensim.corpora.Dictionary(self.processed_docs)            
            self.dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df, keep_n=self.keep_topN)                    
            self.corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_docs]
            if self.weighting_scheme == 'tfidf':
                tfidf = models.TfidfModel(self.corpus)
                self.corpus = tfidf[self.corpus]
        def train_model(self,num_topics = 10, passes = 10,alpha = 'symmetric', beta = 'auto'):                    
            self.num_topics = num_topics
            self.passes = passes
            self.alpha = alpha
            self.beta = beta            
            self.model = gensim.models.LdaMulticore(self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=self.passes, workers = 16, random_state=100, alpha= self.alpha, eta= self.beta)                    
            return self.model
        
        def coherence_score(self):
            coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=self.model, texts=self.processed_docs, dictionary=self.dictionary, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            return coherence_lda    
        
        def visualize(self,filepath):
            vis = gensim_models.prepare(self.model, self.corpus, self.dictionary)
            pyLDAvis.save_html(vis, filepath)
     
    #alpha = np.linspace(0.000001,1,50)
    #beta = np.linspace(0.000001,1,50)
    #topics = [40]
    alpha = ['symmetric']
    beta = [0.0001]
    topics = [30]
    
    ldatextclassifier = TopicClassification()
    ldatextclassifier.process_data(data = raw_data,weighting_scheme = 'tfidf', min_df=100,keep_topN=10000, max_df=0.5)
    for t in topics:
        for a in alpha:
            for b in beta:
                #ldatextclassifier = TopicClassification(num_topics=40,passes=10,data = raw_data,weighting_scheme = 'tfidf', alpha = a, beta = b)
                ldatextclassifier.train_model(num_topics=t,passes=10, alpha = a, beta = b)
                preplexity = ldatextclassifier.model.log_perplexity(ldatextclassifier.corpus)
                print('\nPerplexity: ', preplexity)
                coherence = ldatextclassifier.coherence_score()
                print('\nCoherence Score: ', coherence)    
                for idx, topic in ldatextclassifier.model.print_topics(num_topics=-1,num_words=15):
                    print('Topic: {} Word: {}'.format(idx, topic))        
                
                #ldatextclassifier.model.save('ldamodel')
                ldatextclassifier.visualize(f'lda_{t}_{a}_{b}.html')
if __name__ == "__main__":
    main()