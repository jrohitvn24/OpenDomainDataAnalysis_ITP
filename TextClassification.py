import pandas as pd
import gensim
import pyLDAvis
import plotly.express as px
import plotly.figure_factory as ff
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
            bigrams = gensim.models.phrases.Phraser(bigrams_phrases)            
            data_bigrams = [bigrams[doc] for doc in data_words]            
            self.processed_docs = data_bigrams
            self.dictionary = gensim.corpora.Dictionary(self.processed_docs)            
            self.dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df, keep_n=self.keep_topN)                    
            self.corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_docs]
            if self.weighting_scheme == 'tfidf':
                tfidf = models.TfidfModel(self.corpus)
                self.corpus = tfidf[self.corpus]
        def train_model(self,num_topics = 10, passes = 10,alpha = 'symmetric', beta = 'auto', best_model = ''):                    
            self.num_topics = num_topics
            self.passes = passes
            self.alpha = alpha
            self.beta = beta            
            if best_model == '':                
                self.model = gensim.models.LdaMulticore(self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=self.passes, workers = 16, random_state=100, alpha= self.alpha, eta= self.beta)                    
            else:
                self.model = gensim.models.LdaModel.load(best_model)
            return self.model
        
        def coherence_score(self):
            coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=self.model, texts=self.processed_docs, dictionary=self.dictionary, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            return coherence_lda    
        
        def visualize(self,filepath):
            vis = gensim_models.prepare(self.model, self.corpus, self.dictionary)
            pyLDAvis.save_json(vis, filepath[:-4] + 'json')
            pyLDAvis.save_html(vis, filepath)            
        
        def save_sunburst(self,df,filename):            
            
            fig = px.sunburst(df, path=['country','Topic_Labels'],
                              color='country'
                              )
            fig.update_traces(textinfo="label+percent entry", insidetextorientation = 'radial')
            fig.show()            
            fig.write_html(filename)
        
        def save_country_summary(self,df,filename):            
            df_eng = df.loc[df['country'] == 'England'].groupby(['Topics'])['Topics'].count()
            df_ni = df.loc[df['country'] == 'Northern Ireland'].groupby(['Topics'])['Topics'].count()
            df_w = df.loc[df['country'] == 'Wales'].groupby(['Topics'])['Topics'].count()
            df_s = df.loc[df['country'] == 'Scotland'].groupby(['Topics'])['Topics'].count()
            topic_labels = pd.Series(df['Topic_Labels'].unique())
            df_summary = pd.concat([topic_labels,df_eng,df_ni,df_w,df_s],axis = 1)
            df_summary.columns = ['Topics Labels','England','Northern Ireland','Wales','Scotland']
            df_summary = df_summary.fillna(0)
            
            
            fig = ff.create_table(df_summary)
            fig.show()
            fig.write_html(filename)
            
        def transform_data(self,model = None, corpus = None, tranform_data = None, topic_labels = None):
            
            if tranform_data is None:
                tranform_data = self.data
            tranform_data.reset_index(inplace=True, drop=True)  
            if model is None:
                model = self.model
            
            if corpus is None:
                corpus = self.corpus
            
            sent_topics_df = pd.DataFrame()
            
            # Get main topic in each document
            for i, row in enumerate(model[corpus]):
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                if topic_labels is None:
                    topic_labels = list(range(1,self.num_topics+1))
                # Get the Dominant topic, Perc Contribution and Keywords for each document
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:  # => dominant topic
                        wp = model.show_topic(topic_num,topn = 30)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num+1), round(prop_topic,4), topic_keywords,topic_labels[int(topic_num)]]), ignore_index=True)
                    else:
                        break
            sent_topics_df.columns = ['Topics', 'Perc_Contribution', 'Topic_Keywords','Topic_Labels']
        
            # Add original text to the end of the output            
            sent_topics_df = pd.concat([sent_topics_df, tranform_data], axis=1)
            return sent_topics_df

    alpha = 'symmetric'
    beta = 'auto'
    topics = 40
    
    ldatextclassifier = TopicClassification()
    ldatextclassifier.process_data(data = raw_data,weighting_scheme = 'tfidf', min_df=100,keep_topN=10000, max_df=0.5)    
    ldatextclassifier.train_model(num_topics=topics,passes=10, alpha = alpha, beta = beta, best_model = 'BestModel/ldamodel')
    preplexity = ldatextclassifier.model.log_perplexity(ldatextclassifier.corpus)
    print('\nPerplexity: ', preplexity)
    coherence = ldatextclassifier.coherence_score()
    print('\nCoherence Score: ', coherence)     
    for idx, topic in ldatextclassifier.model.print_topics(num_topics=-1,num_words=15):
        print('Topic: {} Word: {}'.format(idx, topic))        
    
    #ldatextclassifier.model.save('BestModel/ldamodel')
    ldatextclassifier.visualize(f'lda_{topics}_{alpha}_{beta}.html')
    topic_labels = ['Sports','Entertainment','Farm-Location','Farming','After-School','Buildings','School','Community','Real-Estate','Nature','Church','Routes and Places','Transport','Industrial','News','Shop','Gardening','Mail','Animal Farm','Home','Notable Name','Death','Hospital','Crime','Airways','Water Treatment','Jobs','Plant','Harbour','Scary','Language','Towns','Fire','Notable Names','Towns','Crops','Pets','Parliament','Location','Dumfri']
    df = ldatextclassifier.transform_data(topic_labels=topic_labels)
    #df.to_csv('tranformed.csv')
    ldatextclassifier.save_country_summary(df, 'summary_table.html')
    ldatextclassifier.save_sunburst(df, 'sunburst.html')
if __name__ == "__main__":
    main()
