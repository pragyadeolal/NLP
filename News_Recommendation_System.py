#!/usr/bin/env python
# coding: utf-8

# ## Project - News Recommendation System

# ### Problem Statement:
# 
# iPrint is an upcoming media house in India that offers media and information services to the people. The company’s business extends across a wide range of media, including news and information services on sports, weather, education, health, research, stocks and healthcare. Over the years, through its online application, iPrint has been efficiently delivering news and information to the common people. However, with time and technological advancements, several new competitors of iPrint have emerged in the market. Hence, it has decided to begin providing a more personalised experience to its customers.
# 
# iPrint wants you as a data scientist to identify and build an appropriate recommendation system that would:
# 
#  - Recommend new top 10 relevant articles to a user when he visits the app at the start of the day
#  - Recommend top 10 similar news articles that match the ones clicked by the user.
# 
# 
# You have to ensure that the system does not recommend any news article that has been pulled out from the app or has already been seen by the user. The final generated list must contain the names of the recommended articles, along with their IDs.

# #### Importing the required modules and datasets

# In[73]:


#Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime,timedelta
import re
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk import download
from nltk import word_tokenize         
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

download('punkt')
download('stopwords')
               
stop_words = stopwords.words('english')

from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups


            


# In[74]:


#loading the datasets

df_platform = pd.read_csv('platform_content.csv')


# In[75]:


df_platform.head(5)


# In[76]:


df_platform.info()


# In[77]:


df_platform.shape


# The dataset platform_content.csv has 13 columns and 3122 rows. 
# Let's look at the missing values

# In[78]:


print('nMissing values:  ', df_platform.isnull().sum().values.sum())
df_platform.isnull().sum()


# In[79]:


df_platform.nunique()


# In[80]:


df_platform.language.unique()


# There are articles in 5 different languages in the platform_content.csv dataset. We need to make sure that only the articles that are written in the English language are considered, as specified in the problem statement. 

# Statistical properties of the dataset:

# In[81]:


df_platform.describe()


# In[82]:


#Importing the second dataset

df_consumer = pd.read_csv('consumer_transanctions.csv')
df_consumer.head(5)


# In[83]:


df_consumer.info()


# In[84]:


df_consumer.shape


# In[85]:


df_consumer.nunique()


# In[86]:


print('nMissing values:  ', df_consumer.isnull().sum().values.sum())
df_consumer.isnull().sum()


# Statistical properties of the dataset:

# In[87]:


df_consumer.describe()


# In[88]:


n_consumer = df_consumer.consumer_id.nunique()
n_consumer


# There are 1895 unique customer ids in the given dataset. 

# In[89]:


df_consumer.interaction_type.unique()


# Interaction_type has 'content_watched', 'content_followed', 'content_saved','content_liked', 'content_commented_on rating values, wherein the highest weightage is given to content_followed, followed by content_commented_on, content saved, content liked and content_watched. 
# 
# 
# Next, we'll be joining the two datasets platform_content.csv and consumer_transanctions.csv on item_id:

# In[90]:


#Joining the two datasets on 'item_id'

df=df_consumer.merge(df_platform,on='item_id')


# In[91]:


df


# In[92]:


df.shape


# Note: We have to ensure that the system does not recommend any news article that has been pulled out from the app or has already been seen by the user. So We'll be dropping out all the articles that have been already pulled out or already been seen by the user.

# In[93]:


#Dropping already pulled out articles

df=df[df.interaction_type_y=='content_present']
df.drop('interaction_type_y',axis=1,inplace=True)


# #### User-based collaborative recommendation

# In order to recommend new top 10 relevant articles to a user at the start of the day we'll be considering the following factors:
# 
# - Recency: Only Recent articles should be recommended
# - Region: The article recommended should be within the same region (Same country or state)
# - Content interaction: Based on user's interaction with the articles i.e. watched, liked, followed, downloaded - Most popular

# #### Recency
# 
# Considering the factor - How recent the artciles are, so that only Recent articles are recommended.

# In[94]:


#plotting histogram for timestamp for how recent the article is

df['days']=pd.to_datetime(df.event_timestamp_x,unit='s').dt.date
sns.histplot(x= df['days'], bins = 50)


# #### Interaction
# 
# Now, Based on user's interaction with the articles i.e. watched, liked, followed, downloaded - Most popular

# In[95]:


df_consumer['interaction_type'].value_counts().plot(kind='barh')


# In[96]:


# Popular items

df_consumer['item_id'].value_counts()[:10].plot(kind='bar')


# #### Region

# In[97]:


# Top regions of the product

df.producer_location.value_counts().index[:10]


# In[136]:


def preprocess(text):
    
   text = text.lower()
   doc = word_tokenize(text)
   doc = [word for word in doc if word not in stop_words]
   doc = [word for word in doc if word.isalpha()]
    
   return doc


# These are the top 10 Regions where we have the most interactive. Thus the regions above can be used in our reconmmended system. 

# ### Proposed Recommendation Engine. 
# 
# #### Ten reconmmend products/articles includes - Four recently popular articles + Three all time popular + Three region specific popular articles  
# 
# - Note: We need to ensure that the system does not recommend any news article that has been pulled out from the app or has already been seen by the user. Thus, all the articles are that are "pulled out" or pre-viewed articles by the user that have already been seen will excluded. Soo, all the articles that are recommended will be new to the user

# In[98]:


df.columns


# In[99]:


def recommed_system1(cons_id):
    '''
    This function takes consumer ID as input and returns 10 recommended products as explained above.
    
    cons_id : Consumer ID
    return : List of 10 products
    
    '''
    # Top 4 recently popular products
    
    dftmp=df[df.consumer_id!=cons_id]
    dftmp=dftmp[dftmp.days>=datetime(2017,1,1).date()]
    dftmp=dftmp[dftmp.interaction_type_x=='content_watched']
   
    recent=list(dftmp['item_id'].value_counts().index[:4])
    
    #Top 3 overall popular
    
    dftmp=df[df.consumer_id!=cons_id]
    dftmp=dftmp[dftmp.interaction_type_x=='content_watched']
    dftmp=dftmp[~dftmp.item_id.isin(recent)] 
    #it shouldn't be from previously included items
    
    overall=list(dftmp['item_id'].value_counts().index[:3])
    
    # Top 3 in that region - 
    
    k=df[df.consumer_id==cons_id].producer_location.value_counts().index[0]
    
    if k in ['SP', 'MG', 'NJ', 'NY', 'ON', 'GA', 'FL', 'IL', 'RJ', 'TX']:
        dftmp=df[df.consumer_id!=cons_id]
        dftmp=dftmp[(dftmp.interaction_type_x=='content_watched')&(dftmp.producer_location==k)]
        dftmp=dftmp[~dftmp.item_id.isin(recent+overall)]
        region=list(dftmp['item_id'].value_counts().index[:3])
    else:
        dftmp=df[df.consumer_id!=cons_id]
        dftmp=dftmp[dftmp.interaction_type_x=='content_watched']
        dftmp=dftmp[~dftmp.item_id.isin(recent+overall)]
        region=list(dftmp['item_id'].value_counts().index[:3])
        
    return df[df.item_id.isin(recent+overall+region)][['item_id','title']].reset_index(drop=True).drop_duplicates()


# In[100]:


# Test the engine
recommed_system1(-1032019229384696495)


# So for a particular user, there is an id associated i.e. consumer_id. Thus, on providing a customer_id (for example: -1032019229384696495) the system recommends the top 10 articles that are most relevant to the user/ customer_id. 

# ### Content based recommendation system. 
# 

# 
# Requirements: 
# Only the articles that are written in the English language must be considered for content-based recommendations. 
# The system should not recommend any news article that has been pulled out from the app or has already been seen by the user.
# 
# The final generated list must contain the names of the recommended articles, along with their IDs.

# ### Model 1 - Cosine similarity

# In[101]:


def preprocess_sentences(text):
    '''
    This function takes the text sentence input and preprocess it to make compatible to NLP models being use
    
    text: text sentence
    
    return : processed sentence
    '''
    text = text.lower()
    temp_sent =[]
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)

    finalsent = ' '.join(temp_sent)
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    return finalsent


# In[102]:


pc_present=df_platform[df_platform["interaction_type"]=="content_present"]

pc_present["all_text"]=pc_present["title"]+pc_present["text_description"]

pc_present.drop(["producer_device_info", "producer_location", "producer_country" ], axis=1, inplace=True)

pc_present.drop(["event_timestamp", "producer_id",
                 "producer_session_id", "item_url", "interaction_type"], axis=1, inplace=True)


# In[103]:


pc_present["all_text_processed"]=pc_present["all_text"].apply(preprocess_sentences)


# In[104]:


pc_present.columns


# In[105]:


# Vectorizing pre-processed movie plots using TF-IDF

tfidfvec = TfidfVectorizer()
tfidf_movie = tfidfvec.fit_transform((pc_present["all_text_processed"]))
cos_sim = cosine_similarity(tfidf_movie, tfidf_movie)

pc_present = pc_present.set_index('item_id') 

pc_present.head()


# In[106]:


# Storing indices of the data

indices = pd.Series(pc_present.index)


# In[107]:


def recommendations(id, cosine_sim = cos_sim):
    
    recommended_articles = []
    clicked_article=pc_present.loc[[id]]
    text=clicked_article['all_text_processed'].values[0]
    ix = indices[indices == id].index[0]
    
    similarity_scores = pd.Series(cosine_sim[ix]).sort_values(ascending = False)
    
    top_10_articles = list(similarity_scores.iloc[1:11].index)
    
    for i in top_10_articles:
        recommended_articles.append(list(pc_present.index)[i])
        
        
    return pc_present.loc[recommended_articles].title


# In[108]:


print("Top 10 articles similar to the clicked ones are - ")

#Insert the clicked item id to get top 10 recommendations
#For example if item clicked is - 5274322067107287523

cosine = recommendations(5274322067107287523)
cosine


# For every click on an article, there is an Item_id associated with it. 
# Thus, on providing an item id (for example:5274322067107287523) the system recommends the top 10 articles that are most relevant and most similar to the articles that have been previously clicked by the user. 

# ### Model 2 - K-means clustering
# 
# - Encode the item text using tfidf tokenzer and then use K-Means clustering to create clusters of the doc. 
# - Once we create the clusters, we can recommend the items from the same clusters. 

# In[109]:


dfkmn=df.copy()
dfkmn=dfkmn[dfkmn.language=='en']


# In[110]:


# combine all the relevent text for an item_id

dfkmn['all_text']=dfkmn['title']+dfkmn['text_description']
dfkmn['all_text']=dfkmn[['all_text','item_id']].groupby('item_id').transform(lambda x:' '.join(x))
dftmp=dfkmn[['all_text','item_id']].drop_duplicates()


# In[111]:


dftmp


# In[112]:


# vectorize the text doc

vectorizer = TfidfVectorizer(stop_words='english',max_features=7500)
docs=vectorizer.fit_transform(dftmp.all_text)


# In[113]:


kmeans = KMeans(16)
clstrs=kmeans.fit_predict(docs)
dftmp['clusters']=clstrs


# In[114]:


def recommend_kmeans(item_id):
    
    a=int(dftmp[dftmp.item_id==item_id].clusters)
    lst=list(dftmp[dftmp.clusters==a].item_id)
    lst=[i for i in lst if i !=item_id][:10]
    
    return dfkmn[dfkmn.item_id.isin(lst)][['item_id','title']].drop_duplicates()


# In[115]:


print("10 similar articles using kmeans clustering - ")

#Insert the clicked item id to get top 10 recommendations
# here if item clicked is - 5274322067107287523

kmns=recommend_kmeans(5274322067107287523)
kmns


# ### Model 3 - Using Topic Modelling
# 
# We'll use LDA to generate the topics for the document. And then we can recommend the articles with similar topics.

# In[116]:


dftpc=df.copy()
dftpc=dftpc[dftpc.language=='en']


# In[117]:


# combine all the relevent text for an item_id

dftpc['all_text']=dftpc['title']+dftpc['text_description']
dftpc['all_text']=dftpc[['all_text','item_id']].groupby('item_id').transform(lambda x:' '.join(x))
dftmp2=dftpc[['all_text','item_id']].drop_duplicates()


# In[118]:


class LemmaTokenizer(object):
    
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


# In[119]:


# vectorize the text doc

vectorizer = TfidfVectorizer(stop_words='english',max_features=7500,max_df=0.9)
docs=vectorizer.fit_transform(dftmp2.all_text)


# In[120]:


vocab=vectorizer.get_feature_names()


# In[121]:


lda = LatentDirichletAllocation(n_components=16)
tpc=lda.fit_transform(docs)


# In[122]:


words_list= lda.components_


# In[123]:


# Checking top 15 words in each topic

def get_top15(ind):
    lst=np.argsort(ind)[:-16:-1]
    return [vocab[i] for i in lst]

for i in words_list:
    print('\nTop words from this Topic')
    print(get_top15(i))


# If we observe output above - top 15 words in each, we notice that they have  been clustered in similar docs
# 
# For example - 
# 
# Top words from one of the topics are -
# 'google', 'data', 'new', 'learning', 'like', 'cloud', 'time', 'use', 'people', 'app', 'code', 'just', 'machine', 'drupal', 'work'
# 
# It can be observed that the these words are from topics similar to technology - goodle, data, cloud etc. 
# 
# Similarly for another topic -
# 'blockchain', 'bitcoin', 'bank', 'banks', 'banking', 'financial', 'kubernetes', 'ethereum', 'fintech', 'payments', 'currency', 'digital', 'payment', 'institutions', 'ledger'
# 
# There's no denying in saying that topic has clustered words reated to digital currency/ cryptocurrency or payments and storage. 

#  Now, Let's Recommend articles based on above clustering

# In[124]:


clstrs=[np.argmax(i) for i in tpc]
dftmp2['clusters']=clstrs

def recommend_topic(item_id):
    
    a=int(dftmp2[dftmp2.item_id==item_id].clusters)
    lst=list(dftmp2[dftmp2.clusters==a].item_id)
    lst=[i for i in lst if i !=item_id][:10]
    
    return dftpc[dftpc.item_id.isin(lst)][['item_id','title']].drop_duplicates()


# In[125]:


print("10 similar articles using Topic Modeling - ")

#Insert the clicked item id to get top 10 recommendations
# here if item clicked is - 5274322067107287523

tpmdl=recommend_topic(5274322067107287523)
tpmdl


# To check the accurary of the three models, we'll be using Latent Semantic Indexing or LSI score. It measures the text similarity. Text similarity has to determine how ‘close’ two pieces of text are both in surface closeness (lexical similarity) and meaning (semantic similarity). This is achieved by reducing the dimensionality of the document vectors by applying latent semantic analysis.

# In[127]:


#LSI score for Model-1 (Cosine Similarity)

dfcos=dftmp2[dftmp2.item_id.isin(cosine.index)].all_text
corpus = [preprocess(text) for text in dfcos.tolist()]


# In[128]:


dictionary = corpora.Dictionary(corpus)
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]
lsi = LsiModel(corpus_tfidf, id2word=dictionary)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sim_matrix = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                for i in range(len(corpus))])


# In[129]:


# LSI score

np.sum(sim_matrix)/(len(corpus)**2)


# In[130]:


# LSI score for Model - 2 (kmeans clustering)

dfcos=dftmp2[dftmp2.item_id.isin(kmns.item_id)].all_text
corpus = [preprocess(text) for text in dfcos.tolist()]


# In[131]:


dictionary = corpora.Dictionary(corpus)
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]
lsi = LsiModel(corpus_tfidf, id2word=dictionary)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sim_matrix = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                for i in range(len(corpus))])


# In[132]:


# LSI score

np.sum(sim_matrix)/(len(corpus)**2)


# In[133]:


# LSI score for model - 3 (Topic Modeling)

dfcos=dftmp2[dftmp2.item_id.isin(tpmdl.item_id)].all_text
corpus = [preprocess(text) for text in dfcos.tolist()]


# In[134]:


dictionary = corpora.Dictionary(corpus)
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]
lsi = LsiModel(corpus_tfidf, id2word=dictionary)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sim_matrix = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                for i in range(len(corpus))])


# In[135]:


# LSI score

np.sum(sim_matrix)/(len(corpus)**2)


# ### Observations:
# 
# LSI score of the models 
# 
# 
# - Cosine similarity : 0.208
# - K-means clustering : 0.1263
# - Topic Modeling : 0.1197
# 
# 
# Model 1 Cosine similarity has the highest LSI score out of the three models. Thus, We can conclude that it suggests most relevant and new articles precisely similar to the ones clicked. 
