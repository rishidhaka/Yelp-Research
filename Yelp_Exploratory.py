#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests # Getting Webpage content
from bs4 import BeautifulSoup as bs # Scraping webpages
import json
import nltk
#nltk.download()
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
from IPython.display import display
import csv


# In[11]:


#creating business_df from json

data = {"name":[],"business_id":[],"city":[],"state":[],"postal_code":[],"categories":[], "is_open":[]}

with open('yelp_academic_dataset_business_21.json', encoding='utf-8') as json_file:
    for line in tqdm(json_file):
        business = json.loads(line)
        data['name'].append(business['name'])
        data['business_id'].append(business['business_id'])
        data['city'].append(business['city'])
        data['state'].append(business['state'])
        data['postal_code'].append(business['postal_code'])
        data['categories'].append(business['categories'])
        data['is_open'].append(business['is_open'])


# In[3]:


business_df = pd.DataFrame(data)
business_df.dropna(inplace=False)


# In[4]:


business_df.head()


# In[3]:


#creating review_df from json

data = {"review_id":[],"business_id":[],"text":[],"stars":[], "date":[]}
with open('yelp_academic_dataset_review_21.json', encoding='utf-8') as f:   
    for line in tqdm(f):
        review = json.loads(line)
        data['review_id'].append(review['review_id'])
        data['business_id'].append(review['business_id'])
        data['text'].append(review['text'])
        data['stars'].append(review['stars'])
        data['date'].append(review['date'])
review_df = pd.DataFrame(data)

review_df.head()


# In[5]:


#data_2021
print(len(business_df.index))


# In[4]:


print(len(review_df))


# In[8]:


#creating csvs

business_df.to_csv('business details_21.csv', index = False)
review_df.to_csv('review details_21.csv', index = False)


# In[9]:


from collections import Counter
import re


# In[179]:


#finding all categories

list_categories=[]
for i in data['categories']:
    if i:
        t=i.split(',')
        list_categories.extend(t)


# In[180]:


#creating set of all unique categories

unique=set(list_categories)
print(len(unique))
print(len(list_categories))


# In[83]:


#all categories

print(list_categories[:150])


# In[12]:


#finding all categories

list_categories_21=[]
for i in data['categories']:
    if i:
        t=i.split(',')
        list_categories_21.extend(t)


# In[13]:


#creating set of all unique categories

unique_21=set(list_categories_21)
print(len(unique_21))
print(len(list_categories_21))


# In[18]:


#all categories

print(unique_21)


# In[15]:


#Restaurants
restaurant=[]
for i in business_df['categories']:
    if i:
        condition = set(re.split(r'[,\s]+', i))
        if 'Food' in condition or 'Restaurant' in condition or 'Restaurants' in condition:
            restaurant.extend(['yes'])
        else:
            restaurant.extend(['no'])
    else:
        restaurant.extend(['no'])


# In[128]:


print(type(business_df['categories'][0]))


# In[16]:


#business_df.drop('is_restaurant', axis=1)
business_df['is_restaurant'] = restaurant


# In[17]:


business_df.head(50)


# In[96]:


#all unique categories

print(unique)


# In[134]:


print(len(review_df['business_id']))


# In[1]:


print(review_df['text'][:20])


# In[19]:


#creating list/set of business ids of all restaurants

business_id = business_df['business_id']
is_restaurant = business_df['is_restaurant']
is_restaurant_list = []

for i in range(len(business_id)):
    if is_restaurant[i]=='yes':
        is_restaurant_list.append(business_id[i])

is_restaurant_set = set(is_restaurant_list)


# In[30]:


#creating new review_df with reviews only for restaurants

data = {"review_id":[],"business_id":[],"text":[],"stars":[], "date":[]}
for i in range(len(review_df['review_id'])):
    if review_df['business_id'][i] in is_restaurant_set:
        data['review_id'].append(review_df['review_id'][i])
        data['business_id'].append(review_df['business_id'][i])
        data['text'].append(review_df['text'][i])
        data['stars'].append(review_df['stars'][i])
        data['date'].append(review_df['date'][i])
        
restaurant_review_df = pd.DataFrame(data)


# In[21]:


print(len(is_restaurant_list))


# In[22]:


#creating new restaurant_business_df

data = {"name":[],"business_id":[],"city":[],"state":[],"postal_code":[],"categories":[], "is_open":[]}
for i in range(len(business_df['business_id'])):
    if business_df['is_restaurant'][i]=='yes':
        data['name'].append(business_df['name'][i])
        data['business_id'].append(business_df['business_id'][i])
        data['city'].append(business_df['city'][i])
        data['state'].append(business_df['state'][i])
        data['postal_code'].append(business_df['postal_code'][i])
        data['categories'].append(business_df['categories'][i])
        data['is_open'].append(business_df['is_open'][i])
        
restaurant_business_df = pd.DataFrame(data)


# In[23]:


#creating csv for restaurant_business_df
restaurant_business_df.to_csv('restaurant details_21.csv', index = False)


# In[24]:


#chunking restaurant_review_df into 6 separate files
count = 1
chunk_size = int(restaurant_review_df.shape[0] / 6)
for start in range(0, restaurant_review_df.shape[0], chunk_size):
    df_subset = restaurant_review_df.iloc[start:start + chunk_size]
    name_txt = 'restaurant_review_21_'+str(count)+'.csv'
    count+=1
    df_subset.to_csv(name_txt, index=False)


#     Topic Modeling

# In[26]:


import pandas as pd
import os
# Load the regular expression library
import re


# In[29]:


#!pip3 install gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


# In[40]:


review_data = pd.read_csv('restaurant_review_21_1.csv')
data_text = review_data['text']
type(data_text)


# In[12]:


def lemmatize_stemming(text):
    #print(type(WordNetLemmatizer().lemmatize(text, pos='v')))
    #return SnowballStemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[42]:


processed_review_data = data_text.map(preprocess)
processed_review_data[:10]


# In[49]:


from gensim import corpora, models
# Create Dictionary
id2word = corpora.Dictionary(processed_review_data)
id2word.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in processed_review_data]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=7, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=20,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

print(lda_model.print_topics())


# In[48]:


print(lda_model.print_topics())


# In[5]:


#reading from all restaurant review files and concatenating them into 1 df

li = []
df = pd.read_csv('restaurant_review_21_1.csv', index_col=None, header=0)
li.append(df)
df = pd.read_csv('restaurant_review_21_2.csv', index_col=None, header=0)
li.append(df)
df = pd.read_csv('restaurant_review_21_3.csv', index_col=None, header=0)
li.append(df)
df = pd.read_csv('restaurant_review_21_4.csv', index_col=None, header=0)
li.append(df)
df = pd.read_csv('restaurant_review_21_5.csv', index_col=None, header=0)
li.append(df)
df = pd.read_csv('restaurant_review_21_6.csv', index_col=None, header=0)
li.append(df)
df = pd.read_csv('restaurant_review_21_7.csv', index_col=None, header=0)
li.append(df)

frame = pd.concat(li, axis = 0, ignore_index = True)


# In[6]:


frame.head()


# In[9]:


random_frame = frame.sample(frac=0.2)


# In[13]:


random_frame.to_csv('random_sample.csv', index=False)


# In[11]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


# In[14]:


random_sample_df = pd.read_csv('random_sample.csv')
random_text = random_sample_df['text']

processed_random_data = random_text.map(preprocess)
processed_random_data[:10]


# In[19]:


from gensim import corpora

# Create Dictionary
id2word = corpora.Dictionary(processed_random_data)
id2word.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in processed_random_data]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=20,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

pprint(lda_model.print_topics())


# In[21]:


print(lda_model.print_topics())


# In[22]:


# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)


# In[40]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

#topic_names = ['Permissions for device/local storage', 'SDK questions, native openreact questions, coding environment', 'Platform policies and support', '(Java) Code error']

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
#tick_formatter = FuncFormatter(lambda x, pos: topic_names[x])
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.set_title('Number of reviews by Dominant Topic', fontdict=dict(size=10))
ax1.set_ylabel('Number of reviews')
ax1.set_ylim(0, 900000)


# Topic Distribution by Topic Weights
ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
ax2.xaxis.set_major_formatter(tick_formatter)
ax2.set_title('Number of reviews by Topic Weightage', fontdict=dict(size=10))

plt.show()


# In[ ]:




