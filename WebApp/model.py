import os

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import nltk
import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
import string
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import mean_squared_error

from nltk.corpus import stopwords
stop = stopwords.words("english")
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')
import seaborn as sns

df = pd.read_csv("data/Clothing_Shoes_Jewelry_Reviews.csv")

#delete unwanted column for the 1st time
if 'Unnamed: 0' in df.columns:
    del df['Unnamed: 0']
else:
    exit
    
    
count = df.groupby("asin", as_index=False).count()
mean = df.groupby("asin", as_index=False).mean()
dfMerged = pd.merge(df, count, how='right', on=['asin'])

dfMerged["totalReviewers"] = dfMerged["reviewerID_y"]
dfMerged["overallScore"] = dfMerged["overall_x"]
dfMerged["summaryReview"] = dfMerged["summary_x"]
dfNew = dfMerged[['asin','summaryReview','overallScore',"totalReviewers"]]


dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
dfCount = dfMerged[dfMerged.totalReviewers >= 50]


dfProductReview = df.groupby("asin", as_index=False).mean()
ProductReviewSummary = dfCount.groupby("asin")["summaryReview"].apply(list)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
ProductReviewSummary.to_csv("ProductReviewSummary.csv")

df3 = pd.read_csv("ProductReviewSummary.csv")
df3 = pd.merge(df3, dfProductReview, on="asin", how='inner')

df3 = df3[['asin','summaryReview','overall']]

regEx = re.compile('[^a-z]+')
def cleanReviews(reviewText):
    reviewText = reviewText.lower()
    reviewText = regEx.sub(' ', reviewText).strip()
    return reviewText

df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)
df3 = df3.drop_duplicates(['overall'], keep='last')
df3 = df3.reset_index()

reviews = df3["summaryClean"] 
countVector = CountVectorizer(max_features = 300, stop_words='english') 
transformedReviews = countVector.fit_transform(reviews) 

dfReviews = DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)


dfReviews.to_csv("dfReviews.csv")

df_meta=pd.read_csv('data/Meta_Clothing_Shoes_Jewelry_Reviews.csv')

if 'Unnamed: 0' in df_meta.columns:
    del df_meta['Unnamed: 0']
else:
    exit



    
    # First let's create a dataset called X
X = np.array(dfReviews)

 #create train and test for 0.9
tpercent = 0.9
tsize = int(np.floor(tpercent * len(dfReviews)))

dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
#len of train and test
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)

neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(dfReviews_train)

input_asin_list=[]
first_asin=[]
second_asin=[]
third_asin=[]
for i in range(lentest):

    a = neighbor.kneighbors([dfReviews_test[i]])    
    related_product_list = a[1]    
    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    third_related_product = [item[2] for item in related_product_list]
    third_related_product = str(third_related_product).strip('[]')
    third_related_product = int(third_related_product)
    print ("Based on product reviews, for ", df3["asin"][lentrain + i] ," average rating is ",df3["overall"][lentrain + i])
    print ("The first similar product is ", df3["asin"][first_related_product] ," average rating is ",df3["overall"][first_related_product])
    print ("The second similar product is ", df3["asin"][second_related_product] ," average rating is ",df3["overall"][second_related_product])
    print ("The third similar product is ", df3["asin"][third_related_product] ," average rating is ",df3["overall"][third_related_product])
    print ("-----------------------------------------------------------")
        
    input_asin_list.append(df3["asin"][lentrain + i])
    first_asin.append(df3["asin"][first_related_product])
    second_asin.append(df3["asin"][second_related_product])
    third_asin.append(df3["asin"][third_related_product])
    
meta_asins_input=[]
meta_asins_first=[]
meta_asins_second=[]
meta_asins_third=[]
    
for i in (input_asin_list):
    meta_asins_input.append(df_meta[df_meta.eq(i).any(1)][['asin','title','imUrl']].values)
for i in (first_asin):
    meta_asins_first.append(df_meta[df_meta.eq(i).any(1)][['asin','title','imUrl']].values)
for i in (second_asin):
    meta_asins_second.append(df_meta[df_meta.eq(i).any(1)][['asin','title','imUrl']].values)
for i in (third_asin):   
    meta_asins_third.append(df_meta[df_meta.eq(i).any(1)][['asin','title','imUrl']].values)


with open("model.pkl","wb") as f:
    pickle.dump(meta_asins_input, f)
    pickle.dump(meta_asins_first, f)
    pickle.dump(meta_asins_second, f)
    pickle.dump(meta_asins_third, f)
    
        
# filename1 = 'meta_asins_input.pkl'
# outfile1 = open(filename1,'wb')  
# pickle.dump(meta_asins_input,outfile1)
# outfile1.close()  

# filename2 = 'meta_asins_first.pkl'
# outfile2 = open(filename2,'wb')  
# pickle.dump(meta_asins_first,outfile2)
# outfile2.close()

# filename3 = 'meta_asins_second.pkl'
# outfile3 = open(filename3,'wb')  
# pickle.dump(meta_asins_second,outfile3)
# outfile3.close()

# filename4 = 'meta_asins_third.pkl'
# outfile4 = open(filename4,'wb')  
# pickle.dump(meta_asins_third,outfile4)
# outfile4.close()