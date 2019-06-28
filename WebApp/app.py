import os

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template,request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle


app = Flask(__name__)



# infile1 = open('meta_asins_input.pkl','rb')
# meta_asins_input = pickle.load(infile1)
# infile1.close()

# infile2 = open('meta_asins_first.pkl','rb')
# meta_asins_first = pickle.load(infile2)
# infile2.close()

# infile3 = open('meta_asins_second.pkl','rb')
# meta_asins_second = pickle.load(infile3)
# infile3.close()

# infile4 = open('meta_asins_third.pkl','rb')
# meta_asins_third = pickle.load(infile4)
# infile4.close()




with open("model/model.pkl", "rb") as f:
    meta_asins_input = pickle.load(f)
    meta_asins_first = pickle.load(f)
    meta_asins_second = pickle.load(f)
    meta_asins_third = pickle.load(f)
    

data_list = []
for i in range(0,len(meta_asins_input)):
    group = meta_asins_input[i]
    for index in range(0,len(group)):
        data_list.append(group[index]) 

df1 = pd.DataFrame(data_list)
df1.rename(columns={0: 'InputASIN', 1: 'InputTitle',2:'InputImg'}, inplace=True)

data_list_2=[]
for i in range(0,len(meta_asins_first)):
    group = meta_asins_first[i]
    for index in range(0,len(group)):
        data_list_2.append(group[index]) 
df2 = pd.DataFrame(data_list_2)
df2.rename(columns={0: 'FirstASIN', 1: 'FirstTitle',2:'FirstImg'}, inplace=True)

data_list_3=[]
for i in range(0,len(meta_asins_second)):
    group = meta_asins_second[i]
    for index in range(0,len(group)):
        data_list_3.append(group[index]) 
df3 = pd.DataFrame(data_list_3)
df3.rename(columns={0: 'SecondASIN', 1: 'SecondTitle',2:'SecondImg'}, inplace=True)

data_list_4=[]
for i in range(0,len(meta_asins_third)):
    group = meta_asins_third[i]
    for index in range(0,len(group)):
        data_list_4.append(group[index]) 
df4 = pd.DataFrame(data_list_4)
df4.rename(columns={0: 'ThirdASIN', 1: 'ThirdTitle',2:'ThirdImg'}, inplace=True)

master_asin_df=pd.concat([df1, df2,df3,df4], axis=1)






@app.route("/")
def indexmain():
    """Return the homepage."""
    
    return render_template("index.html")

@app.route("/option1",methods=["GET", "POST"])
def display_asins():        
        return render_template("styleReco.html",asins=meta_asins_input)


@app.route("/getRecommendations",methods=["GET", "POST"])
def fetch_related_products():
    inp_asin=request.form.get('getReco')
    print(f"The input ASIN IS {inp_asin}")
    first_asin=master_asin_df[master_asin_df.eq(inp_asin).any(1)][['FirstASIN','FirstTitle','FirstImg']].values
    second_asin=master_asin_df[master_asin_df.eq(inp_asin).any(1)][['SecondASIN','SecondTitle','SecondImg']].values
    third_asin=master_asin_df[master_asin_df.eq(inp_asin).any(1)][['ThirdASIN','ThirdTitle','ThirdImg']].values
    print(f"The first ASIN IS {first_asin}")
    return render_template("viewRelatedItems.html",first_asin=first_asin,second_asin=second_asin,third_asin=third_asin)

@app.route("/option2",methods=["GET", "POST"])
def display_wordcloud():        
        return render_template("wordCloud.html")





if __name__ == "__main__":
    app.run(debug=True)

