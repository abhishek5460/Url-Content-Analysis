import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import ast
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv('Translated_tokens.csv')

import pickle
pickle_in = open("word_frequency.picle","rb")
words_frequency = pickle.load(pickle_in)

top = 2500
from collections import Counter

features = np.zeros(df.shape[0] * top).reshape(df.shape[0], top)
labels = np.zeros(df.shape[0])
counter = 0
for i, row in df.iterrows():
    c = [word for word, word_count in Counter(ast.literal_eval(row['tokens_en'])).most_common(top)]
    labels[counter] = list(set(df['main_category'].values)).index(row['main_category'])
    for word in c:
        if word in words_frequency[row['main_category']]:
            features[counter][words_frequency[row['main_category']].index(word)] = 1
    counter += 1

    from tkinter import *
window = Tk()
window.title("Welcome to URL Content Analysis App")
window.geometry('300x200')
lbl = Label(window, text="Enter the URL",font=("Arial Bold", 15))
lbl.place(x=90,y=10)
lbl1=Label(window)
urls1=Entry(window)
urls1.place(x=50, y=40)

def code():
    #urls = ['https://flipkart.com']
    addre=urls1.get()
    urls=[]
    urls.append(addre)
    top = 2500
    for url in urls:
        html = urlopen(url, timeout=15).read()
        soup = BeautifulSoup(html, "html.parser")
        [tag.decompose() for tag in soup("script")]
        [tag.decompose() for tag in soup("style")]
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk.lower() for chunk in chunks if chunk)
        tokens = nltk.word_tokenize(text)

        from collections import Counter
        counter = 0
        features_pred = np.zeros(top * len(words_frequency)).reshape(len(words_frequency), top)
        c = [word for word, word_count in Counter(tokens).most_common(top)]

        for category in words_frequency.keys():
            for word in c:
                if word in words_frequency[category]:
                    features_pred[counter][words_frequency[category].index(word)] = 1
            counter+=1

        category_weight = []
        for i in features_pred:
            weight_cof = np.where(i == 1)[0]
            weight_sum = 0
            for cof in weight_cof:
                weight_sum += top - cof
            category_weight.append(weight_sum)
        cat_index = category_weight.index(max(category_weight))
        category = list(words_frequency.keys())[cat_index]
        feature = features_pred[cat_index].reshape(-1, top)
        lbl1.configure(text=category)
        lbl1.place(x=110,y=120)
 
btn = Button(window, text="Submit", command=code)
 
btn.place(x=120,y=80)
 
window.mainloop()