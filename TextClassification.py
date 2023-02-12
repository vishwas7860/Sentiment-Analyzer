import pandas as pd
import numpy as np
import re
import string
import emoji
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from tqdm import tqdm
from matplotlib import pyplot as plt


def docVec(model, d):
    doc = [word for word in d.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)

def wordVec(x,y):
    text = x.apply(gensim.utils.simple_preprocess)
    model = gensim.models.Word2Vec(
                                    window=10,
                                    min_count = 2 
                                    )
    model.build_vocab(text)
    model.train(text, total_examples = model.corpus_count, epochs = model.epochs)
    x_train_bow = []
    for w in tqdm(x.values):
        x_train_bow.append(docVec(model, w))
    x_train_bow = np.array(x_train_bow)
    x_train, x_test, y_train, y_test = train_test_split(x_train_bow,y, test_size = 0.33, random_state = 1)
    return x_train, x_test, y_train, y_test, model
    
    
def CountVect(x,y, b=False, ng = (1,1)):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 1)
    cv = CountVectorizer(binary=b, ngram_range=ng)
    x_train_bow = cv.fit_transform(x_train).toarray()
    x_test_bow = cv.transform(x_test).toarray()
    
    return x_train_bow, x_test_bow, y_train, y_test, cv
    
    
def TFIDF(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 1)
    tfidf = TfidfVectorizer()
    x_train_bow = tfidf.fit_transform(x_train).toarray()
    x_test_bow = tfidf.transform(x_test).toarray()
    
    return x_train_bow, x_test_bow, y_train, y_test, tfidf
    

def tagRemoval(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)
    
def urlRemoval(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)
    
def puncRemoval(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def emojiReplacing(text):
    return emoji.demojize(text)
    
def stopwordsRemoval(text):
    return remove_stopwords(text)

    
def dataProcessing(df):
    #converts all the text in lower case........
    df.iloc[:,0] = df.iloc[:,0].str.lower()
    
    #Removes html tags from the text.......
    df.iloc[:,0] = df.iloc[:,0].apply(tagRemoval)
    
    #Removes urls from the text..........
    df.iloc[:,0] = df.iloc[:,0].apply(urlRemoval)
    
    #Removes punctuations from the text......
    df.iloc[:,0] = df.iloc[:,0].apply(puncRemoval)
    
    #Replace emojis with its meanings........
    df.iloc[:,0] = df.iloc[:,0].apply(emojiReplacing)
    
    #Removes duplicates rows.......
    df.drop_duplicates(inplace=True)
    
    #Removes stop words from text.......
    df.iloc[:,0] = df.iloc[:,0].apply(stopwordsRemoval)
    
    
    return df
    
    
def main():
    path = input("Please Enter File path:  ")
    df = pd.read_csv(path, nrows=1000)
    df = dataProcessing(df)
    
    #Splitting the data into two parts........
    x,y = df.iloc[:,0], df.iloc[:,1]
    
    #Encoding the y's value.....
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print("Read 10000 rows only..................")
    print("Please Selects the algorithm:")
    print("1. CountVectorizer\n2. TfIdf\n3. Word2Vec")
    algo = int(input())
    
    if algo == 1:
        choise = int(input("Please Enter no. 1. Default, 2. Customize:  "))
        
        if choise == 2:
            b = int(input("Binary: 1. True, 0: False:  "))
            print("n_grams (eg. 1 2, 3 3, etc.)")
            ng = tuple(map(int, input().split()))
            
            x_train, x_test, y_train, y_test, cv = CountVect(x,y, b, ng)
        elif choise == 1:
            x_train, x_test, y_train, y_test, cv = CountVect(x,y)
        
    elif algo == 2:
        x_train, x_test, y_train, y_test, tfidf = TFIDF(x,y)
    else:
        x_train, x_test, y_train, y_test, model = wordVec(x,y)
    
    alg = int(input("Choose Training Algo.... 1. NaiveBayes, 2. RandomForestClassifier:  "))
    print("-----------------------------------------------Accuracy Score------------------------------------------------------------")
    if alg == 1:
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        print()
        print()
    else:
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        print()
        print()
    o = int(input("Please Choose Option:  1. Upload csv file.   2. Enter text.  "))
    if o == 1:
        path = input("Please Enter File path:  ")
        df1 = pd.read_csv(path)
  
    elif o == 2:
        s = input("Please Enter Text.........\n")
        df1 = pd.DataFrame({"text":s}, index=[0])
        
    df1 = dataProcessing(df1)
    if algo == 1:
        test = cv.transform(df1.iloc[:,0]).toarray()
    elif algo == 2:
        test = tfidf.transform(df1.iloc[:,0]).toarray()
    else:
        test = []
        for w in tqdm(df1.iloc[:,0].values):
            test.append(docVec(model, w))
        test= np.array(test)
    if alg == 1:
        pred = gnb.predict(test)
    else:
        pred = rfc.predict(test)
    
    if o == 1:
        df1["sentiment"] = pred
        df1["sentiment"].replace({1:"positive", 0:"negative"}, inplace=True)
        def label_function(val):
            return f'{val / 100 * len(df1):.0f}\n{val:.0f}%'
        df1.groupby('sentiment').size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 20},
                                 colors=['red', 'lime'])
        plt.tight_layout()
        plt.show()
        
        df1.to_csv("Result.csv", index=False)
    elif o == 2:
        print("----------------------------------------------------------Sentiment-----------------------------------------------------------")
        if pred:
            print("Positive")
        else:
            print("Negative")
    
    
if __name__ == "__main__":
    main()