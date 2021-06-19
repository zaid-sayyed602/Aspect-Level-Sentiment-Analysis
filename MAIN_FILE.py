from tkinter import *
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sb
import re, string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from wordcloud import WordCloud,STOPWORDS
import pickle
from nltk import pos_tag
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import wordnet
import spacy
root = Tk()
root.geometry("1500x1000")
nlp = spacy.load("en_core_web_sm")
df=pd.read_csv("corpus.csv",encoding="latin-1")
df=df.drop("label",axis=1)
df=df.dropna()
print(df)
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = re.sub('[^a-zA-Z]',' ', text)
    text = [word.strip(string.punctuation) for word in text.split()]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)
df["text"] = df["text"].apply(lambda x: clean_text(x))

def rev_clean(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]',' ', text)
    text = [word.strip(string.punctuation) for word in text.split()]
    text = [t for t in text if (len(t)>0)]
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return(text)

#-----finding polarity of the review
df["polarity"]=df.apply(lambda x: TextBlob(x["text"]).sentiment.polarity,axis=1)
l1=[]
for pol in df["polarity"]:
    if(pol<0):
        l1.append("negative")
    elif(pol>0):
        l1.append("positive")
    elif(pol==0):
        l1.append("neutral")
df["polarity"]=l1
le=LabelEncoder()
df["polarity"]=le.fit_transform(df["polarity"].tolist())
print(df)

##---WORD CLOUD------------
negative = df['text'][df['polarity'] == 0 ]
neutral = df['text'][df['polarity'] == 1 ]
positive = df['text'][df['polarity'] == 2 ]
def features_func(text):
  tvec = TfidfVectorizer(stop_words='english', max_df=0.3, min_df=2)
  tvec.fit_transform(text)
  features = tvec.get_feature_names()
  return features

positive_features = features_func(positive)
negative_features = features_func(negative)
netural_features = features_func(neutral)

def word_cloud(text,name):
  stop_words = STOPWORDS
  wc = WordCloud(background_color = "white", max_words = 300,stopwords = stop_words).generate(str(text))
  wc.to_file(name)
  plt.imshow(wc)
  plt.title(name)
  plt.show()

def word_clouds():
    word_cloud(positive,'positive.png')
    word_cloud(negative,'negative.png')
    word_cloud(neutral,'netural.png')

##----PREPARING THE MODEL FOR TRAIN-----
xtrain, xtest, ytrain, ytest = train_test_split(df["text"], df["polarity"], test_size=0.2, random_state=7)

td = TfidfVectorizer(max_features = 5000,stop_words='english',sublinear_tf = True,use_idf = True)
td.fit(df["text"])
xtrain_vec = td.transform(xtrain)
xtest_vec = td.transform(xtest)
clf = svm.SVC(kernel='linear')
clf.fit(xtrain_vec,ytrain)
##pickle.dump(clf, open("svmmodel.sav", 'wb'))
##print("model saved to a file")
##clf_load= pickle.load(open("svmmodel.sav", 'rb'))
ypred = clf.predict(xtest_vec)
print("ACCURACY SCORE\n",metrics.accuracy_score(ytest,ypred)*100)
acc=metrics.accuracy_score(ytest,ypred)*100
print("CLASSIFICATION REPORTS\n",metrics.classification_report(ytest, ypred))
print("CONFUSION MATRIX\n",metrics.confusion_matrix(ytest, ypred))

def readwords( filename ):
    f=open(filename)
    words = [line.rstrip() for line in f.readlines()]
    return words
positive = readwords("positive-words.txt")
negative = readwords("negative-words.txt")
negation = readwords("negation-words.txt")

def find_sentiment(doc):
    ner_heads = {ent.root.idx: ent for ent in doc.ents}
    rule3_pairs = []
    des = []
    for token in doc:
        children = token.children
        A = ""
        M = ""
        neg_prefix = ""
        add_neg_pfx = False
        for child in children:
##            print(child.text, child.dep_, child.head.text, child.head.pos_,
##            child.pos_,[child1 for child1 in child.children])
##            
            if(child.dep_ == "nsubj" and child.pos_=="NOUN"and not child.is_stop) or (child.head.pos_ == "NOUN" and child.pos_=="NOUN"and not child.is_stop
            ) or (child.pos_ == "NOUN" and child.dep_ == "pobj" and not child.is_stop)  or (child.pos_ == "NOUN" and child.dep_ == "dobj" and not child.is_stop) or (child.pos_ == "NOUN" and child.dep_ == "compound" and not child.is_stop):
                if child.idx in ner_heads:
                    A = ner_heads[child.idx].text
                else:
                    A = child.text
            if(child.pos_ == "ADJ"and not child.is_stop) or (child.dep_ == "advmod" and child.pos_ == "ADV" and not child.is_stop
            ) or (child.head.pos_ == "AUX" and child.pos_ == "PRON"and not child.is_stop):
                N = ''
                for child1 in child.children:
                    if(child1.pos_ == "ADV" ) or (child1.pos_ == "ADJ"):
                       N += child1.text + " "
                M += N + " " + child.text + " "
##                M += child.text + " "
            if(child.pos_ == "NOUN" and child.dep_ == "pobj" and not child.is_stop)  or (child.pos_ == "NOUN" and child.dep_ == "dobj" and not child.is_stop) or (child.pos_ == "NOUN" and child.dep_ == "compound" and not child.is_stop):
                for neg_words in negative:
                    if(child.head.text == neg_words ):
                        M += child.head.text + ' '
                for pos_words in positive:
                    if(child.head.text == pos_words ):
                        M += child.head.text + ' '
            if(child.dep_ == "aux" and child.tag_ == "MD"): 
                neg_prefix = "not"
                add_neg_pfx = True
            if(child.dep_ == "neg"):
                if(child.text == "nt" or child.text == "n't"):
                    neg_prefix = "not"
                else:
                    neg_prefix = child.text
                add_neg_pfx = True
        if (add_neg_pfx and M != ""):
            M = neg_prefix + " " + M
        if(A != "" and M != ""):
            og_rev = M.split()
##            print("Original",len(og_rev))
##            print(M)
            rev = td.transform([M])
##            print(rev)
            ypred1 = clf.predict(rev)
##            print(ypred1)
            ans = ''
            if(ypred1 == 0):
                ans = "Negative"
                for i in range(len(positive)):
                    for j in range(len(negative)):
                        if(len(og_rev) == 1):
                            if(negative[j] in og_rev):
                                ans = "Negative"
                            if(positive[i] in og_rev):
                                ans = "Positive"
                        else:
                            if(negative[j] in og_rev):
                                pos = og_rev.index(negative[j])
                                for n in negative:
                                    if(og_rev[pos-1]==n or og_rev[pos-2]==n):
                                        ans="Positive"
                                for p in positive:
                                    if(og_rev[pos-1]==p or og_rev[pos-2]==p):
                                        ans = "NEgative"
                                    
            
            elif(ypred1 == 1):
                ans = "Neutral"

            elif(ypred1 == 2):
                ans= "Positive"
                for i in range(len(positive)):
                    for j in range(len(negative)):
                        if(len(og_rev) == 1):
                            if(positive[i] in og_rev):
                                ans = "positive"
                            if(negative[j] in og_rev):
                                ans = "negative"
                        else:
                            if(positive[i] in og_rev):
                                pos = og_rev.index(positive[i])
                                for n in negative:
                                    if(og_rev[pos-1]==n or og_rev[pos-2]==n):
                                        ans="Negative"
                                for p in positive:
                                    if(og_rev[pos-1]==p or og_rev[pos-2]==n):
                                        ans = "POsitive"

            rule3_pairs.append([A, M,ans])            
    return rule3_pairs       



def find_sentiment2(sentences):
    for sentence in sentences:
        doc = nlp(sentence)
        ner_heads = {ent.root.idx: ent for ent in doc.ents}
        upd_rev=[]
        A = ''
        M = ''
        add_neg = False
        for token in doc:
##            print(token.text, token.dep_, token.head.text, token.head.pos_,
##            token.pos_,[child for child in token.children])
            if(token.dep_ == "nsubj" and token.pos_=="NOUN") or (token.head.pos_ == "NOUN" and token.pos_=="NOUN"
            ) or (token.pos_ == "NOUN" and token.dep_ == "pobj"  or token.dep_ == "dobj" or token.dep_ == "compound"):
                if(token.idx in ner_heads):
                    A = nre_heads[token.idx].text
                else:
                    A = token.text
            if(token.pos_ == "ADJ"and not token.is_stop) or (token.dep_ == "advmod" and token.pos_ == "ADV" and not token.is_stop
            ) or (token.head.pos_ == "AUX" and token.pos_ == "PRON"and not token.is_stop):
                N = ''
                for child1 in token.children:
                    if(child1.pos_ == "ADV" ) or (child1.pos_ == "ADJ"):
                       N += child1.text + " "
                M = N + " " + token.text + " "
##                M += child.text + " "
            if(token.pos_ == "NOUN" and token.dep_ == "pobj" and not token.is_stop)  or (token.pos_ == "NOUN" and token.dep_ == "dobj" and not token.is_stop) or (token.pos_ == "NOUN" and token.dep_ == "compound" and not token.is_stop):
                for neg_words in negative:
                    if(token.head.text == neg_words ):
                        M += token.head.text + ' '
                for pos_words in positive:
                    if(token.head.text == pos_words ):
                        M += token.head.text + ' '
            if(token.dep_ == "neg"): 
                if(token.text == "nt" or token.text == "n't"):
                    neg = "not"
                else:
                    neg = token.text
                add_neg = True
        if(add_neg and M != "999999"):
            M = neg + ' ' + M
        if(A != "999999" and M != "999999"):
            og_rev = M.split()
##            print("Original",len(og_rev))
##            print(M)
            rev = td.transform([M])
##            print(rev)
            ypred1 = clf.predict(rev)
##            print(ypred1)
            
            ans = ''
            if(ypred1 == 0):
                ans = "Negative"
                for i in range(len(positive)):
                    for j in range(len(negative)):
                        if(len(og_rev) == 1):
                            if(negative[j] in og_rev):
                                ans = "Negative"
                            if(positive[i] in og_rev):
                                ans = "Positive"
                        else:
                            if(negative[j] in og_rev):
                                pos = og_rev.index(negative[j])
                                for n in negative:
                                    if(og_rev[pos-1]==n or og_rev[pos-2]==n):
                                        ans="Positive"
                                for p in positive:
                                    if(og_rev[pos-1]==p or og_rev[pos-2]==p):
                                        ans = "NEgative"
                                    
            
            elif(ypred1 == 1):
                ans = "Neutral"

            elif(ypred1 == 2):
                ans= "Positive"
                for i in range(len(positive)):
                    for j in range(len(negative)):
                        if(len(og_rev) == 1):
                            if(positive[i] in og_rev):
                                ans = "positive"
                            if(negative[j] in og_rev):
                                ans = "negative"
                        else:
                            if(positive[i] in og_rev):
                                pos = og_rev.index(positive[i])
                                for n in negative:
                                    if(og_rev[pos-1]==n or og_rev[pos-2]==n):
                                        ans="Negative"
                                for p in positive:
                                    if(og_rev[pos-1]==p or og_rev[pos-2]==n):
                                        ans = "POsitive"
        upd_rev.append([A,M,ans])

    return upd_rev

def Take_input():
    n = inputtxt.get("1.0", "end-1c")            
    ##n =input("Enter a reveiw\n")
    text = rev_clean(n)
    s_text = text.split()
    asp_des = find_sentiment(nlp(text))

    asp_des2 = ''
    for i in range(len(asp_des)):
        for j in range(len(negation)):
            if(len(asp_des) == 0) or (s_text[0] == negation[j] and len(asp_des) == 1):
                asp_des2 = find_sentiment2([text])
    if(asp_des2 != ''):
        print("IN ASP DES 2")
        for i in range(len(asp_des2)):
            asp = asp_des2[i][0]
            des = asp_des2[i][1]
            pol = asp_des2[i][2]
            print("ASPECT:-",asp)
            print("DESCRIPTION:-",des)
            print("POLARITY:-",pol,"\n")
            Output.insert(END,"Aspect:-"+asp+" ")
            Output.insert(END,"Description:-"+des+" ") 
            Output.insert(END,"Polarity:-"+pol+" ")
        Output.insert(END,"ACCURACY OF THE MODEL WAS "+str(acc))
    else:
        print("IN ASP DES")
        for i in range(len(asp_des)):
            asp = asp_des[i][0]
            des = asp_des[i][1]
            pol = asp_des[i][2]
            print("ASPECT:-",asp)
            print("DESCRIPTION:-",des)
            print("POLARITY:-",pol,"\n")
            Output.insert(END,"Aspect:-"+asp+" ")
            Output.insert(END,"Description:-"+des+" ")
            Output.insert(END,"Polarity:-"+pol+" ")
            Output.insert(END,"\n")
        Output.insert(END,"ACCURACY OF THE MODEL WAS "+str(acc))

  
f1=Frame(root,bg="cyan",borderwidth=10,relief=SUNKEN)
f1.pack(side=TOP,fill="x")
border_color = Frame(f1, background="black",borderwidth=3,relief=SUNKEN)
label = Label(border_color, text="ASPECT RANKING SENTIMENT ANALYSIS",bg="cyan",fg="black",font=("comicsansms",40,"bold"),padx=30,pady=30)
label.pack(padx=5, pady=5)
border_color.pack(padx=30, pady=30)

f2=Frame(root,bg="yellow",borderwidth=10,relief=SUNKEN)
f2.pack(side=TOP,fill="x")
label1 = Label(f2, text="ENTER A REVIEW",bg="yellow",fg="black",font=("comicsansms",40,"bold"),padx=25,pady=25)
label1.pack(padx=5, pady=5)

f3 = Frame(root,bg="red",borderwidth=10,relief=SUNKEN)
f3.pack(side=BOTTOM,fill="x")
label2 = Label(f3, text="OUTPUT SCREEN", bd=0,bg="red",fg="black",font=("comicsansms",40,"bold"),padx=20,pady=10)
label2.pack(padx=5, pady=5)
inputtxt = Text (f2,height=3,width =60,bg="black",fg="white",font=("comicsansms",20))
  
Output = Text (f3,height=12,width =60,bg="black",fg="white",font=("comicsansms",20))
  
Display = Button(f2,bg="black",fg="white",text="Submit Review",font=("comicsansms",15),pady=15,padx=30,command = lambda:Take_input())
b2 =Button(f2,bg="black",fg="white",text="Click for word clouds",font=("comicsansms",15),pady=15,padx=30,command = lambda:word_clouds())
b2.pack(side=LEFT,anchor="sw")
inputtxt.pack(fill="x",padx=50, pady=30)
Display.pack(padx=5)
Output.pack()
  
mainloop()
