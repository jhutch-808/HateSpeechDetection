#Ensure you pip install
#TODO: N-grams 
#Stop word removal and tokenization
#Add any extra text cleaning wanted
#Save to the CSV when all checked so that people can do work in other files
#Currently only works with the cleaned dataset. Lots more work needed for other one
#Get rid of other languages

import pandas as pd
import numpy as np 
import re

pd.set_option('display.max_colwidth', None)

#Replace with your own path
f = open("../data/HateSpeechDataset.csv",'r')
r_cols = ['tweet', 'hate', 'nums']
tweets = pd.read_csv(f, sep=',', names=r_cols)
tweets = tweets[1:]

word_dict = dict()
word_counter = 0

#This is only in english, not sure how many languages are in there...
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
              "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", 
              "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", 
              "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", 
              "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
              "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
              "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
              "against", "between", "into", "through", "during", "before", "after", 
              "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
              "under", "again", "further", "then", "once", "here", "there", "when", "where", 
              "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
              "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
              "very", "s", "t", "can", "will", "just", "don", "should", "now"]

counter2 = [0,0,0,0,0,0]
nGrams = [dict(),dict(),dict(),dict(),dict(),dict()]
#Couple potential additions: 
#Add other domains like .info, .uk, .de etc. There are a couple instances in data where would be nice
#Hyper text transfer protocol seems to Start many of these urls. Probably shouldnt matter because we use naive bayes
#Also, like iloc 532 shows, there are other url components that this doesnt remove, but i dont want to kill the whole tweet
#Any ideas to fix this are appreciated

#Currently just removes urls
def clean_text(text):
    clean_tweet = text + " "
    clean_tweet = re.sub(" http| https", "", clean_tweet)

    clean_tweet = re.sub("hyper text transfer protocol", "", clean_tweet)

    clean_tweet = re.sub("www .* com ", "", clean_tweet)
    clean_tweet = re.sub("www .* org ", "", clean_tweet)
    clean_tweet = re.sub("www .* net ", "", clean_tweet)
    clean_tweet = re.sub("www .* uk ", "", clean_tweet)

    clean_tweet = re.sub(" $", "", clean_tweet)
    clean_tweet = clean_tweet.lower()
    return clean_tweet

#Uses hash table to go through the tweets and assign numbers pretty fast
#Honestly pretty simple, should do with and without stop words for testing

def number_words(text):
    #Im being told by online people it is bad design to use global variables
    #Feel free to fix as wanted
    global word_dict
    global word_counter
    new_numbers = []
    for i in text:
        if i in word_dict:
            new_numbers.append(word_dict[i])
        else:
            word_dict[i] = word_counter
            word_counter = word_counter + 1
            new_numbers.append(word_dict[i])
    return new_numbers

#Efficiency will be added. JAMES, you must think...
#perhaps another hash table.

def stop_word_removal(text):
    return_text = []
    global stop_words
    for i in range(len(text)):
        if not text[i] in stop_words:
            return_text.append(text[i])
            
    return return_text

#This one is for making n-grams. Might be moved around to after text preperation
#text is a list of numbers so be cool about it
#Supports up to n = 6

def nGrammify(text_nums, n):
    global nGrams
    global counter2
    myList = []
    for i in range(len(text_nums)-n+1):
        nums = ""
        for j in range(n):
            nums = nums + "^" + str(text_nums[i+j])
        if nums in nGrams[n-1]:
            myList.append(nGrams[n-1][nums])
        else:
            nGrams[n-1][nums] = counter2[n-1]
            counter2[n-1] = counter2[n-1]+1
            myList.append(nGrams[n-1][nums])
    return myList


tweets['tweet'] = tweets['tweet'].apply(clean_text)
del tweets['nums']

tweets['splits'] = tweets['tweet'].str.split()
#Numbers words
tweets['newNum'] = tweets['splits'].apply(number_words)
#This takes a while.
tweets['noStop'] = tweets['splits'].apply(stop_word_removal)
tweets['numStop'] = tweets['noStop'].apply(number_words)
#this takes a while too. Not too long that I couldnt be convinced to ditch dictionaries
tweets['2_grams'] = tweets['newNum'].apply(lambda x: nGrammify(x, 2))
tweets['3_grams'] = tweets['newNum'].apply(lambda x: nGrammify(x, 3))
#NS = no stop
tweets['2_grams_NS'] = tweets['numStop'].apply(lambda x: nGrammify(x, 2))
tweets['3_grams_NS'] = tweets['numStop'].apply(lambda x: nGrammify(x, 3))

#We should only run this line once are done with everything we want to do
tweets.to_csv("finished_hate_speech.csv")
