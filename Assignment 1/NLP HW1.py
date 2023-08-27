#Using Python3

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import nltk
import re
from sklearn import metrics
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

dataframe = pd.read_csv(r"data.tsv",sep="\t",usecols = ["review_body","star_rating"])


ratingDict = {1:1, 2:1, 3:2, 4:3, 5:3}
filteredDataFrame = dataframe.replace({"star_rating": ratingDict})
filteredDataFrame = filteredDataFrame.sample(frac=1, random_state=14).reset_index(drop=True)
filteredDataFrame


contractions_dict = { 
"ain't": "are not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i've": "i have"
}


rating1Data = filteredDataFrame[filteredDataFrame['star_rating']==1].head(20000)
rating2Data = filteredDataFrame[filteredDataFrame['star_rating']==2].head(20000)
rating3Data = filteredDataFrame[filteredDataFrame['star_rating']==3].head(20000)

finalRatingsData = rating1Data.append(rating2Data).append(rating3Data).reset_index()

finalRatingsData['review_body'] = finalRatingsData["review_body"].apply(str) #converts review_body column to string type

averageStringLenBeforeDataCleaning, stringLength = 0, 0
for ratings in finalRatingsData['review_body']:
  stringLength += len(ratings)

averageStringLenBeforeDataCleaning = stringLength / len(finalRatingsData['review_body'])

def cleanReviewData(column):
  column = column.lower() #converts string to lowercase
  column = re.sub(r'\d+','',column) #remove numerical characters from the string
  column = re.sub(r'[^\w\s]','', column) #removes punctuations from the string
  column = column.strip() #remove extra spaces from the string
  column = re.sub(r"http\S+", "", column) #removes URLs from the string
  colummn = re.sub(re.compile('<.*?>'), '', column) #removes HTML tags from the string

  return column


finalRatingsData['review_body'] = finalRatingsData["review_body"].apply(cleanReviewData)

for reviews in finalRatingsData['review_body']:
  review = reviews.split()
  for char in review:
    if char in contractions_dict:
      reviews = reviews.replace(char, contractions_dict[char])


averageStringLengAfterDataCleaning, stringLength = 0, 0
for ratings in finalRatingsData['review_body']:
  stringLength += len(ratings)

averageStringLengAfterDataCleaning = stringLength / len(finalRatingsData['review_body'])

averageStringLengBeforeDataPreprocessing, stringLength = 0, 0
for ratings in finalRatingsData['review_body']:
  stringLength += len(ratings)

averageStringLengBeforeDataPreprocessing = stringLength / len(finalRatingsData['review_body'])

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
finalRatingsData['review_body'] = finalRatingsData['review_body'].apply(lambda key: ' '.join([word for word in key.split() if word not in (stop_words)]))


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])
finalRatingsData['review_body'] = finalRatingsData['review_body'].apply(lemmatize_text)

averageStringLengAfterDataPreprocessing, stringLength = 0, 0
for ratings in finalRatingsData['review_body'].to_list():
  stringLength += len(ratings)

averageStringLengAfterDataPreprocessing = stringLength / len(finalRatingsData['review_body'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidvectorizer = TfidfVectorizer()
x = tfidvectorizer.fit_transform(finalRatingsData['review_body'])


from sklearn.model_selection import train_test_split
part1 = finalRatingsData['review_body'].to_list()
part2 = finalRatingsData['star_rating'].to_list()
part1_train, part1_test, part2_train, part2_test = train_test_split(part1, part2, test_size=0.2, shuffle = True)

Train_X_Tfidf = tfidvectorizer.transform(part1_train)
Test_X_Tfidf = tfidvectorizer.transform(part1_test)

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,part2_train)
predictions_NB = Naive.predict(Test_X_Tfidf)

svm = LinearSVC()
svm.fit(Train_X_Tfidf,part2_train)
predictions_svm = svm.predict(Test_X_Tfidf)

perceptron = Perceptron()
perceptron.fit(Train_X_Tfidf,part2_train)
predictions_perp = perceptron.predict(Test_X_Tfidf)

logisticRegression = LogisticRegression(max_iter=1000)
logisticRegression.fit(Train_X_Tfidf,part2_train)
predictions_log = logisticRegression.predict(Test_X_Tfidf)

print(str(averageStringLenBeforeDataCleaning) + "," + str(averageStringLengAfterDataCleaning))
print(str(averageStringLengBeforeDataPreprocessing) + "," + str(averageStringLengAfterDataPreprocessing))

predictions_perp_output = metrics.classification_report(predictions_perp, part2_test, output_dict=True)
for key, val in predictions_perp_output.items():
    if key == "accuracy" or key == "macro avg":
        continue
    if key == "1":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "2":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "3":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "weighted avg":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
        
predictions_svm_output = metrics.classification_report(predictions_svm, part2_test, output_dict=True)
for key, val in predictions_svm_output.items():
    if key == "accuracy" or key == "macro avg":
        continue
    if key == "1":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "2":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "3":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "weighted avg":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))


predictions_log_output = metrics.classification_report(predictions_log, part2_test, output_dict=True)
for key, val in predictions_log_output.items():
    if key == "accuracy" or key == "macro avg":
        continue
    if key == "1":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "2":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "3":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "weighted avg":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))


predictions_NB_output = metrics.classification_report(predictions_NB, part2_test, output_dict=True)
for key, val in predictions_NB_output.items():
    if key == "accuracy" or key == "macro avg":
        continue
    if key == "1":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "2":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "3":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))
    elif key == "weighted avg":
        print(str(val['precision']) + "," + str(val['recall']) + "," + str(val['f1-score']))