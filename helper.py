import numpy as np
import pandas as pd
import re
import string
import pickle
import emoji

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

# load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# load tokens
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def convert_emojis_to_words(text): 
    # Convert emojis to words using emoji.demojize
    text_with_emojis_as_words = emoji.demojize(text, delimiters=(' ', ' '))
    return text_with_emojis_as_words

def preprocessing(sentences):
    preprocessed_sentences = []

    for text in sentences:
        data = pd.DataFrame([text], columns=['Full_text'])
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        data["Full_text"] = data['Full_text'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
        data['Full_text'] = data['Full_text'].apply(convert_emojis_to_words)
        data["Full_text"] = data["Full_text"].apply(remove_punctuations)
        data["Full_text"] = data['Full_text'].str.replace('\d+', '', regex=True)
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
        data["Full_text"] = data["Full_text"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
        preprocessed_sentences.append(data["Full_text"].iloc[0])

    return preprocessed_sentences

def vectorizer(ds, vocabulary):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new
# Define the thresholds for categorization
negative_threshold = 0.4
positive_threshold = 0.6

# Categorize the results
def categorize(probability):
    if probability < negative_threshold:
        return 'negative'
    elif negative_threshold <= probability < positive_threshold:
        return 'neutral'
    else:
        return 'positive'
    
def get_prediction(vectorized_text):
    vectorized_text = vectorized_text.reshape(1, -1)
    prediction_score = model.predict_proba(vectorized_text)
    return prediction_score
