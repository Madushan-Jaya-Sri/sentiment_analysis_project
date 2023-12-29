from flask import Flask, render_template, request
from helper import preprocessing, vectorizer, get_prediction,categorize, tokens
from logger import logging
import pandas as pd

app = Flask(__name__)

logging.info('Flask server started')

data = dict()

positive = 0
negative = 0
neutral = 0 

@app.route("/")
def index():
    data['positive'] = positive
    data['neutral'] = neutral
    data['negative'] = negative

    # Create an empty DataFrame
    output_df = pd.DataFrame(columns=['Sentence', 'Sentiment'])

    logging.info('========== Open home page ============')

    return render_template('index.html', data=data, output_df=output_df)

@app.route("/", methods=['post'])
def my_post():
    sentences = request.form['text'].split('\n')  # Split sentences by newline
    logging.info(f'Text : {sentences}')

    rows = []

    for sentence in sentences:
        if sentence.strip():  # Check if the sentence is not empty or contains only whitespace
            preprocessed_sentence = preprocessing([sentence])
            logging.info(f'Preprocessed Text : {preprocessed_sentence}')

            vectorized_sentence = vectorizer(preprocessed_sentence, tokens)
            logging.info(f'Vectorized Text : {vectorized_sentence}')

            prediction_scores = get_prediction(vectorized_sentence)

            prediction = categorize(prediction_scores[0, 1])
            logging.info(f'Prediction : {prediction}')
 
            rows.append({'Sentence': sentence, 'Sentiment': prediction})

    # Create a DataFrame
    output_df = pd.DataFrame(rows)
    global negative, positive ,neutral
            
    print(output_df['Sentiment'].value_counts())

    output = output_df['Sentiment'].value_counts()
    positive = output.reset_index().iloc[:1,1:]
    neutral = output.reset_index().iloc[1:2,1:]
    negative = output.reset_index().iloc[2:3,1:]
    
    # positive = output.get('positive', 0)  # Using get method to handle the case when key is not present
    # neutral = output.get('neutral', 0)
    # negative = output.get('negative', 0)

    print(positive)
    print(neutral)
    print(negative)

    
    negative, neutral , positive = 0, 0, 0  # Reset counters

    for prediction in output_df['Sentiment']:
        if prediction == 'negative':
            negative += 1
        elif prediction == 'neutral' :
            neutral += 1
            
        else:
            positive+=1
  
    return render_template('index.html', data=data, output_df=output_df, output = output, positive= positive, negative = negative, neutral = neutral)

if __name__ == "__main__":
    app.run()
