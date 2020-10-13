# We will run sentiment analysis on a corpus corresponding to a predetermined set of topics.
# We are using watson sentiment analysis because it appears to be the best in the business
# (better than textblob and VADER) according to this article (https://medium.com/@Intellica.AI/vader-ibm-watson-or-textblob-which-is-better-for-unsupervised-sentiment-analysis-db4143a39445)
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, CategoriesOptions
from dotenv import load_dotenv

import os
import numpy as np
load_dotenv()
authenticator = IAMAuthenticator(os.environ.get("WATSON_API_KEY"))

natural_language_understanding = NaturalLanguageUnderstandingV1(                                         
    version='2020-08-01',
    authenticator=authenticator
    )
natural_language_understanding.set_service_url("https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/fca4023a-1ac4-483c-b14e-92bb3579cc48")

def Sentiment_score(input_text): 
    # Input text can be sentence, paragraph or document
    response = natural_language_understanding.analyze (
    text = input_text,
    features = Features(sentiment=SentimentOptions())).get_result()
    # From the response extract score which is between -1 to 1
    res = response.get('sentiment').get('document').get('score')
    return res

def get_sentiment_on_speech(speech_array, topics_to_analyze):
    organized_topics = {}
    topic_scores = {}
    for sentence in speech_array:
        if len(sentence) > 10:
            for topic in topics_to_analyze.keys():
                for word in topics_to_analyze[topic]:
                    if word in sentence:
                        if topic not in organized_topics:
                            organized_topics[topic] = []
                        organized_topics[topic].append(sentence)

    for topic in organized_topics:
        for sentence in organized_topics[topic]:
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(Sentiment_score(sentence))
    print(topic_scores)
    median_scores = {}
    for topic in topic_scores:
         median_scores[topic] = np.median(topic_scores[topic])
    return median_scores

topics_to_analyze = {
    "guns": ["gun", "rifle", "protect", "police", "officer"],
    "economy": ["tax", "rates", "money", "economy", "growth", "relief"],
    "school": ["education", "college", "elementary", "opportunity"],
    "climate": ["climate", "environment", "global warming", "ecosystem", "fire"],

}

f = open('./inslee_speech.txt', "r")
inslee = f.read()
inslee = inslee.split(".")
inslee_sentiment = get_sentiment_on_speech(inslee, topics_to_analyze)
print(inslee_sentiment)
