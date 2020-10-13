# hack-a-thing-2: Playing around with NLP

## Description
For the local election helper, we will need to run some NLP models on the data to see what the candidates care about. The main idea is that we extract topics from the text, then find the sentences that refer to those topics, then we run sentiment analysis on those topics to see how the candidate feels about said topic. 

This is our attempt to learn some basic ML stuff and try out different models for sentiment analysis and topic extraction. We peer programmed a lot, using the VSCode liveshare feature. We began with a binary classificiation tutorial, which classified movie reviews as either positive or negative. 

Then we started topic extraction, starting with an LDA model, using a set of headlines from ABC. While doing topic extraction with this data set, we didn't think the classification was very good because it seemed like a lot of headlines got lumped into incorrect topics. We hypothosized that this was because the data set was too broad. We decided to try the same thing on a set of trump tweets, and found the topic extraction ran slightly more accurately. 

The last model Emma did, she played around with watson sentiment analysis. She ran sentiment analysis on a corpus corresponding to a predetermined set of topics.

We are planning to use these sentiment analysis tools on data we scrape to determine the stance politicians take on various issues. For local elections, small time politicians may not have an actual list of their stances posted, so it may be up to us to try to decipher what they think about important issues.

### What didn’t work
The ml models are not all that accurate as of yet. I am worried about this because we trained them on a large set of data. If we have even less data about small niche topics I fear the models will become even less accurate.
## Authors
Grace Dorgan and Emma Rafkin
