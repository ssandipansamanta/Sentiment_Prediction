Sentiment-Prediction
=======================

# Description

The goal is to predict sentiment -- the emotional intent behind a statement -- from text. 
For example, the sentence: "This movie was terrible!" has a negative sentiment, 
whereas "loved this cinematic masterpiece" has a positive sentiment.

To simplify the task, It's considered sentiment binary: labels of 1 indicate a sentence has a positive sentiment, and 
labels of 0 indicate that the sentence has a negative sentiment.

# Dataset

The dataset is split across three files, representing three different sources -- Amazon, Yelp and IMDB. 
Your task is to build a sentiment analysis model using both the Yelp and IMDB data as your training-set, 
and test the performance of your model on the Amazon data.

Each file can be found in the ../input directory, and contains 1000 rows of data. Each row contains a sentence, a tab character and then a label -- 0 or 1.


### __For the full datasets__

* imdb: Maas et. al., 2011 'Learning word vectors for sentiment analysis'
* amazon: McAuley et. al., 2013 'Hidden factors and hidden topics: Understanding rating dimensions with review text'
* yelp: Yelp dataset challenge http://www.yelp.com/dataset_challenge

# Steps to Run the Code Base
```xml
1. Run 'Trigger.py' using CMD
2. First Data Preparation will start, it will take few minutes
3. Wait till you see the commands like 'Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)'
4. Open 'SentimentAnalysisAPI.html' and Click the different ML to get the scores.
5. You can interrupt by pressing CTRL+C in CMD.
```

# Set up Structure

* Input Files would be kept in Input folder.
* Output Files would be saved in Output folder automatically.
	Mainly Four types of output have been saved.
	* Distribution of Sentiments accross all files
	* Ready to Model(RTM) Data set for all files.
	* WordCloud for all the data sets by positive/negative sentiments
	* Confusion Matrix for all the ML-algos (after clicking from the UI)
* src contains all source codes
	* _Main.py: Calling all the function
	* DataPreparation.py: Preparing Text for Modeling
	* EnvironmentSetup.py: Loading all Packages and few user defined function.
	* MLAlgo.py: Transform TFiDF and apply NB,Logistic, RF,GBM,SVM,NNet for prediction
	* WordCloud.py: Basic Data Checks for Train and Test data set for +/- sentiments.
* static and templates are for html output folder
* Codes are refactored and loosely coupled, Will not be very much difficult to change into another language for a specific module. 

# Methodology/Logic Flow

* By Triggering 'Trigger.py', it will first delete all the existing csv and pdf outputs present in output folder.
* Run '_Main.py' to get input/output/src folder path as well as train and test data set names.
* '_Main.py' also call DataPreparation, EnvironmentSetup, WordCloud and MLAlgo python codes.
* Based on Number of Train data set, multiprocessing has been started to prepare RTM data set. In this exercise, three process would be triggered i.e. for imdb, yelp and Amazon each.
* After finishing each process, they will save RTM data sets in output folder and also save wordcloud separately for +/- sentiments.
* After creating RTM data set, user needs to click on specific algo from UI to get the predictive answer. Howerver, the process would be same for all the algos.
	e.g. if user select Naive Bayes, All Train Data sets would read in python memory and append all into a single data set. Test Data set also read at the same time.
* TFiDF transformation would be applied on Train and Test Data sets.
* Fit the model using Train Data sets(i.e. imdb and yelp) and predict for the Test Data set(i.e.Amazon)
* Predict accuracy and compute confusion Matrix for the same. Confusion matrix will be saved in Output folder.
	
# Future Work

* Deep Learning Needs to be applied and gain more accuracy.
* Combine all the algos to get more roubust prediction viz. max voting, Stacking etc.
* Error Logging to have production level readyness.

# Reading Materials
  From Group to Individual Labels using Deep Features, _Kotzias et. al,. KDD_ 2015