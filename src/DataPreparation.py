
def PreparingData(Input_Path,FileName,Format,Output_Path):

    # FileName = 'imdb_labelled'
    # CommonWordsCount=10, RareWordsCount=10
    # try:
    RawData = pd.read_csv(Input_Path + FileName + Format, sep='\\t', header=None)
        # logger.info('--- Reading Files Complete')
    # except ImportError:
        # logger.info('[ERROR]: Loading Files Failed')

    # logger.info('--- Setting up Col Names')
    RawData.columns = ['User_Feedback','Sentiment']

    # logger.info('--- Distribution of Sentiments')
    CountTable = pd.crosstab(index=RawData["Sentiment"], columns="count")
    DistSentiments = CountTable/CountTable.sum()

    # try:
    Exportingcsv(DataSet=DistSentiments, OutputPath= Output_Path + 'DistSentiments_' + FileName +'.csv', EncodingStyle='utf-8', indexingFlag=False)
         # logger.info('--- Exporting File Successful')
    # except ImportError:
         # logger.info('[ERROR]: Exporting File Failed')

    del CountTable, DistSentiments
    gc.collect()

    # RawData['Lengh_Sentence'] = RawData['User_Feedback'].str.len()#apply(lambda x: len(str(x).split(" ")))
    # logger.info('--- Converting to lower case')
    RawData['User_Feedback'] = RawData['User_Feedback'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # logger.info('--- Removing')
    RawData['User_Feedback'] = RawData['User_Feedback'].str.replace('[^\w\s]', '')

    # logger.info('--- Removing Numerical Values')
    RawData['User_Feedback'] = RawData['User_Feedback'].apply(lambda x: " ".join(re.sub('\d', ' ', x) for x in x.split()))

    # logger.info('--- Removing Stop Words')
    RawData['User_Feedback'] = RawData['User_Feedback'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords.words('english')))

    # Time consuming, so removing from implementation
    # from textblob import TextBlob
    # RawData['User_Feedback'].apply(lambda x: str(TextBlob(x).correct()))

    # logger.info('--- Converts Words into its Root Word')
    RawData['User_Feedback'] = RawData['User_Feedback'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    RawData['FileName'] = FileName

    # This step might be useful, need to check if removing Rarewords can improve the prediction
    # CommonWordsCount = 10;RareWordsCount = 10
    # CommonWords = pd.Series(' '.join(RawData['User_Feedback']).split()).value_counts()[:CommonWordsCount]
    # RareWords = freq = pd.Series(' '.join(RawData['User_Feedback']).split()).value_counts()

    Exportingcsv(DataSet=RawData, OutputPath= Output_Path + 'RTM_' + FileName + '.csv', EncodingStyle='utf-8', indexingFlag=False)

    PositiveSentiment = RawData[RawData.Sentiment == 1].reset_index(drop=True)
    WordCloud(Output_Path=Output_Path,FileName=FileName,data=PositiveSentiment,type='Positive')

    NegativeSentiment = RawData[RawData.Sentiment == 0].reset_index(drop=True)
    WordCloud(Output_Path=Output_Path,FileName=FileName,data=NegativeSentiment,type='Negative')

    del PositiveSentiment, NegativeSentiment, RawData
    gc.collect()

    return None