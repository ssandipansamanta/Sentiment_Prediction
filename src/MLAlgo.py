def ReadRTMDataset(TrainDName,TestDName,Path):
    TrainDataset = pd.DataFrame()
    for d in TrainDName:
        _tempd = pd.read_csv(Path + 'RTM_' + d + '.csv', encoding='utf-8')
        _tempd = _tempd[_tempd['User_Feedback'].notnull()]
        TrainDataset = TrainDataset.append(_tempd)

    TestDataset = pd.read_csv(Path + 'RTM_' + TestDName[0] + '.csv', encoding='utf-8')
    return TrainDataset, TestDataset

def TFiDfTransform(TRData,TSData):
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()

    X_TrCounts = count_vect.fit_transform(TRData['User_Feedback'])
    count_vect.vocabulary_.get(u'algorithm')
    X_TR_TFiDf = tfidf_transformer.fit_transform(X_TrCounts)

    X_TSCounts = count_vect.transform(list(TSData['User_Feedback']))
    X_TS_TFiDf = tfidf_transformer.transform(X_TSCounts)

    return X_TR_TFiDf, X_TS_TFiDf

def TrainTestDataPreparation(TrainDatasetName,TestDatasetName,Output_Path):
    TrainDataSet, TestDataSet = ReadRTMDataset(TrainDName = TrainDatasetName, TestDName = TestDatasetName, Path = Output_Path)
    Train, Test = TFiDfTransform(TRData=TrainDataSet,TSData = TestDataSet)
    X_Train = Train;Y_Train = TrainDataSet["Sentiment"].tolist()
    X_Test = Test; Y_Test = TestDataSet["Sentiment"].tolist()
    return X_Train, Y_Train, X_Test, Y_Test

def AccuracyMeasure(YTest, PredictedSeries,Method):
    Score = accuracy_score(YTest, PredictedSeries)
    ConfusionMetrics = pd.DataFrame(metrics.confusion_matrix(YTest, PredictedSeries))
    ConfusionMetrics.columns = ['Predicted Un-happy','Predicted Happy']
    ConfusionMetrics['Method'] = Method
    ConfusionMetrics['Actual'] = ['Actual Un-happy', 'Actual Happy']
    ConfusionMetrics = ConfusionMetrics[['Method','Actual','Predicted Un-happy','Predicted Happy']].reset_index(drop=True)
    return Score, ConfusionMetrics

def NaiveBayes(TrainDatasetName,TestDatasetName,Output_Path,MLName):

    X_Train, Y_Train, X_Test, Y_Test = TrainTestDataPreparation(TrainDatasetName = TrainDatasetName,
                                                                TestDatasetName = TestDatasetName,
                                                                Output_Path = Output_Path)
    from sklearn.naive_bayes import BernoulliNB
    ML = BernoulliNB(alpha=0.0, # required for multinomial distribution
                     binarize=0.0, fit_prior=True, class_prior=None)

    ML.fit(X_Train, Y_Train); PS = ML.predict(X_Test)
    Score,ConfustionMatrix = AccuracyMeasure(YTest=Y_Test, PredictedSeries=PS,Method=MLName)
    Exportingcsv(DataSet=ConfustionMatrix, OutputPath=Output_Path + 'CM_' + MLName + '.csv',
                 EncodingStyle='utf-8', indexingFlag=False)

    return Score,ConfustionMatrix
# NaiveBayes(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Naive Bayes')

def LogisticRegression(TrainDatasetName,TestDatasetName,Output_Path,MLName):
    X_Train, Y_Train, X_Test, Y_Test = TrainTestDataPreparation(TrainDatasetName=TrainDatasetName,
                                                                TestDatasetName=TestDatasetName,
                                                                Output_Path=Output_Path)
    from sklearn.linear_model import LogisticRegression
    ML = LogisticRegression(penalty='l2',dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None, random_state=12345,
                            solver='lbfgs', max_iter=500, multi_class='ovr', verbose=0,
                            warm_start=False, n_jobs=-1)
    ML.fit(X_Train, Y_Train);
    PS = ML.predict(X_Test)
    Score, ConfustionMatrix = AccuracyMeasure(YTest=Y_Test, PredictedSeries=PS, Method=MLName)
    Exportingcsv(DataSet=ConfustionMatrix, OutputPath=Output_Path + 'CM_' + MLName + '.csv',
                 EncodingStyle='utf-8',indexingFlag=False)

    return Score, ConfustionMatrix

# LogisticRegression(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Logistic Reg')

def RandomForest(TrainDatasetName,TestDatasetName,Output_Path,MLName):
    X_Train, Y_Train, X_Test, Y_Test = TrainTestDataPreparation(TrainDatasetName=TrainDatasetName,
                                                                TestDatasetName=TestDatasetName,
                                                                Output_Path=Output_Path)
    from sklearn.ensemble import RandomForestClassifier
    ML = RandomForestClassifier(n_estimators = 1500, criterion ='gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 8,
                                min_weight_fraction_leaf = 0.0, max_features ='auto', max_leaf_nodes = 15, min_impurity_decrease = 0.0,
                                min_impurity_split = None, bootstrap = True, oob_score = False, n_jobs = -1, random_state = 12345,
                                verbose = 0, warm_start = False, class_weight = None)
    ML.fit(X_Train, Y_Train);
    PS = ML.predict(X_Test)
    Score, ConfustionMatrix = AccuracyMeasure(YTest=Y_Test, PredictedSeries=PS, Method=MLName)
    Exportingcsv(DataSet=ConfustionMatrix, OutputPath=Output_Path + 'CM_' + MLName + '.csv',
                 EncodingStyle='utf-8',indexingFlag=False)

    return Score, ConfustionMatrix
# RandomForest(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Random Forest')

def GBM(TrainDatasetName,TestDatasetName,Output_Path,MLName):
    X_Train, Y_Train, X_Test, Y_Test = TrainTestDataPreparation(TrainDatasetName=TrainDatasetName,
                                                                TestDatasetName=TestDatasetName,
                                                                Output_Path=Output_Path)
    from sklearn.ensemble import GradientBoostingClassifier

    ML = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=500, subsample=1.0,criterion='friedman_mse',
                                    min_samples_split=2, min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_depth=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=12345,
                                    max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

    ML.fit(X_Train, Y_Train);
    PS = ML.predict(X_Test)
    Score, ConfustionMatrix = AccuracyMeasure(YTest=Y_Test, PredictedSeries=PS, Method=MLName)
    Exportingcsv(DataSet=ConfustionMatrix, OutputPath=Output_Path + 'CM_' + MLName + '.csv',
                 EncodingStyle='utf-8',indexingFlag=False)

    return Score, ConfustionMatrix
# GBM(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Gradient Boosting')

def SVM(TrainDatasetName,TestDatasetName,Output_Path,MLName):
    X_Train, Y_Train, X_Test, Y_Test = TrainTestDataPreparation(TrainDatasetName=TrainDatasetName,
                                                                TestDatasetName=TestDatasetName,
                                                                Output_Path=Output_Path)
    from sklearn.svm import SVC
    ML = SVC(C=0.5, kernel='linear', degree=0, gamma='auto', coef0=0.0, shrinking=True, probability=True,tol=0.00001,
              cache_size=500, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ov0', random_state=12345)
    ML.fit(X_Train, Y_Train);
    PS = ML.predict(X_Test)
    Score, ConfustionMatrix = AccuracyMeasure(YTest=Y_Test, PredictedSeries=PS, Method=MLName)
    Exportingcsv(DataSet=ConfustionMatrix, OutputPath=Output_Path + 'CM_' + MLName + '.csv',
                 EncodingStyle='utf-8',indexingFlag=False)

    return Score, ConfustionMatrix

# SVM(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Support Vector Machine')

def NeuralNetwork(TrainDatasetName,TestDatasetName,Output_Path,MLName):
    X_Train, Y_Train, X_Test, Y_Test = TrainTestDataPreparation(TrainDatasetName=TrainDatasetName,
                                                                TestDatasetName=TestDatasetName,
                                                                Output_Path=Output_Path)
    from sklearn.neural_network import MLPClassifier
    ML = MLPClassifier(hidden_layer_sizes=(200, ), activation='logistic', solver= 'lbfgs', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=None, max_iter=500, shuffle=True, random_state=12345,
                       tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    ML.fit(X_Train, Y_Train);
    PS = ML.predict(X_Test)
    Score, ConfustionMatrix = AccuracyMeasure(YTest=Y_Test, PredictedSeries=PS, Method=MLName)
    Exportingcsv(DataSet=ConfustionMatrix, OutputPath=Output_Path + 'CM_' + MLName + '.csv',
                 EncodingStyle='utf-8',indexingFlag=False)

    return Score, ConfustionMatrix

# NeuralNetwork(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Neural Network')
