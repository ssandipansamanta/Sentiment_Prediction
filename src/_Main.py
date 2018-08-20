Input_Path = path + 'Inputs/'
Output_Path = path + 'Outputs/'
Code_Path = path + 'src/'
Log_Path = path + 'Log/'

Format='.txt'

TrainDatasetName = ['imdb_labelled','yelp_labelled']
TestDatasetName = ['amazon_cells_labelled']

exec(open(Code_Path+"EnvironmentSetup.py").read(), globals())
exec(open(Code_Path+"DataPreparation.py").read(), globals())
exec(open(Code_Path+"WordCloud.py").read(), globals())
exec(open(Code_Path+"MLAlgo.py").read(), globals())