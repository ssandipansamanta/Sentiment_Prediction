def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

results = map(install_and_import, ('pandas', 'numpy','nltk','wordcloud','logging','datetime'))
set(results)

def find_filenames( path_to_dir, suffix):
    from os import listdir
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix) ]

def remove_files(path_name, suffix):
    import os
    files_num = len(find_filenames(path_name, suffix))
    for file in find_filenames(path_name, suffix):
        os.remove(os.path.join(path_name, file))

def Exportingcsv(DataSet,OutputPath,EncodingStyle,indexingFlag):
    DataSet.to_csv(OutputPath,encoding=EncodingStyle,index=indexingFlag)

import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from textblob import Word
import gc

from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn import metrics