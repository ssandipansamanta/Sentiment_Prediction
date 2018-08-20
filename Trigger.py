from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False

def get_current_dir():
   import os
   return os.path.dirname(os.path.realpath(__file__))

def get_current_dir_unixy():
   return get_current_dir().replace('\\', '/')

def get_working_dir():
    import sys
    working_dir = get_current_dir_unixy()
    if not sys.platform.startswith('win'):
        working_dir = working_dir + '/'
    return working_dir

path = get_working_dir() + "/"
exec(open(path + 'src/' + '_Main.py').read(), globals())


@app.route("/NaiveBayes")
def NaiveBayesAPI():
    Score, CM = NaiveBayes(TrainDatasetName=TrainDatasetName, TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Naive_Bayes')
    return render_template('OutputTemplate.html', tables=[CM.to_html(classes='CM',index=False,index_names=False)],
                           titles = ['na',str('Prediction Score %.2f' %(Score*100) + ' %')])

@app.route("/Logistic")
def LogisticAPI():
    Score, CM = LogisticRegression(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Logistic_Reg')
    return render_template('OutputTemplate.html', tables=[CM.to_html(classes='CM',index=False,index_names=False)],
                           titles = ['na',str('Prediction Score %.2f' %(Score*100)  + ' %')])

@app.route("/RandomForest")
def RandomforestAPI():
    Score, CM = RandomForest(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Random_Forest')
    return render_template('OutputTemplate.html', tables=[CM.to_html(classes='CM',index=False,index_names=False)],
                           titles = ['na',str('Prediction Score %.2f' %(Score*100)  + ' %')])

@app.route("/GBM")
def GBMAPI():
    Score, CM = GBM(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Gradient_Boosting')
    return render_template('OutputTemplate.html', tables=[CM.to_html(classes='CM',index=False,index_names=False)],
                           titles = ['na',str('Prediction Score %.2f' %(Score*100)  + ' %')])

@app.route("/SVM")
def SVMAPI():
    Score, CM = SVM(TrainDatasetName=TrainDatasetName,TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Support_Vector_Machine')
    return render_template('OutputTemplate.html', tables=[CM.to_html(classes='CM',index=False,index_names=False)],
                           titles = ['na',str('Prediction Score %.2f' %(Score*100)  + ' %')])

@app.route("/NeuralNetwork")
def NeuralNetworkAPI():
    Score, CM = NeuralNetwork(TrainDatasetName=TrainDatasetName, TestDatasetName=TestDatasetName,Output_Path=Output_Path,MLName='Neural_Network')
    return render_template('OutputTemplate.html', tables=[CM.to_html(classes='CM',index=False,index_names=False)],
                           titles = ['na',str('Prediction Score %.2f' %(Score*100)  + ' %')])

from multiprocessing import Process, Pool
from itertools import repeat

if __name__ == "__main__":

    remove_files(Output_Path, '.pdf')
    remove_files(Output_Path, '.csv')

    DataSetName = TrainDatasetName + TestDatasetName
    pool = Pool(processes= len(DataSetName))
    pool.starmap(PreparingData,zip(repeat(Input_Path),DataSetName,repeat(Format),repeat(Output_Path)))

    app.run(host='0.0.0.0', port=5000, debug=False)
