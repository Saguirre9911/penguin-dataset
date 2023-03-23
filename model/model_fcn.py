import pickle
from pathlib import Path
import numpy as np
import json


BASE_DIR= Path(__file__).resolve(strict=True).parent
answer_list=[]
pred_list=[]

with(open(f"{BASE_DIR}/svm_clas_penguins.pkl", 'rb')) as f:
    pickled_model=pickle.load(f)


classes= [
    "Adelie", 
    "Gentoo",
    "Chinstrap"
]

def predict_fcn(X_test):
    """
    Predicts if a tumor is bening or malignant base on the input data, then creates a diccionary based on 
    the amount of predictions and the diagnosis to write a json file ready to be upload into a Database
    Arguments:
    X_test: list[] of 30 features in string format used as input of the ML model. 
    """
    X_test=np.array(X_test)
    X_test= X_test.reshape(1, -1)
    print(X_test)
    y_pred=pickled_model.predict(X_test)
    id=0
    for element in y_pred:
        answer=classes[element]
        answer_list.append(answer)
        pred_list.append("diagnosis "+ str(id+1))
        id=id+1
    answer= 'The penguin is an: '+ answer_list[0]

    return answer
