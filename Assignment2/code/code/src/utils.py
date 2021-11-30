import time
import pickle
 
from datetime import datetime

def get_timestamp():
    return datetime.now()


def get_time():
    # datetime object containing current date and time
    now = datetime.now()

    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    
    return dt_string

def save(obj,name,mode="outputs"):
    """ Save an object as pickle
    Parameter:
        obj  : Object to save
        name : Filename
        mode : Where to save it inputs/output  
    """
    pickle.dump(obj,open( mode+"/"+name+".p", "wb" ))
from sklearn import metrics
def print_result(y_true,y_test_pre):
    result={}
    cf=metrics.confusion_matrix(y_true, y_test_pre)
    print(cf)
    tn, fp, fn, tp = cf.ravel()

    sums = tn + fp + fn + tp
    print("tn:{},fp:{},fn:{},tp:{}".format(tn / sums, fp / sums, fn / sums, tp / sums))
    TPR = tp / (tp + fn)
    FPR = fp / (tn + fp)
    mark = (5 + 0.1) / (TPR - FPR)
    precision=metrics.precision_score(y_true, y_test_pre)
    Recall=metrics.recall_score(y_true, y_test_pre)
    F1_score=metrics.f1_score(y_true, y_test_pre)
    ACC=metrics.accuracy_score(y_true, y_test_pre)
    AUC_score=metrics.roc_auc_score(y_true, y_test_pre)
    result={"TPR":TPR,
            "FPR":FPR,
            "mark":mark,
            "Precision":precision,
            "Recall":Recall,
            "F1_score":F1_score,
            "ACC":ACC,
            "AUC_score":AUC_score}
    return result
def load(name,mode="inputs"):
    """ Loads an object from pickle
    Parameters:
        name: Filename
        mode: Where to load it
    """
    return pickle.load( open( "../"+mode+"/"+name+".p", "rb" ) )
    
