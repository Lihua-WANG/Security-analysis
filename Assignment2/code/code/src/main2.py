import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import from_model
from sklearn import metrics
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.decomposition import PCA

pca_reducer = PCA(n_components=5)
minMaxer = preprocessing.MinMaxScaler(feature_range=(0,1))
# 对特征进行PCA特征提取
cols_name_str=[]
def feature_split(feature_input):
    feature_hot=feature_input[cols_name_str]
    feature_hot=pd.get_dummies(feature_hot)
    feature_input.drop(cols_name_str,axis=1,inplace=True)
    feature_input=pd.concat([feature_input,feature_hot],axis=1)
    # feature_input=feature_input.join(feature_hot)
    return feature_input

if __name__=="__main__":
    org_name = ["timestamp", "duration", "protocol", "sourceIP", "sourcePort", "direction"
        , "destinationIP", "destinationPort", "state", "STS", "DTS", "totalPackage",
                "totalBytes_Bidirection", "totalBytes_single_direction","label"]
    datas_test_org = pd.read_csv(r"data\A2_2\test_data_with_labels", header=None, names=org_name)
    datas_test_org["stream_ID"]=pd.Series(list(range(0,datas_test_org.shape[0])))


    data_train= pd.read_csv(r"data\A2_2\1_train.csv",index_col=0)
    data_train_ip=data_train["sourceIP"]
    data_train.drop("sourceIP",inplace=True,axis=1)

    data_val = pd.read_csv(r"data\A2_2\1_val.csv",index_col=0)
    data_val_ip=data_val["sourceIP"]
    data_val.drop("sourceIP",inplace=True,axis=1)

    data_test = pd.read_csv(r"data\A2_2\1_test.csv",index_col=0)
    data_test_ip=data_test["sourceIP"]
    data_test.drop("sourceIP",inplace=True,axis=1)

    data_train_label=data_train["label"].values
    data_train.drop("label",axis=1,inplace=True)

    data_val_label=data_val["label"].values
    data_val.drop("label",axis=1,inplace=True)

    data_test_label=data_test["label"].values
    data_test.drop("label",axis=1,inplace=True)

    print(data_train.head())

    data_train.hist(figsize=(8, 8))
    # 数据显示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)
    plt.show()

    cols_name_num = [cols for cols in data_train.columns if str(data_train[cols].dtype)!="object"]
    cols_name_str =[cols for cols in data_train.columns if str(data_train[cols].dtype)=="object"]

    datas = pd.concat([data_train, data_test,data_val], axis=0)
    datas = feature_split(datas).values
    datas =minMaxer.fit_transform(datas)


    data_train=datas[0:data_train.shape[0],:]
    data_test = datas[data_train.shape[0]:data_test.shape[0]+data_train.shape[0],:]
    data_val = datas[data_test.shape[0]+data_train.shape[0]:,:]

    feature_name=["numerical feature","one-hot feature","one-hot pca"]
    data_train_feature={}
    data_test_feature = {}
    data_val_feature = {}
    data_train_feature[feature_name[0]]=data_train[:,0:len(cols_name_num)].copy()
    data_train_feature[feature_name[1]] =data_train
    data_train_feature[feature_name[2]] =pca_reducer.fit_transform(data_train)

    data_test_feature[feature_name[0]]=data_test[:,0:len(cols_name_num)].copy()
    data_test_feature[feature_name[1]] =data_test
    data_test_feature[feature_name[2]] =pca_reducer.transform(data_test)

    data_val_feature[feature_name[0]]=data_val[:,0:len(cols_name_num)].copy()
    data_val_feature[feature_name[1]] =data_val
    data_val_feature[feature_name[2]] =pca_reducer.transform(data_val)

#     one-hot pca 特征进行线性回归分析
    from sklearn.metrics import confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import  r2_score
    model=LogisticRegression()

    score_list = {}
    model.fit(data_train_feature[feature_name[2]], data_train_label)

    # 验证
    datas_map={"val":(data_val_feature[feature_name[2]],data_val_label),
               "test":(data_test_feature[feature_name[2]],data_test_label)}
    for key,value in datas_map.items():
        print("dataset name: ",key)
        input_feature=value[0]
        y_true = value[1]

        # R^2分数
        score = model.score(input_feature, y_true)
        y_test_pre = model.predict(input_feature)

        tn, fp, fn, tp = confusion_matrix(y_true, y_test_pre).ravel()
        sums=tn+fp+fn+tp
        print("tn:{},fp:{},fn:{},tp:{}".format(tn/sums,fp/sums,fn/sums,tp/sums))
        TPR = tp / (tp + fn)
        FPR = fp / (tn + fp)
        mark = (5 + 0.1) / (TPR - FPR)
        print("mark is: {0}".format(mark))
        print('Precision:', metrics.precision_score(y_true, y_test_pre))
        print('Recall:', metrics.recall_score(y_true, y_test_pre))
        print('F1-score:', metrics.f1_score(y_true, y_test_pre))
        print('ACC:', metrics.accuracy_score(y_true, y_test_pre))
        print('AUC score：', metrics.roc_auc_score(y_true, y_test_pre))
        if key=="test":
            prob=model.predict_proba(input_feature)[:,1]
            datas_test_org["prob"]=pd.Series(prob)


            datas_test_org.sort_values(by="prob", ascending=False,inplace=True)
            data_test_save = datas_test_org[["timestamp", "duration", "protocol", "sourceIP", "sourcePort"
                , "destinationIP", "destinationPort","stream_ID"]][y_test_pre == 1]

            data_test_save.to_csv(r"output/task2_one_hot_pca_LR.csv",index=None)
            data_test_save['conversation'] = data_test_save[['sourceIP', 'destinationIP']].apply(lambda x: '->'.join(x),
                                                                                                 axis=1)
            print(data_test_save['conversation'].value_counts()[:15])


    # 测试集合里面找botnet
    botnet_idx=0
    for i in range(y_true.shape[0]):
        if y_true[i]==1 and y_test_pre[i]==1:
            botnet_idx=i
            break
    botnet_vec=datas_map["test"][0][botnet_idx,:].reshape(1,-1)
    botnet_label = datas_map["test"][1][botnet_idx]
    botnet_vec_adversarial=botnet_vec.copy()
    rating=0.001
    epoch=0
    while True:
        epoch=epoch+1
        res_prob = model.predict_proba(botnet_vec_adversarial)
        print("epoch:{},grad for sample,prob:{}".format(epoch, res_prob))
        label_pred =np.argmax(res_prob)
        if label_pred==1:
            botnet_vec_adversarial=botnet_vec_adversarial-model.coef_*rating
        else:
            print("success generate adversarial sample")
            break
