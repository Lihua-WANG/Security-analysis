import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import from_model
from sklearn import metrics
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.decomposition import PCA
import clustering
import utils
import matplotlib.pyplot as plt
pca_reducer = PCA(n_components=5)
stdScaler = preprocessing.StandardScaler()
# 对特征进行PCA特征提取
cols_name_str=[]


def get_pc(X, num_components=2):
    """Performs dimensionality reduction to a matrix

    Parameters
    ----------
        num_components  : New dimensions
        tsne            : True if reduction with TSNE, PCA otherwise

    Returns
    -------


    """
    reduced_data = PCA(n_components=num_components).fit_transform(X)


    return reduced_data


def feature_split(feature_input):
    feature_hot=feature_input[cols_name_str]
    feature_hot=pd.get_dummies(feature_hot)
    feature_input.drop(cols_name_str,axis=1,inplace=True)
    feature_input=pd.concat([feature_input,feature_hot],axis=1)
    return feature_input

def dbscan_predict(model, X):

    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * 0

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)
        y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]



    return y_new
if __name__=="__main__":
    org_name = ["timestamp", "duration", "protocol", "sourceIP", "sourcePort", "direction"
        , "destinationIP", "destinationPort", "state", "STS", "DTS", "totalPackage",
                "totalBytes_Bidirection", "totalBytes_single_direction"]
    datas_test_org = pd.read_csv(r"data\A2_1\test_data", header=None, names=org_name)
    datas_test_org["stream_ID"]=pd.Series(list(range(0,datas_test_org.shape[0])))

    data_train= pd.read_csv(r"data\A2_1\1_train.csv",index_col=0)
    data_train=data_train.head(int(data_train.shape[0]/2))
    data_train_ip=data_train["sourceIP"]
    data_train.drop("sourceIP",inplace=True,axis=1)

    data_val = pd.read_csv(r"data\A2_1\1_val.csv",index_col=0)
    data_val_org=data_val.copy(deep=True)
    data_val_ip=data_val["sourceIP"]
    data_val.drop("sourceIP",inplace=True,axis=1)

    data_test = pd.read_csv(r"data\A2_1\1_test.csv",index_col=0)
    data_test_ip=data_test["sourceIP"]
    data_test.drop("sourceIP",inplace=True,axis=1)

    data_val_label = pd.read_csv(r"data\A2_1\1_valLabel.csv",index_col=0)
    # data_train=data_val.copy(deep=True)
    # data_test = data_val.copy(deep=True)
    # data_val = data_val.copy(deep=True)
    print(data_train.head())

    data_train.hist(figsize=(8, 8))
    # 数据显示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)
    plt.savefig("output/data_train_hist.png")

    cols_name_num = [cols for cols in data_train.columns if str(data_train[cols].dtype)!="object"]
    cols_name_str =[cols for cols in data_train.columns if str(data_train[cols].dtype)=="object"]

    datas = pd.concat([data_train, data_test,data_val], axis=0)
    datas = feature_split(datas).values
    datas =stdScaler.fit_transform(datas)

    # data_train=feature_split(data_train).values
    # data_test=feature_split(data_test).values
    # data_val=feature_split(data_val).values
    data_train=datas[0:data_train.shape[0],:]
    data_test = datas[data_train.shape[0]:data_test.shape[0]+data_train.shape[0],:]
    data_val = datas[data_test.shape[0]+data_train.shape[0]:,:]

    feature_name=["numerical_feature","onehot_feature","onehot_pca"]
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


    # 构建算法

    from sklearn.cluster import KMeans,DBSCAN
    from sklearn.metrics import confusion_matrix
    # "IsolationForest",
    models_name=["IsolationForest","LocalOutlierFactor","OPTICS"]
    for fs in feature_name:
        feature_train=data_train_feature[fs]
        featuren_val=data_val_feature[fs]
        featuren_test = data_test_feature[fs]

        for model_s in models_name:
            print(utils.get_time()," feature is {},and model is {}".format(fs,model_s))
            if model_s==models_name[0]:
                # # IFOREST
                #  model iforest
                iforest = clustering.iforest_train(feature_train,contam='auto')
                print(utils.get_time(), "iforest train end")
                labels = clustering.iforest_predict(featuren_val, iforest)
                labels_test=clustering.iforest_predict(featuren_test, iforest)
            elif model_s==models_name[1]:
                labels_all=clustering.lof_fit_predict(np.vstack((feature_train,featuren_val,featuren_test)))
                labels=labels_all[feature_train.shape[0]:feature_train.shape[0]+featuren_val.shape[0]]
                labels_test=labels_all[feature_train.shape[0]+featuren_val.shape[0]:]
            elif model_s==models_name[2]:
                labels_all=clustering.optics_fit_predict(np.vstack((feature_train,featuren_val,featuren_test)),54, 'dbscan', 5)
                labels=labels_all[feature_train.shape[0]:feature_train.shape[0]+featuren_val.shape[0]]
                labels_test=labels_all[feature_train.shape[0]+featuren_val.shape[0]:]
            labels_test = (1 - labels_test) / 2
            
            labels=(1-labels)/2
            vv=data_val_label.values.reshape(-1)
            result=utils.print_result(data_val_label.values,labels)
            print(result)
            # dimensionality reduction
            XR = get_pc(featuren_val, 2)
            fig, ax = plt.subplots()
            colors=["red","blue"]
            for la in range(0,2):
                x=XR[labels==la,:]
                plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[la])
            fig.savefig("output/" + fs +"_"+model_s + ".png")
        #     保存csv文件
            data_test_save=datas_test_org[["timestamp", "duration", "protocol", "sourceIP", "sourcePort"
        , "destinationIP", "destinationPort","stream_ID"]][labels_test==1]
            data_test_save.to_csv("output/task1_" + fs +"_"+model_s + ".csv",index=None)
            data_test_save['conversation'] = data_test_save[['sourceIP', 'destinationIP']].apply(lambda x: '->'.join(x),
                                                                                                 axis=1)
            print(data_test_save['conversation'].value_counts()[:15])



        # print results

        #  get anomalies
        # df_anomalies_iforest = clustering.iforest_anomalies(data_val_org, labels)
        #
        # utils.save(df_anomalies_iforest, "df_anomalies_iforest")

    #
    # # kmeans聚类算法，分析聚类中心数
    # # elbow
    # from sklearn.cluster import KMeans,DBSCAN
    # from sklearn.metrics import confusion_matrix
    # for fs in feature_name:
    #     # inertias = []
    #     # n=[]
    #     # print("features name: ",fs)
    #     # max_cluster = 5
    #     # for N in range(2, max_cluster,2):
    #     #     km = KMeans(n_clusters=N)
    #     #     km.fit(data_train_feature[fs])
    #     #     inertias.append(km.inertia_)
    #     #     print(inertias)
    #     #     n.append(N)
    #     #
    #     # plt.figure()
    #     # plt.plot(n,inertias)
    #     # plt.title("cluster elbow {0}".format(fs))
    #     # plt.ylabel("KMeans inertia")
    #     # plt.xlabel("n_clusters")
    #     # plt.show()
    #
    #     method_dict={"dbscan":DBSCAN(),
    #                 "kmeans": KMeans(n_clusters=10),
    #                  }
    #
    #     for method_name,model in method_dict.items():
    #         model.fit(data_train_feature[fs][1:100,:])
    #
    #         ax = plt.subplot(224, projection='3d')  # 创建一个三维的绘图工程
    #         cluster_number=len(set(model.labels_))
    #         # 记录每个镞的评分
    #         if method_name=="kmeans":
    #             labels_pred = model.predict(data_val_feature[fs])
    #         else:
    #             labels_pred =dbscan_predict(model,data_val_feature[fs])
    #
    #         labels_map={}
    #         for idx in range(cluster_number):
    #             # 该类中的标签
    #             labelData = data_val_label.loc[labels_pred == idx,:]
    #             tmp=labelData.iloc[:,0].unique()
    #             if len(tmp)>0:
    #                 labels_map[idx]=tmp[0]
    #         labels_pred=[labels_map[x] for x in labels_pred ]
    #         label_true=data_val_label['label'].values.tolist()
    #         tn, fp, fn, tp  = confusion_matrix(label_true,labels_pred,labels=list(set(label_true))).ravel()
    #         TPR=tp/(tp+fn)
    #         FPR = fp / (tn + fp)
    #         mark=(5+0.1)/(TPR-FPR)
    #         print("mark is: {0}".format(mark))


