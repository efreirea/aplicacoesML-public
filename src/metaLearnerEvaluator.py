from lib.tsmfe import MFE
from lib.utils import Reader
from lib.utils import ErrorMetricCalculator
from lib.utils import ClusterParser
import argparse
import os
import pandas as pd
import numpy as np

from sklearn.svm import  SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn import metrics


class MetalearnerEvaluator(object):
    _best_params = {'kernel': 'rbf', 'C': 8}
    _features_selected_idx = ['autocorrelation', 'autocorrelation_mean', 'len', 'basic_trend','kurtosis', 'max_fft_real', 'skew', 'std', 'trend']
    def __init__(self,predicted_path,original_path,metabase_path,reference_method):
        self._results = []
        self._predicted_path = predicted_path
        self._original_path = original_path
        self._metabase_path = metabase_path
        self._reference_method = reference_method
    def preprocess(self,data,igCol=None):
        data_preproc = data.copy(deep=True)
        data_preproc['len'] = data_preproc['len'].astype(float)
        data_preproc['max_fft_imag'] = data_preproc['max_fft_imag'].astype(float)
        # data_preproc.dropna(subset=['Winner','Community_0','Community_1','Community_2'],inplace=True)
        data_preproc.dropna(subset=['Winner'],inplace=True)

        if igCol is not None and len(igCol)>0:
            data_preproc.drop(columns=igCol,inplace=True)

        self._le = LabelEncoder()
        data_preproc['Winner'] = self._le.fit_transform(data_preproc['Winner'])
        cat_columns = ['sector'] if igCol is None or 'sector' not in igCol else []
        for i in range(7): ## GAMBIARRA. DEIXAR MAIs ORG DEPOIS
            col_name='Community_{}'.format(i)
            if col_name in data_preproc.columns:
                print("================COMMUNITY COLUMN DETECTED=================")
                data_preproc[col_name] = data_preproc[col_name].astype(str)
                data_preproc.fillna(value={col_name: 'NA'},inplace=True)
                cat_columns.append(col_name)
        data_preproc = pd.get_dummies(data_preproc, prefix_sep="__", columns=cat_columns,dtype=float)

        var_thresh = VarianceThreshold()
        var_thresh.fit(data_preproc)
        data_preproc = data_preproc.iloc[:,var_thresh.get_support()]

        return data_preproc

    def calculate_baselines(self,X,Y):
        baselines = []
        #Using majority class as baseline just for accuracy
        dummy_model = DummyClassifier(strategy='most_frequent').fit(X,Y)
        y_pred = dummy_model.predict(X)
        baselines.append(metrics.accuracy_score(Y, y_pred))

        #Using stratfied random rediction as baseline just for the other metrics
        dummy_model = DummyClassifier(strategy='stratified').fit(X,Y)
        y_pred = dummy_model.predict(X)
        baselines.append(metrics.cohen_kappa_score(Y, y_pred))
        baselines.append(metrics.f1_score(Y, y_pred))
        baselines.append(metrics.recall_score(Y, y_pred))
        return baselines
    def _prepare_metabase(self):
        meta_base = pd.read_csv(self._metabase_path,index_col=0)
        data_preproc = self.preprocess(meta_base)
        selected_features = MetalearnerEvaluator._features_selected_idx
        selected_features.append('Winner') # ading class
        selected_columns_idx =  pd.Index(selected_features)
        self._meta_base = data_preproc.loc[:,selected_columns_idx]
    def _train_model(self):
        self._classifier = SVC(**MetalearnerEvaluator._best_params,gamma='scale')
        X = self._meta_base.drop(columns=['Winner'])
        Y = self._meta_base.Winner
        x_train, self._x_test, y_train, self._y_test = train_test_split(X, Y, test_size=0.3,stratify=Y,random_state=42)

        print(x_train)
        # Scaling
        self._scaling_model = MinMaxScaler()
        self._scaling_model = self._scaling_model.fit(x_train)
        x_train = self._scaling_model.transform(x_train)

        print(self._x_test)

        self._classifier = self._classifier.fit(x_train,y_train)
    def evaluate(self):
        self._prepare_metabase()
        self._train_model()

        self._x_test.iloc[:,:] = self._scaling_model.transform(self._x_test)
        print("====Baselines=======")
        baselines = self.calculate_baselines(self._x_test,self._y_test)
        print(baselines)

        y_true, y_pred = self._y_test, self._classifier.predict(self._x_test.loc[self._x_test.index])
        print('Results on the test set:')
        print(metrics.classification_report(y_true, y_pred))
        print("Accuracy score", metrics.accuracy_score(self._y_test, y_pred))


        self._pb_partial_results=[]

        for ticker,recommended_model in zip(self._x_test.index,self._le.inverse_transform(y_pred)):
            predicted_reference, _ = Reader.read_predicted_and_original_preproc_by_model_indexed(ticker,self._reference_method,self._predicted_path,self._original_path)
            # predicted_recommendation, original_recommendation = Reader.read_predicted_and_original_preproc_by_model_indexed(ticker,self._reference_method,self._predicted_path,self._original_path)  #Sanity check..
            predicted_recommendation, original_recommendation = Reader.read_predicted_and_original_preproc_by_model_indexed(ticker,recommended_model,self._predicted_path,self._original_path)
            # Reindexing the reference to use the same indexes of recommendation
            predicted_reference = Reader.reindex_original(predicted_reference,predicted_recommendation)
            res = ErrorMetricCalculator.pb_single_series(predicted_recommendation,original_recommendation,predicted_reference)
            self._pb_partial_results.append(res)
            print(ticker,res)
        print(ErrorMetricCalculator.pb_aggregate(self._pb_partial_results))
    def save(self):
        pass

if __name__=='__main__':
    np.random.seed(1)
    parser = argparse.ArgumentParser(description='')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d',help='Directory containing all original series, one per file')
    parser.add_argument('--metabase',help='File Containing Metabase',required=True)
    parser.add_argument('-p',help='Directory containing all predicted series, one per file',required=True)
    parser.add_argument('--reference',help='Method to be used as reference',required=True)
    args = parser.parse_args()

    if args.d is not None:
        m_evaluate = MetalearnerEvaluator(args.p,args.d,args.metabase,args.reference)
        m_evaluate.evaluate()
