from lib.tsmfe import MFE
from lib.utils import Reader
from lib.utils import ErrorMetricCalculator
from lib.utils import ClusterParser
import argparse
import os
import pandas as pd
import numpy as np


class MetadatabaseBuilder(object):
    AVAILABLE_METRICS = {
        'pocid': ErrorMetricCalculator.POCID,
        'mse': ErrorMetricCalculator.mse,
        'theilu': ErrorMetricCalculator.theilU,
        'mcpm': ErrorMetricCalculator.MCPM
    }
    def __init__(self,predicted_path,metric_name,arch=False,garch=False,holt=False,rw=False,ar=False,sectors=False,communities=None):
        self._results = []
        self._arch = arch
        self._garch = garch
        self._holt = holt
        self._rw = rw
        self._ar = ar
        self._predicted_path = predicted_path
        self._sectors = sectors
        self._communities = communities
        self._metric = MetadatabaseBuilder.AVAILABLE_METRICS[metric_name]
        self._metric_name = metric_name

    def _evaluate_model(self,original_path,series_original,model,res):
        try:
            series_predicted = Reader.read_predicted_by_model(original_path,model,self._predicted_path)
            series_original_reindexed = Reader.reindex_original(series_original,series_predicted)
            res[model] = self._metric(series_predicted,series_original_reindexed)
        except:
            res[model] = None
    def performance_eval(self,original_path):
        series_original = Reader.read_preproc(original_path)
        res = {}
        if self._arch:
            self._evaluate_model(original_path,series_original,"CH",res)
        if self._garch:
            self._evaluate_model(original_path,series_original,"GCH",res)
        if self._holt:
            self._evaluate_model(original_path,series_original,"HL",res)
        if self._rw:
            self._evaluate_model(original_path,series_original,"RW",res)
        if self._ar:
            self._evaluate_model(original_path,series_original,"AR",res)
        if None not in res.values():
            ordered = sorted(res.items(), key= lambda x : x[1])
            if ordered[0][1] == ordered[1][1]:
                print("Draw")
                res["Winner"] = None
            else:
                res["Winner"] = ordered[0][0]
        else:
            print(res)
            res["Winner"] = None
        return res
    def extract_mf(self,series_original,series_preproc,prefix=""):
        mfes_stationary = [MFE.kurtosis,MFE.skew,MFE.var_coef,MFE.autocorrelation_mean,MFE.std]
        mfes_common = [MFE.len,MFE.basic_trend, MFE.trend,MFE.turningpoints_ratio,MFE.autocorrelation]

        res_common = {prefix+method.__name__:method(series_original) for method in mfes_common}
        # TReating complex numbere differently
        fft_result = MFE.max_fft(series_original)
        res_common['max_fft_real'] = np.real(fft_result)
        res_common['max_fft_imag'] = np.imag(fft_result)

        res_stationary = {prefix+method.__name__:method(series_preproc) for method in mfes_stationary}
        res = dict(**res_stationary,**res_common)
        return res
    def append(self,original_path):
        original_raw = Reader.read_raw(original_path)
        original_preproc = Reader.read_preproc(original_path)
        ticker,_ = Reader.get_ticker_and_model(original_path)
        metafeatures = self.extract_mf(original_raw,original_preproc)
        if self._sectors:
            metafeatures['sector']=os.path.basename(os.path.dirname(original_path))
        performance = self.performance_eval(original_path)
        result_dict = {**metafeatures, **performance}
        result_dict['ticker'] = ticker
        self._results.append(result_dict)
    def save(self):
        df = pd.DataFrame(self._results).set_index("ticker")
        if self._communities is not None:
            res_all = ClusterParser.parse(self._communities)
            print(len(res_all))
            for i in range(len(res_all)):
                res = res_all[i]
                for key,value in res.items():
                    intersec = pd.Index(value).intersection(df.index)
                    df.loc[intersec,'Community_{}'.format(i)] = key
        df.to_csv(os.path.join(self._predicted_path,"metabase_{}.csv".format(self._metric_name)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract Meta features from time series.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f',help='Single Series File')
    group.add_argument('-d',help='Directory containing all original series, one per file')
    parser.add_argument('-p',help='Directory containing all predicted series, one per file')
    parser.add_argument('--sectors',action='store_true',help='Indicates the directory given in -d contains subdirectories indicating the sectors.')
    parser.add_argument('--communities',action='store',help='File containing communities indication')
    parser.add_argument('--metric',help='MEtric to evaluate performance on base level',required=True,choices=['pocid','mcpm','theilu','mse'])
    args = parser.parse_args()

    if args.f is not None:
        read_and_extract_all(args.f)
    elif args.d is not None:
        m_builder = MetadatabaseBuilder(args.p,args.metric,False,False,True,True,True,args.sectors,args.communities)
        for subdir, dirs, files in os.walk(args.d):
            for file in files:
                filename, file_extension = os.path.splitext(file)
                full_path = os.path.join(subdir,file)
                if file_extension=='.csv':
                    print(full_path)
                    m_builder.append(full_path)
        m_builder.save()
