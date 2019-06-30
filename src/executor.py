import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import sklearn as sk
import math
import argparse
import operator
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import warnings
import os
import re
from arch import arch_model
from random import seed

class ErrorMetricCalculator(object):
    @staticmethod
    def pb(predicted,original,reference):
        "Ideally, a low value of this metric indicates that predicted is better than reference"
        predicted_error = np.absolute(predicted-original)
        reference_error = np.absolute(reference-original)
        #Reference Better Than Predicted
        ref_better_than_pred = reference_error < predicted_error
        m = (predicted_error != reference_error).sum()
        ret = ref_better_than_pred.sum()/m
        # print(ret)
        return ret

class Predictor(object):
    def __init__(self,train_data,h=20):
        self._train_data = train_data
        self._h = h
    def Holt(self):
        # model = Holt(self._train_data,damped=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = Holt(self._train_data.values)
            model_fit = model.fit()
            output = model_fit.forecast(self._h)
        # print(output)
        return output
    def RandomWalk(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = ARIMA(self._train_data.values, order=(0,1,0))
            model_fit = model.fit(disp=-1)
            output = model_fit.forecast(self._h)
        # print(output)
        return output[0]
    def ARIMA(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # model = ARIMA(self._train_data.values, order=(3,0,3))
            model = ARIMA(self._train_data.values, order=(1,0,0))
            model_fit = model.fit(disp=-1)
            output = model_fit.forecast(self._h)
        # print(output)
        return output[0]
    def ARCH(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = arch_model(self._train_data, p=1, q=0)
            model_fit = model.fit(disp='off')
            forecasts = model_fit.forecast(horizon=self._h, method='simulation', simulations=1)
            sims = forecasts.simulations
            output = sims.values[-1,:,:].T.ravel()
            # print(output)
            return output
    def GARCH(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = arch_model(100.0*self._train_data, p=1, q=1)
            model_fit = model.fit(disp='off')
            forecasts = model_fit.forecast(horizon=self._h, method='simulation', simulations=1)
            sims = forecasts.simulations
            output = sims.values[-1,:,:].T.ravel()*0.01
            # print(output)
            return output
class PerformanceEvaluator(object):
    def __init__(self,filename,hl=False,rw=False,arima=False,arch=False,garch=False,output_folder=None,h=20,window_size=240,workers=1,preproc=False):
        self._filename = filename
        self._hl = hl
        self._rw = rw;
        self._arima = arima;
        self._arch = arch;
        self._garch = garch;
        self._h = h
        self._preproc = preproc
        self._window_size = window_size
        self._workers = workers
        self._output_folder=output_folder
        if (self._output_folder is not None) and not os.path.isdir(self._output_folder):
            raise ValueError("Output folder does not exists")
    def _eval_error(self,aggregated):
        # original_test = self._series[self._window_size:].values
        # original_test = self._series.loc[aggregated['RW'].index].values
        # print(len(original_test))
        # print(len(aggregated['RW']))
        return {key:0 for key,value in aggregated.items()}
        # return {key:ErrorMetricCalculator.pb(value,original_test,aggregated['SES']) for key,value in aggregated.items()}
    def _save_output_file(self,aggregated):
        outfile_template_name = "{}_{}.csv"
        ticker_search = re.search(".*/(.*)_data.csv",self._filename)
        if ticker_search:
            ticker = ticker_search.group(1)
        else:
            ticker , _ = os.path.splitext(os.path.basename(self._filename))
        for key,value in aggregated.items():
            outfile_path = os.path.join(self._output_folder,outfile_template_name.format(ticker,key))
            value.to_csv(outfile_path)
    def evaluate(self):
        self._full_data = pd.read_csv(self._filename, header=0, parse_dates=[0], index_col=0, squeeze=True)
        self._series = self._full_data['close']
        if self._preproc:
            self._series = (self._series/self._series.shift(1)).dropna()
        work_queue = self._sliding_window_splitter()
        if self._workers == 1:
            results = []
            for work in work_queue:
                results.append(self.predict(work));
        else:
            with Pool(processes=self._workers) as pool:
                results = pool.map(self.predict,work_queue)
        aggregated = self._aggregate(results)
        # print(aggregated)
        if self._output_folder is not None:
            self._save_output_file(aggregated)
        # print('===========')
        # print(aggregated['SES'])
        # print("EvaluationAmounts ",len(work_queue))
        # errors_ev = self._eval_error(aggregated)
        # print(errors_ev)
        self._series = None
        self._full_data = None
    def _aggregate(self,work_results):
        results = sorted(work_results,key= lambda x : x['index'][0])
        # print(results)
        ret = {}
        if len(work_results)>0:
            key_list=list(work_results[0].keys())
            key_list.remove('index')
            # ret = {key:np.array([work[key] for work in results]) for key in key_list}
            for key in key_list:
                concatened_values = np.concatenate(tuple(predicted_values[key]  for predicted_values in results))
                concatened_indexes = np.concatenate(tuple(predicted_indexes['index']  for predicted_indexes in results))
                # print(len(concatened_values))
                # print(len(concatened_indexes))
                ret[key] = pd.DataFrame(concatened_values,index=concatened_indexes,columns=['close'])
        return ret
    def _sliding_window_splitter(self):
        series_length = len(self._series)
        # print(series_length)
        start_indexes = range(0,series_length-self._window_size-self._h+1,self._h) #plus one because range() is exclusive on left
        end_indexes = range(self._window_size,series_length-self._h+1,self._h) # open interval
        return [{'start': start,'end':end} for start,end in zip(start_indexes,end_indexes)]
    def predict(self,work):
        train = self._series[work['start']:work['end']]
        predictor = Predictor(train,self._h)
        # ret = {'index': work['end']+self._h-1}
        ret = {'index': self._series.iloc[work['end']:work['end']+self._h].index.values}
        if self._hl:
            predicted_HL = predictor.Holt()
            ret['HL'] = predicted_HL
        if self._rw:
            predicted_RW = predictor.RandomWalk()
            ret['RW'] = predicted_RW
        if self._arima:
            predicted_AR = predictor.ARIMA()
            ret['AR'] = predicted_AR
        if self._arch:
            predicted_CH = predictor.ARCH()
            ret['CH'] = predicted_CH
        if self._garch:
            predicted_GCH = predictor.GARCH()
            ret['GCH'] = predicted_GCH
        return ret
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate performance of time series algorithms.')

    parser.add_argument('-HL','--Holt', action='store_true',help='Run Holt Sxponential Smoothing')
    parser.add_argument('-RW', '--RandomWalk', action='store_true',help='Run RandomWalk')
    parser.add_argument('-AR', '--Arima', action='store_true',help='Run Arima with parameters (1,0,0)')
    parser.add_argument('-CH', '--Arch', action='store_true',help='Run arch ')
    parser.add_argument('-GCH', '--Garch', action='store_true',help='Run garch ')
    # parser.add_argument('-r','-R', action='store_true',help='Recursive walk ')
    parser.add_argument('-w','--workers',action='store', type=int,default=1,help='How many works to run in parallel. Default 1 ')
    parser.add_argument('-o','--output',action='store',help='Directory to store predicted series. If not specified, nothing will be stored')
    # parser.add_argument('--preproc',action='store_true',help='wwether or not to apply prepoc')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f',help='Single Series File')
    group.add_argument('-d',help='Directory containing all series, one per file')
    args = parser.parse_args()
    print(args)
    if args.f is not None:
        pe = PerformanceEvaluator(args.f,args.Holt,args.RandomWalk,args.Arima,args.Arch,args.Garch,workers=args.workers,output_folder=args.output,h=2,preproc=True,window_size=2)
        pe.evaluate()
    elif args.d is not None:
        for subdir, dirs, files in os.walk(args.d):
            for file in files:
                filename, file_extension = os.path.splitext(file)
                full_path = os.path.join(subdir,file)
                if file_extension=='.csv':
                    # print(full_path)
                    pe = PerformanceEvaluator(full_path,args.Holt,args.RandomWalk,args.Arima,args.Arch,args.Garch,workers=args.workers,output_folder=args.output,h=5,preproc=True,window_size=100)
                    pe.evaluate()
