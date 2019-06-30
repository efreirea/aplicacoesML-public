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
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from lib.utils import ErrorMetricCalculator
from lib.utils import Reader

class Plotter(object):
    @staticmethod
    def _get_original_name_and_model_and_reference(filename_predicted):
        ticker_model_search = re.search("(.*)_(.*).csv",os.path.basename(filename_predicted))
        if ticker_model_search:
            ticker = ticker_model_search.group(1)
            model = ticker_model_search.group(2)
        else:
            raise ValueError("Filename template not recognied: {}".format(filename_predicted))

        original_name = "{}_data.csv".format(ticker)
        reference_name = "{}_SES.csv".format(ticker)
        return original_name,model,reference_name
    @staticmethod
    def _plot_single(original,predicted,label_predicted,plotname=""):
        plt.figure()
        # print(original)
        # print(predicted)
        # original_preproc = np.log(original/original.shift(1)).dropna()
        plt.plot(original,label="Original")
        plt.plot(predicted,label=label_predicted)
        plt.title(plotname)
        plt.legend()

    @staticmethod
    def plot(predicted_file,original_folder,original_column_name='close'):
        series_predicted,series_original = Reader.read_predicted_and_original_preproc_indexed(predicted_file,original_folder,original_column_name)
        ticker,model = Reader.get_ticker_and_model(predicted_file)
        print({"ticker":ticker,"model": model,"RMSE":ErrorMetricCalculator.rmse(series_predicted.values,series_original.values)})
        Plotter._plot_single(series_original,series_predicted,model,ticker)
    @staticmethod
    def plot_all(predicted_folder, original_folder):
        for subdir, dirs, files in os.walk(predicted_folder):
            for file in files:
                filename, file_extension = os.path.splitext(file)
                full_path = os.path.join(subdir,file)
                if file_extension=='.csv':
                    # print(full_path)
                    Plotter.plot(full_path,original_folder)
        plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate performance of time series algorithms.')

    parser.add_argument('-o','--original',action='store',help='Directory storing the original files')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d',help='Folder container the preditions')
    args = parser.parse_args()
    print(args)
    if args.d is not None and args.original:
        Plotter.plot_all(args.d,args.original)
    elif args.d is not None:
        pass
