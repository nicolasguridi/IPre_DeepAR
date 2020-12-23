# import required libraries
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from itertools import islice

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.distribution.gaussian import GaussianOutput
from gluonts.distribution.poisson import PoissonOutput
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
# from gluonts.dataset.field_names import FieldName

from hydroeval import nse_c2m
from hydroeval import evaluator as hydroevaluator

def get_df(df, columns, past_years):
    """
    Create dataframe with specified timeseries and time period
   
    Parameters
    ----------
    df : pandas Dataframe
        dataframe containing all time series and years available.
    
    columns : list
        list with column names of the time series to be selected.
    
    past years : int
        number of years to consider into the past.
    
    Returns
    -------
    pandas Dataframe
        dataframe containing the last `past years` of the selected time series.

    """
    new_df = df[columns].iloc[- past_years * 12:] 
    new_df = new_df.set_index(new_df.columns[0])
    return new_df

def calculate_nse(tss, forecasts):
    """
    Calculate the Nash-Sultcliffe Efficiency of a probabilistic forecast given its observed values.

    Parameters
    ----------
    tss : list
        list of the real observations of the time series.
    
    forecasts : list
        list of glounts.model.forecast.Forecast that contains the prediction samples made by the trained predictor.
    
    Returns
    -------
    np.array
        array with the mean NSE of all forecasts in `forecasts`.
    """
    nses = []
    for i in range(len(forecasts)):
        sim_flow = np.median(forecasts[i].samples, axis=0)
        obs_flow = tss[i][-len(sim_flow):].to_numpy().flatten()
        nses.append(hydroevaluator(nse_c2m, sim_flow, obs_flow, axis=1)[0])
    return np.array(nses).mean()

def plot_forecasts(tss, forecasts, past_length):
    """
    Plots forecasts and observed values

    Parameters
    ----------
    tss : list
        list of the real observations of the time series.
    
    forecasts : list
        list of glounts.model.forecast.Forecast that contains the prediction samples made by the trained predictor.
    
    past_length : int
        number of years to plot
    
    Returns
    -------
    """
    for target, forecast in islice(zip(tss, forecasts), len(forecasts)):
        target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.show()

def fit(df, time_series, prediction_length=12, epochs=10):
    """
    Trains DeepAR model with provided time_series data

    Parameters
    ----------

    df : pd.Dataframe
        datframe containing all the time series to be used on model training.
    
    time_series : list
        list of the time series names on the dataframe
    
    prediction_lenght : int, optional
        number of consecutive months to predict (default is 12; a year).
    
    epochs : int, optional
        number of epochs to use on model training (default is 10).
    
    Returns
    -------
    DeepAREstimator
        DeepAR model predictor trained over the provided data.
    """
    # set train dataset
    start = df.index[0]
    training_data = ListDataset(
        [
            {"start": start, "target": df[time_series[i]][:"2013-01"]} for i in range(1, len(time_series))
        ],
        freq = "M"
    )

    # create estimator, other configuration options can be used
    estimator = DeepAREstimator(freq="M", 
                                prediction_length=prediction_length,
                                context_length=24,
                                distr_output=NegativeBinomialOutput(),
                                trainer=Trainer(epochs=epochs, batch_size=24))

    # train estimator
    predictor = estimator.train(training_data=training_data)
    return predictor

def predict(predictor, df, test_station, prediction_length=12, plot=False):
    """
    Make probabilistic predictions, plot and evaluate the results

    Parameters
    ----------
    predictor : GluonEstimator
        model predictor trained over the training data.

    df : pd.Dataframe
        datframe containing the time series to be used on model testing.
    
    test_station : str
        name of the time series whose predictions will be tested
    
    prediction_lenght : int, optional
        number of consecutive months to predict (default is 12; a year).
    
    plot : bool, optional
        plot the test predictions against the observed values.
    
    Returns
    -------
    pd.Dataframe
        dataframe containing result metrics of the model predictor and a seasonal naive predictor
    """
    # set test dataset
    start = df.index[0]
    test_data = ListDataset(
        [
            {"start": start, "target": df[test_station][:"2018-01"]},
            {"start": start, "target": df[test_station][:"2017-01"]},
            {"start": start, "target": df[test_station][:"2016-01"]},
            {"start": start, "target": df[test_station][:"2015-01"]},
            {"start": start, "target": df[test_station][:"2014-01"]}
        ],
        freq = "M"
    )
    # make predictions
    forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=predictor, num_samples=50)
    
    # evaluate predictions
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], seasonality=12)
    forecasts = list(forecast_it)
    tss = list(ts_it)

    if (plot):
        plot_forecasts(tss, forecasts, past_length=12)

    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    agg_metrics['NSE'] = calculate_nse(tss, forecasts)

    # baseline predictor
    seasonal_predictor = SeasonalNaivePredictor(freq="M", prediction_length=prediction_length, season_length=12)

    forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=seasonal_predictor, num_samples=10)
    seasonal_forecasts = list(forecast_it)
    seasonal_tss = list(ts_it)
    
    # if (plot):
    #     plot_forecasts(seasonal_tss, seasonal_forecasts, past_length=24, num_plots=len(test_data))
    agg_metrics_seasonal, item_metrics_seasonal = evaluator(iter(seasonal_tss), iter(seasonal_forecasts), num_series=len(test_data))

    agg_metrics_seasonal['NSE'] = calculate_nse(seasonal_tss, seasonal_forecasts)
    # compare models

    df_metrics = pd.DataFrame.join(
        pd.DataFrame.from_dict(agg_metrics, orient='index').rename(columns={0: "DeepAR"}),
        pd.DataFrame.from_dict(agg_metrics_seasonal, orient='index').rename(columns={0: "Seasonal naive"})
    )
    #return df_metrics.loc[['MASE', 'MSIS', 'NSE']]
    return df_metrics