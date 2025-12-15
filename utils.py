from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgb_model import xgb_model
from ar_model import ar_model
import seaborn as sns
import pandas as pd
import numpy as np
import time

def calculate_metrics(test, preds):
    mse = mean_squared_error(test, preds)
    mae = mean_absolute_error(test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, preds)
    return mse, mae, rmse, r2

def get_results_xgb_rolling(dataset, target, p, prop_test=0.1, verbose=True):
    df = pd.read_csv(f'data/{dataset}.csv')
    ts = df[target]
    train = ts[:int(len(ts) * (1 - prop_test))]
    test = ts[int(len(ts) * (1 - prop_test)):]
    
    xgb_md = xgb_model(p=p)
    xgb_md.train_model(train=train)
    test_hat = xgb_md.forecast(train=train, n_forecast=test.size)
    mse_xgb, mae_xgb, rmse_xgb, r2_xgb = calculate_metrics(test=test, preds=test_hat)
    
    if verbose:
        print(f'Results for xgb on {target}')
        print(f'Mean Squared Error: {mse_xgb}')
        print(f'Root Mean Squared Error: {rmse_xgb}')
        print(f'Mean Absolute Error: {mae_xgb}')
        print(f'R^2: {r2_xgb}')
    
    return {
        'test': test,
        'predictions': test_hat,
        'mse': mse_xgb,
        'mae': mae_xgb,
        'rmse': rmse_xgb,
        'r2': r2_xgb
    }

def plot_best_rolling(results):
    """Plots the forecast for the best XGB and AR models (rolling approach)"""
    plt.close()
    test = results['XGB']['metrics'][results['XGB']['best_p']]['test']
    preds_xgb = results['XGB']['metrics'][results['XGB']['best_p']]['predictions']
    preds_ar = results['AR']['metrics'][results['AR']['best_p']]['predictions']
    t = np.arange(len(test))

    best_p_xgb = results['XGB']['best_p']
    best_p_ar = results['AR']['best_p']

    sns.lineplot(x=t, y=test, label='Test', zorder=2, alpha=0.2, color='#0d00ff')
    sns.lineplot(x=t, y=preds_xgb, label=f'XGB Predictions (p={best_p_xgb})', zorder=2, alpha=0.2, color='#ff00bf')
    sns.lineplot(x=t, y=preds_ar, label=f'AR Predictions (p={best_p_ar})', zorder=2, alpha=0.2, color='#59ff00')

    plt.title(f'Best XGBoost and AR Models (Rolling Approach)')
    plt.ylabel('Power Consumption (MW)')
    plt.xlabel('Time')
    plt.grid(alpha=0.4, zorder=1)
    plt.legend()
    plt.savefig('figs/plot-rolling-comparison.png')

def get_results_AR_rolling(dataset, target, p, prop_test=0.1, verbose=True):
    df = pd.read_csv(f'data/{dataset}.csv')
    ts = df[target]
    train = ts[:int(len(ts) * (1 - prop_test))]
    test = ts[int(len(ts) * (1 - prop_test)):]

    ar_md = ar_model(p=p)
    ar_md.train_model(train=train)
    test_hat = ar_md.forecast(train=train, n_forecast=test.size)
    mse_AR, mae_AR, rmse_AR, r2_AR = calculate_metrics(test=test, preds=test_hat)
    
    if verbose:
        print(f'Results for AR (rolling) on {target}')
        print(f'Mean Squared Error: {mse_AR}')
        print(f'Root Mean Squared Error: {rmse_AR}')
        print(f'Mean Absolute Error: {mae_AR}')
        print(f'R^2: {r2_AR}')
    
    return {
        'test': test,
        'predictions': test_hat,
        'mse': mse_AR,
        'mae': mae_AR,
        'rmse': rmse_AR,
        'r2': r2_AR
    }

def find_best_model_rolling(dataset, target, p_candidates, prop_test=0.1):
    results = {
        'XGB': {'best_p': None, 'best_mse': np.inf, 'metrics': {}},
        'AR': {'best_p': None, 'best_mse': np.inf, 'metrics': {}}
    }
    for p in p_candidates:
        try:
            xgb_results = get_results_xgb_rolling(dataset, target, p, prop_test, verbose=False)
            results['XGB']['metrics'][p] = xgb_results
            if xgb_results['mse'] < results['XGB']['best_mse']:
                results['XGB']['best_mse'] = xgb_results['mse']
                results['XGB']['best_p'] = p
        except Exception as e:
            print(f"XGB MODEL FAILED AT P={p}")
        
        try:
            ar_results = get_results_AR_rolling(dataset, target, p, prop_test, verbose=False)
            results['AR']['metrics'][p] = ar_results
            if ar_results['mse'] < results['AR']['best_mse']:
                results['AR']['best_mse'] = ar_results['mse']
                results['AR']['best_p'] = p
        except Exception as e:
            print(f"AR MODEL FAILED AT P={p}")

    best_xgb_p = results['XGB']['best_p']
    best_xgb = results['XGB']['metrics'][best_xgb_p]
    print(f"Best XGB Model - Rolling (p={best_xgb_p})")
    print(f"MSE: {best_xgb['mse']:.4f}")
    print(f"RMSE: {best_xgb['rmse']:.4f}")
    print(f"MAE: {best_xgb['mae']:.4f}")
    print(f"R_squared: {best_xgb['r2']:.4f}")
    print()
    
    best_ar_p = results['AR']['best_p']
    best_ar = results['AR']['metrics'][best_ar_p]
    print(f"Best AR Model - Rolling (p={best_ar_p})")
    print(f"MSE: {best_ar['mse']:.4f}")
    print(f"RMSE: {best_ar['rmse']:.4f}")
    print(f"MAE: {best_ar['mae']:.4f}")
    print(f"R_squared: {best_ar['r2']:.4f}")
    print()

    if results['XGB']['best_mse'] < results['AR']['best_mse']:
        print(f"Overall Best Model: XGB with p={best_xgb_p} (MSE: {results['XGB']['best_mse']:.4f})")
    else:
        print(f"Overall Best Model: AR with p={best_ar_p} (MSE: {results['AR']['best_mse']:.4f})")

    return results


def get_results_xgb_uptodate(dataset, target, p, prop_test=0.1, verbose=True):
    """XGBoost predictions using true y values (up-to-date approach)"""
    df = pd.read_csv(f'data/{dataset}.csv')
    ts = df[target]
    train_size = int(len(ts) * (1 - prop_test))
    train = ts[:train_size]
    test = ts[train_size:]
    
    xgb_md = xgb_model(p=p)
    xgb_md.train_model(train=train)
    test_hat = xgb_md.predict(full_series=ts, train_size=train_size)
    mse_xgb, mae_xgb, rmse_xgb, r2_xgb = calculate_metrics(test=test, preds=test_hat)
    
    if verbose:
        print(f'Results for XGB (up-to-date) on {target}')
        print(f'Mean Squared Error: {mse_xgb}')
        print(f'Root Mean Squared Error: {rmse_xgb}')
        print(f'Mean Absolute Error: {mae_xgb}')
        print(f'R^2: {r2_xgb}')
    
    return {
        'test': test,
        'predictions': test_hat,
        'mse': mse_xgb,
        'mae': mae_xgb,
        'rmse': rmse_xgb,
        'r2': r2_xgb
    }


def get_results_AR_uptodate(dataset, target, p, prop_test=0.1, verbose=True):
    """AR predictions using true y values (up-to-date approach)"""
    df = pd.read_csv(f'data/{dataset}.csv')
    ts = df[target].values
    train_size = int(len(ts) * (1 - prop_test))
    train = ts[:train_size]
    test = ts[train_size:]

    ar_md = ar_model(p=p)
    ar_md.train_model(train=train)
    test_hat = ar_md.predict(full_series=ts, train_size=train_size)
    mse_AR, mae_AR, rmse_AR, r2_AR = calculate_metrics(test=test, preds=test_hat)
    
    if verbose:
        print(f'Results for AR (up-to-date) on {target}')
        print(f'Mean Squared Error: {mse_AR}')
        print(f'Root Mean Squared Error: {rmse_AR}')
        print(f'Mean Absolute Error: {mae_AR}')
        print(f'R^2: {r2_AR}')
    
    return {
        'test': test,
        'predictions': test_hat,
        'mse': mse_AR,
        'mae': mae_AR,
        'rmse': rmse_AR,
        'r2': r2_AR
    }


def find_best_model_uptodate(dataset, target, p_candidates, prop_test=0.1):
    """Find best model using up-to-date approach"""
    results = {
        'XGB': {'best_p': None, 'best_mse': np.inf, 'metrics': {}},
        'AR': {'best_p': None, 'best_mse': np.inf, 'metrics': {}}
    }
    for p in p_candidates:
        try:
            xgb_results = get_results_xgb_uptodate(dataset, target, p, prop_test, verbose=False)
            results['XGB']['metrics'][p] = xgb_results
            if xgb_results['mse'] < results['XGB']['best_mse']:
                results['XGB']['best_mse'] = xgb_results['mse']
                results['XGB']['best_p'] = p
        except Exception as e:
            print(f"XGB MODEL FAILED AT P={p}: {e}")

        try:
            ar_results = get_results_AR_uptodate(dataset, target, p, prop_test, verbose=False)
            results['AR']['metrics'][p] = ar_results
            if ar_results['mse'] < results['AR']['best_mse']:
                results['AR']['best_mse'] = ar_results['mse']
                results['AR']['best_p'] = p
        except Exception as e:
            print(f"AR MODEL FAILED AT P={p}: {e}")

    best_xgb_p = results['XGB']['best_p']
    best_xgb = results['XGB']['metrics'][best_xgb_p]
    print(f"Best XGB Model - Up-to-date (p={best_xgb_p})")
    print(f"MSE: {best_xgb['mse']:.4f}")
    print(f"RMSE: {best_xgb['rmse']:.4f}")
    print(f"MAE: {best_xgb['mae']:.4f}")
    print(f"R_squared: {best_xgb['r2']:.4f}")
    print()
    
    best_ar_p = results['AR']['best_p']
    best_ar = results['AR']['metrics'][best_ar_p]
    print(f"Best AR Model - Up-to-date (p={best_ar_p})")
    print(f"MSE: {best_ar['mse']:.4f}")
    print(f"RMSE: {best_ar['rmse']:.4f}")
    print(f"MAE: {best_ar['mae']:.4f}")
    print(f"R_squared: {best_ar['r2']:.4f}")
    print()

    if results['XGB']['best_mse'] < results['AR']['best_mse']:
        print(f"Overall Best Model: XGB with p={best_xgb_p} (MSE: {results['XGB']['best_mse']:.4f})")
    else:
        print(f"Overall Best Model: AR with p={best_ar_p} (MSE: {results['AR']['best_mse']:.4f})")

    return results


def plot_best_uptodate(results):
    """Plots the forecast for the best XGB and AR models (up-to-date approach)"""
    plt.close()
    test = results['XGB']['metrics'][results['XGB']['best_p']]['test']
    preds_xgb = results['XGB']['metrics'][results['XGB']['best_p']]['predictions']
    preds_ar = results['AR']['metrics'][results['AR']['best_p']]['predictions']
    t = np.arange(len(test))

    best_p_xgb = results['XGB']['best_p']
    best_p_ar = results['AR']['best_p']

    sns.lineplot(x=t, y=test, label='Test', zorder=2, alpha=0.2, color='#0d00ff')
    sns.lineplot(x=t, y=preds_xgb, label=f'XGB Predictions (p={best_p_xgb})', zorder=2, alpha=0.2, color='#ff00bf')
    sns.lineplot(x=t, y=preds_ar, label=f'AR Predictions (p={best_p_ar})', zorder=2, alpha=0.2, color='#59ff00')

    plt.title(f'Best XGBoost and AR Models (Up-to-date Approach)')
    plt.ylabel('Power Consumption (MW)')
    plt.xlabel('Time')
    plt.grid(alpha=0.4, zorder=1)
    plt.legend()
    plt.savefig('figs/plot-uptodate-comparison.png')
