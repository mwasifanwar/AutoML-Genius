# core/hyperparameter_optimizer.py
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Dict, Any, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class HyperparameterOptimizer:
    def __init__(self):
        self.optimization_history = []
    
    def optimize(self, model, X: pd.DataFrame, y: pd.Series, method: str = "Bayesian Optimization", 
                n_trials: int = 100, cv_folds: int = 5) -> Dict[str, Any]:
        
        if method == "Bayesian Optimization":
            return self._bayesian_optimization(model, X, y, n_trials, cv_folds)
        elif method == "Genetic Algorithm":
            return self._genetic_algorithm(model, X, y, n_trials, cv_folds)
        elif method == "Random Search":
            return self._random_search(model, X, y, n_trials, cv_folds)
        elif method == "Grid Search":
            return self._grid_search(model, X, y, cv_folds)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _bayesian_optimization(self, model, X: pd.DataFrame, y: pd.Series, 
                             n_trials: int, cv_folds: int) -> Dict[str, Any]:
        
        model_type = type(model).__name__
        
        def objective(trial):
            params = self._suggest_hyperparameters(trial, model_type)
            optimized_model = self._create_model_with_params(model_type, params)
            
            scores = cross_val_score(optimized_model, X, y, cv=cv_folds, scoring='accuracy')
            score = np.mean(scores)
            
            self.optimization_history.append({
                'trial': len(self.optimization_history) + 1,
                'score': score,
                'params': params
            })
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_model = self._create_model_with_params(model_type, study.best_params)
        best_model.fit(X, y)
        
        optimization_plot = self._create_optimization_plot()
        
        return {
            'best_model': best_model,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'improvement': study.best_value - self.optimization_history[0]['score'] if self.optimization_history else 0,
            'optimization_history': optimization_plot,
            'study': study
        }
    
    def _genetic_algorithm(self, model, X: pd.DataFrame, y: pd.Series, 
                          n_trials: int, cv_folds: int) -> Dict[str, Any]:
        
        model_type = type(model).__name__
        best_score = -np.inf
        best_params = None
        
        for trial in range(n_trials):
            params = self._generate_random_params(model_type)
            optimized_model = self._create_model_with_params(model_type, params)
            
            try:
                scores = cross_val_score(optimized_model, X, y, cv=cv_folds, scoring='accuracy')
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                self.optimization_history.append({
                    'trial': trial + 1,
                    'score': score,
                    'params': params
                })
                
            except Exception as e:
                continue
        
        if best_params is None:
            raise ValueError("No valid parameters found during genetic optimization")
        
        best_model = self._create_model_with_params(model_type, best_params)
        best_model.fit(X, y)
        
        optimization_plot = self._create_optimization_plot()
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'improvement': best_score - self.optimization_history[0]['score'] if self.optimization_history else 0,
            'optimization_history': optimization_plot
        }
    
    def _random_search(self, model, X: pd.DataFrame, y: pd.Series, 
                      n_trials: int, cv_folds: int) -> Dict[str, Any]:
        
        model_type = type(model).__name__
        best_score = -np.inf
        best_params = None
        
        for trial in range(n_trials):
            params = self._generate_random_params(model_type)
            optimized_model = self._create_model_with_params(model_type, params)
            
            try:
                scores = cross_val_score(optimized_model, X, y, cv=cv_folds, scoring='accuracy')
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                self.optimization_history.append({
                    'trial': trial + 1,
                    'score': score,
                    'params': params
                })
                
            except Exception as e:
                continue
        
        if best_params is None:
            raise ValueError("No valid parameters found during random search")
        
        best_model = self._create_model_with_params(model_type, best_params)
        best_model.fit(X, y)
        
        optimization_plot = self._create_optimization_plot()
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'improvement': best_score - self.optimization_history[0]['score'] if self.optimization_history else 0,
            'optimization_history': optimization_plot
        }
    
    def _grid_search(self, model, X: pd.DataFrame, y: pd.Series, 
                    cv_folds: int) -> Dict[str, Any]:
        
        model_type = type(model).__name__
        param_grid = self._get_param_grid(model_type)
        best_score = -np.inf
        best_params = None
        
        from sklearn.model_selection import ParameterGrid
        grid = ParameterGrid(param_grid)
        
        for params in list(grid)[:100]:
            optimized_model = self._create_model_with_params(model_type, params)
            
            try:
                scores = cross_val_score(optimized_model, X, y, cv=cv_folds, scoring='accuracy')
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                self.optimization_history.append({
                    'trial': len(self.optimization_history) + 1,
                    'score': score,
                    'params': params
                })
                
            except Exception as e:
                continue
        
        if best_params is None:
            raise ValueError("No valid parameters found during grid search")
        
        best_model = self._create_model_with_params(model_type, best_params)
        best_model.fit(X, y)
        
        optimization_plot = self._create_optimization_plot()
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'improvement': best_score - self.optimization_history[0]['score'] if self.optimization_history else 0,
            'optimization_history': optimization_plot
        }
    
    def _suggest_hyperparameters(self, trial, model_type: str) -> Dict[str, Any]:
        params = {}
        
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        elif model_type in ['XGBClassifier', 'XGBRegressor']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            params['gamma'] = trial.suggest_float('gamma', 0, 5)
        
        elif model_type in ['LGBMClassifier', 'LGBMRegressor']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        
        elif model_type in ['SVC', 'SVR']:
            params['C'] = trial.suggest_float('C', 0.1, 10.0)
            params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            if params['kernel'] == 'rbf':
                params['gamma'] = trial.suggest_float('gamma', 0.01, 1.0)
        
        elif model_type in ['LogisticRegression', 'LinearRegression']:
            if model_type == 'LogisticRegression':
                params['C'] = trial.suggest_float('C', 0.1, 10.0)
                params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2'])
                params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        
        return params
    
    def _generate_random_params(self, model_type: str) -> Dict[str, Any]:
        params = {}
        
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor']:
            params['n_estimators'] = np.random.randint(50, 500)
            params['max_depth'] = np.random.randint(3, 20)
            params['min_samples_split'] = np.random.randint(2, 20)
            params['min_samples_leaf'] = np.random.randint(1, 10)
            params['max_features'] = np.random.choice(['sqrt', 'log2', None])
        
        elif model_type in ['XGBClassifier', 'XGBRegressor']:
            params['n_estimators'] = np.random.randint(50, 500)
            params['max_depth'] = np.random.randint(3, 15)
            params['learning_rate'] = np.random.uniform(0.01, 0.3)
            params['subsample'] = np.random.uniform(0.6, 1.0)
            params['colsample_bytree'] = np.random.uniform(0.6, 1.0)
            params['gamma'] = np.random.uniform(0, 5)
        
        elif model_type in ['LGBMClassifier', 'LGBMRegressor']:
            params['n_estimators'] = np.random.randint(50, 500)
            params['max_depth'] = np.random.randint(3, 15)
            params['learning_rate'] = np.random.uniform(0.01, 0.3)
            params['num_leaves'] = np.random.randint(20, 100)
            params['subsample'] = np.random.uniform(0.6, 1.0)
            params['colsample_bytree'] = np.random.uniform(0.6, 1.0)
        
        return params
    
    def _get_param_grid(self, model_type: str) -> Dict[str, Any]:
        param_grid = {}
        
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor']:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        elif model_type in ['XGBClassifier', 'XGBRegressor']:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        return param_grid
    
    def _create_model_with_params(self, model_type: str, params: Dict[str, Any]):
        if model_type == 'RandomForestClassifier':
            return RandomForestClassifier(**params, random_state=42)
        elif model_type == 'RandomForestRegressor':
            return RandomForestRegressor(**params, random_state=42)
        elif model_type == 'XGBClassifier':
            return XGBClassifier(**params, random_state=42)
        elif model_type == 'XGBRegressor':
            return XGBRegressor(**params, random_state=42)
        elif model_type == 'LGBMClassifier':
            return LGBMClassifier(**params, random_state=42)
        elif model_type == 'LGBMRegressor':
            return LGBMRegressor(**params, random_state=42)
        elif model_type == 'SVC':
            return SVC(**params, random_state=42)
        elif model_type == 'SVR':
            return SVR(**params)
        elif model_type == 'LogisticRegression':
            return LogisticRegression(**params, random_state=42)
        elif model_type == 'LinearRegression':
            return LinearRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_optimization_plot(self):
        if not self.optimization_history:
            return None
        
        trials = [h['trial'] for h in self.optimization_history]
        scores = [h['score'] for h in self.optimization_history]
        
        best_scores = []
        current_best = -np.inf
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(
            go.Scatter(x=trials, y=scores, mode='markers', name='Trial Score',
                      marker=dict(color='blue', size=8, opacity=0.6))
        )
        
        fig.add_trace(
            go.Scatter(x=trials, y=best_scores, mode='lines', name='Best Score',
                      line=dict(color='red', width=3))
        )
        
        fig.update_layout(
            title='Hyperparameter Optimization Progress',
            xaxis_title='Trial',
            yaxis_title='Score',
            height=400
        )
        
        return fig

class AdvancedHyperparameterOptimizer(HyperparameterOptimizer):
    def __init__(self):
        super().__init__()
    
    def optimize_with_early_stopping(self, model, X: pd.DataFrame, y: pd.Series, 
                                   method: str = "Bayesian Optimization", n_trials: int = 100,
                                   patience: int = 20, cv_folds: int = 5) -> Dict[str, Any]:
        
        if method != "Bayesian Optimization":
            return self.optimize(model, X, y, method, n_trials, cv_folds)
        
        model_type = type(model).__name__
        best_score = -np.inf
        no_improvement_count = 0
        
        def objective(trial):
            nonlocal best_score, no_improvement_count
            
            params = self._suggest_hyperparameters(trial, model_type)
            optimized_model = self._create_model_with_params(model_type, params)
            
            scores = cross_val_score(optimized_model, X, y, cv=cv_folds, scoring='accuracy')
            score = np.mean(scores)
            
            self.optimization_history.append({
                'trial': len(self.optimization_history) + 1,
                'score': score,
                'params': params
            })
            
            if score > best_score:
                best_score = score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                trial.study.stop()
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_model = self._create_model_with_params(model_type, study.best_params)
        best_model.fit(X, y)
        
        optimization_plot = self._create_optimization_plot()
        
        return {
            'best_model': best_model,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'improvement': study.best_value - self.optimization_history[0]['score'] if self.optimization_history else 0,
            'optimization_history': optimization_plot,
            'early_stopping_triggered': no_improvement_count >= patience
        }
    
    def multi_objective_optimization(self, model, X: pd.DataFrame, y: pd.Series,
                                   objectives: list, n_trials: int = 100) -> Dict[str, Any]:
        
        model_type = type(model).__name__
        
        def objective(trial):
            params = self._suggest_hyperparameters(trial, model_type)
            optimized_model = self._create_model_with_params(model_type, params)
            
            scores = {}
            for obj in objectives:
                if obj == 'accuracy':
                    cv_scores = cross_val_score(optimized_model, X, y, cv=5, scoring='accuracy')
                    scores['accuracy'] = np.mean(cv_scores)
                elif obj == 'training_time':
                    import time
                    start_time = time.time()
                    optimized_model.fit(X, y)
                    scores['training_time'] = time.time() - start_time
            
            return scores['accuracy'], -scores.get('training_time', 0)
        
        study = optuna.create_study(directions=['maximize', 'minimize'])
        study.optimize(objective, n_trials=n_trials)
        
        best_trial = max(study.best_trials, key=lambda t: t.values[0])
        
        best_model = self._create_model_with_params(model_type, best_trial.params)
        best_model.fit(X, y)
        
        return {
            'best_model': best_model,
            'best_params': best_trial.params,
            'best_scores': best_trial.values,
            'pareto_front': study.best_trials
        }