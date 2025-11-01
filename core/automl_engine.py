# core/automl_engine.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

class AutoMLEngine:
    def __init__(self):
        self.classification_models = {
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
            'LightGBM': LGBMClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0),
            'SVM': SVC(),
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        
        self.regression_models = {
            'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor(),
            'CatBoost': CatBoostRegressor(verbose=0),
            'SVR': SVR(),
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'K-Nearest Neighbors': KNeighborsRegressor()
        }
        
        self.metrics = {
            'Accuracy': accuracy_score,
            'F1 Score': f1_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'AUC': roc_auc_score,
            'MSE': mean_squared_error,
            'MAE': mean_absolute_error
        }
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, problem_type: str, optimization_metric: str,
                    max_training_time: int = 1800, enable_ensemble: bool = True, 
                    enable_feature_engineering: bool = True) -> Dict[str, Any]:
        
        start_time = time.time()
        trained_models = {}
        
        if problem_type == "Classification":
            models = self.classification_models
        else:
            models = self.regression_models
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if enable_feature_engineering:
            X_train, X_test = self._apply_feature_engineering(X_train, X_test)
        
        for model_name, model in models.items():
            if time.time() - start_time > max_training_time:
                break
            
            try:
                model_start_time = time.time()
                
                model.fit(X_train, y_train)
                training_time = time.time() - model_start_time
                
                y_pred = model.predict(X_test)
                
                if problem_type == "Classification":
                    score = self._calculate_classification_metric(y_test, y_pred, optimization_metric)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                else:
                    score = self._calculate_regression_metric(y_test, y_pred, optimization_metric)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                    cv_scores = -cv_scores
                
                feature_importance = self._get_feature_importance(model, X_train.columns)
                learning_curve = self._generate_learning_curve(model, X_train, y_train)
                
                trained_models[model_name] = {
                    'model': model,
                    'algorithm': model_name,
                    'score': score,
                    'cv_score': np.mean(cv_scores),
                    'training_time': training_time,
                    'feature_importance': feature_importance,
                    'learning_curve': learning_curve
                }
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        if enable_ensemble and len(trained_models) > 1:
            ensemble_model = self._create_ensemble(trained_models, problem_type)
            ensemble_score = self._evaluate_ensemble(ensemble_model, X_test, y_test, problem_type, optimization_metric)
            
            trained_models['Ensemble'] = {
                'model': ensemble_model,
                'algorithm': 'Ensemble',
                'score': ensemble_score,
                'cv_score': ensemble_score,
                'training_time': sum([m['training_time'] for m in trained_models.values()]),
                'feature_importance': self._get_ensemble_feature_importance(trained_models),
                'learning_curve': None
            }
        
        return dict(sorted(trained_models.items(), key=lambda x: x[1]['score'], reverse=True))
    
    def train_multiple_models(self, X: pd.DataFrame, y: pd.Series, problem_type: str, 
                             optimization_metric: str) -> Dict[str, Any]:
        
        models = {}
        
        if problem_type == "Classification":
            additional_models = {
                'Extra Trees': RandomForestClassifier(n_estimators=50),
                'Gradient Boosting': GradientBoostingClassifier(),
                'AdaBoost': GradientBoostingClassifier()
            }
        else:
            additional_models = {
                'Extra Trees': RandomForestRegressor(n_estimators=50),
                'Gradient Boosting': GradientBoostingRegressor(),
                'AdaBoost': GradientBoostingRegressor()
            }
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in additional_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if problem_type == "Classification":
                    score = self._calculate_classification_metric(y_test, y_pred, optimization_metric)
                else:
                    score = self._calculate_regression_metric(y_test, y_pred, optimization_metric)
                
                models[model_name] = {
                    'model': model,
                    'algorithm': model_name,
                    'score': score,
                    'cv_score': score,
                    'training_time': 0.0,
                    'feature_importance': self._get_feature_importance(model, X_train.columns),
                    'learning_curve': None
                }
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        return models
    
    def compare_models(self, trained_models: Dict[str, Any]) -> pd.DataFrame:
        comparison_data = []
        
        for model_name, model_info in trained_models.items():
            comparison_data.append({
                'Model': model_name,
                'Algorithm': model_info['algorithm'],
                'Score': model_info['score'],
                'CV Score': model_info['cv_score'],
                'Training Time (s)': model_info['training_time']
            })
        
        return pd.DataFrame(comparison_data).sort_values('Score', ascending=False)
    
    def _apply_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_engineered = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_engineered = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_engineered, X_test_engineered
    
    def _calculate_classification_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        metric_func = self.metrics.get(metric, accuracy_score)
        
        if metric == 'AUC':
            return metric_func(y_true, y_pred)
        else:
            return metric_func(y_true, y_pred, average='weighted')
    
    def _calculate_regression_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        metric_func = self.metrics.get(metric, mean_squared_error)
        return metric_func(y_true, y_pred)
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return {}
            
            feature_importance = dict(zip(feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        except:
            return {}
    
    def _generate_learning_curve(self, model, X: pd.DataFrame, y: pd.Series):
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            test_scores = []
            
            for train_size in train_sizes:
                n_samples = int(train_size * len(X))
                X_subset = X.iloc[:n_samples]
                y_subset = y.iloc[:n_samples]
                
                X_train, X_val, y_train, y_val = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
                
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_val, y_val)
                
                train_scores.append(train_score)
                test_scores.append(test_score)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_sizes, y=train_scores, name='Training Score', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=train_sizes, y=test_scores, name='Validation Score', line=dict(color='red')))
            fig.update_layout(title='Learning Curve', xaxis_title='Training Size', yaxis_title='Score')
            
            return fig
        
        except:
            return None
    
    def _create_ensemble(self, trained_models: Dict[str, Any], problem_type: str):
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        estimators = [(name, model_info['model']) for name, model_info in trained_models.items()]
        
        if problem_type == "Classification":
            return VotingClassifier(estimators=estimators, voting='soft')
        else:
            return VotingRegressor(estimators=estimators)
    
    def _evaluate_ensemble(self, ensemble_model, X_test: pd.DataFrame, y_test: pd.Series, 
                          problem_type: str, optimization_metric: str) -> float:
        
        ensemble_model.fit(X_test, y_test)
        y_pred = ensemble_model.predict(X_test)
        
        if problem_type == "Classification":
            return self._calculate_classification_metric(y_test, y_pred, optimization_metric)
        else:
            return self._calculate_regression_metric(y_test, y_pred, optimization_metric)
    
    def _get_ensemble_feature_importance(self, trained_models: Dict[str, Any]) -> Dict[str, float]:
        all_importances = {}
        
        for model_name, model_info in trained_models.items():
            if 'feature_importance' in model_info and model_info['feature_importance']:
                for feature, importance in model_info['feature_importance'].items():
                    if feature in all_importances:
                        all_importances[feature] += importance
                    else:
                        all_importances[feature] = importance
        
        total = sum(all_importances.values())
        if total > 0:
            all_importances = {k: v/total for k, v in all_importances.items()}
        
        return dict(sorted(all_importances.items(), key=lambda x: x[1], reverse=True))

class AdvancedAutoMLEngine(AutoMLEngine):
    def __init__(self):
        super().__init__()
    
    def train_with_metalearning(self, X: pd.DataFrame, y: pd.Series, problem_type: str, 
                               optimization_metric: str) -> Dict[str, Any]:
        
        dataset_characteristics = self._extract_dataset_characteristics(X, y)
        recommended_models = self._recommend_models(dataset_characteristics, problem_type)
        
        trained_models = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name in recommended_models:
            if model_name in self.classification_models:
                model = self.classification_models[model_name]
            elif model_name in self.regression_models:
                model = self.regression_models[model_name]
            else:
                continue
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if problem_type == "Classification":
                    score = self._calculate_classification_metric(y_test, y_pred, optimization_metric)
                else:
                    score = self._calculate_regression_metric(y_test, y_pred, optimization_metric)
                
                trained_models[model_name] = {
                    'model': model,
                    'algorithm': model_name,
                    'score': score,
                    'cv_score': score,
                    'training_time': 0.0,
                    'feature_importance': self._get_feature_importance(model, X_train.columns),
                    'learning_curve': None
                }
                
            except Exception as e:
                continue
        
        return trained_models
    
    def _extract_dataset_characteristics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        characteristics = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_numeric_features': X.select_dtypes(include=[np.number]).shape[1],
            'n_categorical_features': X.select_dtypes(include=['object']).shape[1],
            'sparsity': (X == 0).sum().sum() / (X.shape[0] * X.shape[1]),
            'target_cardinality': len(np.unique(y)) if hasattr(y, 'unique') else 0
        }
        
        return characteristics
    
    def _recommend_models(self, characteristics: Dict[str, Any], problem_type: str) -> List[str]:
        recommended = []
        
        n_samples = characteristics['n_samples']
        n_features = characteristics['n_features']
        
        if problem_type == "Classification":
            if n_samples < 1000:
                recommended.extend(['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree'])
            elif n_samples < 10000:
                recommended.extend(['Random Forest', 'XGBoost', 'LightGBM'])
            else:
                recommended.extend(['LightGBM', 'CatBoost', 'XGBoost'])
        else:
            if n_samples < 1000:
                recommended.extend(['Linear Regression', 'K-Nearest Neighbors', 'Decision Tree'])
            elif n_samples < 10000:
                recommended.extend(['Random Forest', 'XGBoost', 'LightGBM'])
            else:
                recommended.extend(['LightGBM', 'CatBoost', 'XGBoost'])
        
        return recommended