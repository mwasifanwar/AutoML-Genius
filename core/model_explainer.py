# core/model_explainer.py
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, List
import lime
import lime.lime_tabular

class ModelExplainer:
    def __init__(self):
        self.explanation_methods = {
            'SHAP': self._shap_explanation,
            'LIME': self._lime_explanation,
            'Partial Dependence': self._partial_dependence_explanation,
            'Feature Importance': self._feature_importance_explanation
        }
    
    def explain_model(self, model, X: pd.DataFrame, y: pd.Series = None, 
                     method: str = "SHAP", sample_size: int = 1000) -> Dict[str, Any]:
        
        if method not in self.explanation_methods:
            raise ValueError(f"Unsupported explanation method: {method}")
        
        explanation_func = self.explanation_methods[method]
        return explanation_func(model, X, y, sample_size)
    
    def _shap_explanation(self, model, X: pd.DataFrame, y: pd.Series = None, 
                         sample_size: int = 1000) -> Dict[str, Any]:
        
        if sample_size < len(X):
            X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
        else:
            X_sample = X
        
        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
            
            summary_plot = self._create_shap_summary_plot(shap_values, X_sample)
            feature_importance = self._create_shap_feature_importance(shap_values, X_sample)
            dependence_plots = self._create_shap_dependence_plots(shap_values, X_sample)
            
            return {
                'summary_plot': summary_plot,
                'feature_importance': feature_importance,
                'dependence_plots': dependence_plots,
                'shap_values': shap_values,
                'explainer': explainer
            }
        
        except Exception as e:
            return self._fallback_explanation(model, X_sample)
    
    def _lime_explanation(self, model, X: pd.DataFrame, y: pd.Series = None,
                         sample_size: int = 1000) -> Dict[str, Any]:
        
        if sample_size < len(X):
            X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
        else:
            X_sample = X
        
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_sample.values,
                feature_names=X_sample.columns.tolist(),
                mode='classification' if hasattr(model, 'predict_proba') else 'regression'
            )
            
            instance_explanations = []
            for i in range(min(5, len(X_sample))):
                exp = explainer.explain_instance(
                    X_sample.iloc[i].values,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=10
                )
                instance_explanations.append(exp)
            
            lime_plot = self._create_lime_plot(instance_explanations, X_sample)
            
            return {
                'lime_plot': lime_plot,
                'instance_explanations': instance_explanations,
                'explainer': explainer
            }
        
        except Exception as e:
            return self._fallback_explanation(model, X_sample)
    
    def _partial_dependence_explanation(self, model, X: pd.DataFrame, y: pd.Series = None,
                                      sample_size: int = 1000) -> Dict[str, Any]:
        
        if sample_size < len(X):
            X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
        else:
            X_sample = X
        
        try:
            top_features = self._get_top_features(model, X_sample, n_features=5)
            pdp_plots = {}
            
            for feature in top_features:
                pdp_results = partial_dependence(
                    model, X_sample, [feature], kind='average'
                )
                
                pdp_plot = self._create_pdp_plot(pdp_results, feature)
                pdp_plots[feature] = pdp_plot
            
            return {
                'pdp_plots': pdp_plots,
                'top_features': top_features
            }
        
        except Exception as e:
            return self._fallback_explanation(model, X_sample)
    
    def _feature_importance_explanation(self, model, X: pd.DataFrame, y: pd.Series = None,
                                      sample_size: int = 1000) -> Dict[str, Any]:
        
        if sample_size < len(X):
            X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
        else:
            X_sample = X
        
        try:
            feature_importance = self._get_feature_importance(model, X_sample)
            importance_plot = self._create_feature_importance_plot(feature_importance)
            
            return {
                'feature_importance': importance_plot,
                'importance_scores': feature_importance
            }
        
        except Exception as e:
            return self._fallback_explanation(model, X_sample)
    
    def _create_shap_summary_plot(self, shap_values, X: pd.DataFrame):
        feature_names = X.columns.tolist()
        
        if len(shap_values.shape) == 2:
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
        else:
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=True)
        
        fig = go.Figure()
        
        for feature in mean_abs_shap.index:
            fig.add_trace(go.Bar(
                y=[feature],
                x=[mean_abs_shap[feature]],
                orientation='h',
                name=feature
            ))
        
        fig.update_layout(
            title='SHAP Feature Importance',
            xaxis_title='Mean |SHAP value|',
            yaxis_title='Features',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _create_shap_feature_importance(self, shap_values, X: pd.DataFrame):
        feature_names = X.columns.tolist()
        
        if len(shap_values.shape) == 2:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='SHAP Feature Importance'
        )
        
        fig.update_layout(height=400)
        return fig
    
    def _create_shap_dependence_plots(self, shap_values, X: pd.DataFrame):
        plots = {}
        feature_names = X.columns.tolist()
        
        for i, feature in enumerate(feature_names[:3]):
            if len(shap_values.shape) == 2:
                shap_feature = shap_values[:, i]
            else:
                shap_feature = shap_values[:, i]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=X[feature],
                y=shap_feature,
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color=shap_feature,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title=f'SHAP Dependence Plot: {feature}',
                xaxis_title=feature,
                yaxis_title='SHAP Value',
                height=400
            )
            
            plots[feature] = fig
        
        return plots
    
    def _create_lime_plot(self, explanations, X: pd.DataFrame):
        if not explanations:
            return None
        
        first_exp = explanations[0]
        features = []
        scores = []
        
        for feature, score in first_exp.as_list():
            features.append(feature)
            scores.append(score)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=scores,
            y=features,
            orientation='h'
        ))
        
        fig.update_layout(
            title='LIME Feature Importance for First Instance',
            xaxis_title='LIME Score',
            yaxis_title='Features',
            height=400
        )
        
        return fig
    
    def _create_pdp_plot(self, pdp_results, feature: str):
        values = pdp_results['values'][0]
        averages = pdp_results['average'][0]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=values,
            y=averages,
            mode='lines+markers',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title=f'Partial Dependence Plot: {feature}',
            xaxis_title=feature,
            yaxis_title='Partial Dependence',
            height=400
        )
        
        return fig
    
    def _create_feature_importance_plot(self, feature_importance: Dict[str, float]):
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        sorted_indices = np.argsort(importances)
        features = [features[i] for i in sorted_indices]
        importances = [importances[i] for i in sorted_indices]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importances,
            y=features,
            orientation='h'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=400
        )
        
        return fig
    
    def _get_top_features(self, model, X: pd.DataFrame, n_features: int = 5) -> List[str]:
        feature_importance = self._get_feature_importance(model, X)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in sorted_features[:n_features]]
    
    def _get_feature_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                from sklearn.inspection import permutation_importance
                result = permutation_importance(model, X, np.random.randn(len(X)), n_repeats=10)
                importances = result.importances_mean
            
            feature_importance = dict(zip(X.columns, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        except:
            return {col: 1.0 for col in X.columns}
    
    def _fallback_explanation(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        feature_importance = self._get_feature_importance(model, X)
        importance_plot = self._create_feature_importance_plot(feature_importance)
        
        return {
            'feature_importance': importance_plot,
            'importance_scores': feature_importance,
            'fallback_used': True
        }

class AdvancedModelExplainer(ModelExplainer):
    def __init__(self):
        super().__init__()
    
    def explain_with_confidence(self, model, X: pd.DataFrame, y: pd.Series = None,
                              method: str = "SHAP", confidence_level: float = 0.95) -> Dict[str, Any]:
        
        base_explanation = self.explain_model(model, X, y, method)
        
        if method == "SHAP" and 'shap_values' in base_explanation:
            confidence_intervals = self._calculate_shap_confidence(
                base_explanation['shap_values'], confidence_level
            )
            base_explanation['confidence_intervals'] = confidence_intervals
        
        return base_explanation
    
    def compare_explanations(self, model1, model2, X: pd.DataFrame, 
                           method: str = "SHAP") -> Dict[str, Any]:
        
        explanation1 = self.explain_model(model1, X, method=method)
        explanation2 = self.explain_model(model2, X, method=method)
        
        comparison_plot = self._create_comparison_plot(explanation1, explanation2, method)
        
        return {
            'model1_explanation': explanation1,
            'model2_explanation': explanation2,
            'comparison_plot': comparison_plot
        }
    
    def _calculate_shap_confidence(self, shap_values, confidence_level: float = 0.95):
        if len(shap_values.shape) == 2:
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100
            
            lower_bounds = np.percentile(shap_values, lower_percentile, axis=0)
            upper_bounds = np.percentile(shap_values, upper_percentile, axis=0)
            
            return {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'confidence_level': confidence_level
            }
        
        return {}
    
    def _create_comparison_plot(self, explanation1: Dict[str, Any], 
                              explanation2: Dict[str, Any], method: str):
        
        if method == "Feature Importance":
            scores1 = explanation1.get('importance_scores', {})
            scores2 = explanation2.get('importance_scores', {})
            
            features = list(scores1.keys())[:10]
            importance1 = [scores1.get(f, 0) for f in features]
            importance2 = [scores2.get(f, 0) for f in features]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Model 1',
                x=features,
                y=importance1
            ))
            
            fig.add_trace(go.Bar(
                name='Model 2',
                x=features,
                y=importance2
            ))
            
            fig.update_layout(
                title='Feature Importance Comparison',
                xaxis_title='Features',
                yaxis_title='Importance',
                barmode='group',
                height=400
            )
            
            return fig
        
        return None