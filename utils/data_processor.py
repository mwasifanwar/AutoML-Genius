# utils/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.feature_selector = None
        self.pca = None
        self.target_column = None
        self.feature_names = None
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str, 
                       handle_missing: bool = True, encode_categorical: bool = True,
                       scale_features: bool = True, feature_selection: bool = False,
                       n_features: int = None) -> pd.DataFrame:
        
        self.target_column = target_column
        
        df_processed = df.copy()
        
        if handle_missing:
            df_processed = self._handle_missing_values(df_processed)
        
        if encode_categorical:
            df_processed = self._encode_categorical_features(df_processed)
        
        if scale_features:
            df_processed = self._scale_features(df_processed)
        
        if feature_selection and n_features:
            df_processed = self._select_features(df_processed, n_features)
        
        self.feature_names = [col for col in df_processed.columns if col != target_column]
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].isnull().sum() / len(df_processed) > 0.5:
                    df_processed = df_processed.drop(columns=[col])
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown')
        
        return df_processed
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == self.target_column:
                if df_processed[col].nunique() <= 10:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                continue
            
            if df_processed[col].nunique() <= 10:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed = df_processed.drop(columns=[col])
        
        return df_processed
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        if len(numeric_cols) > 0:
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        
        return df_processed
    
    def _select_features(self, df: pd.DataFrame, n_features: int) -> pd.DataFrame:
        df_processed = df.copy()
        
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        if y.dtype == 'object' or y.nunique() <= 10:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        else:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
        
        X_selected = self.feature_selector.fit_transform(X, y)
        
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        selected_features.append(self.target_column)
        
        return df_processed[selected_features]
    
    def apply_pca(self, df: pd.DataFrame, n_components: int = None) -> pd.DataFrame:
        df_processed = df.copy()
        
        X = df_processed.drop(columns=[self.target_column])
        
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        pca_columns = [f'PC_{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df_processed.index)
        df_pca[self.target_column] = df_processed[self.target_column].values
        
        return df_pca
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        summary = {
            'target_column': self.target_column,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'missing_values_handled': True,
            'categorical_encoded': len(self.label_encoders) > 0,
            'features_scaled': True,
            'feature_selection_applied': self.feature_selector is not None,
            'pca_applied': self.pca is not None
        }
        
        if self.feature_selector:
            summary['selected_features'] = self.feature_names
        
        if self.pca:
            summary['pca_components'] = self.pca.n_components_
            summary['explained_variance'] = self.pca.explained_variance_ratio_.sum()
        
        return summary
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        for col in categorical_cols:
            if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                new_categories = set(df_processed[col].unique()) - set(encoder.classes_)
                if new_categories:
                    for category in new_categories:
                        df_processed.loc[df_processed[col] == category, col] = encoder.classes_[0]
                df_processed[col] = encoder.transform(df_processed[col])
        
        numeric_cols_to_scale = [col for col in numeric_cols if col in df_processed.columns]
        if numeric_cols_to_scale:
            df_processed[numeric_cols_to_scale] = self.scaler.transform(df_processed[numeric_cols_to_scale])
        
        if self.feature_selector:
            available_features = [col for col in self.feature_names if col in df_processed.columns]
            missing_features = set(self.feature_names) - set(available_features)
            
            for feature in missing_features:
                df_processed[feature] = 0
            
            df_processed = df_processed[self.feature_names]
        
        return df_processed

class AdvancedDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.feature_engineering = {}
    
    def engineer_features(self, df: pd.DataFrame, methods: List[str] = None) -> pd.DataFrame:
        df_processed = df.copy()
        
        if methods is None:
            methods = ['polynomial', 'interaction', 'datetime', 'aggregation']
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        if 'polynomial' in methods and len(numeric_cols) >= 2:
            df_processed = self._create_polynomial_features(df_processed, numeric_cols)
        
        if 'interaction' in methods and len(numeric_cols) >= 2:
            df_processed = self._create_interaction_features(df_processed, numeric_cols)
        
        if 'datetime' in methods:
            df_processed = self._extract_datetime_features(df_processed)
        
        if 'aggregation' in methods and len(numeric_cols) >= 3:
            df_processed = self._create_aggregation_features(df_processed, numeric_cols)
        
        return df_processed
    
    def _create_polynomial_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        df_processed = df.copy()
        
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df_processed[numeric_cols])
        
        feature_names = poly.get_feature_names_out(numeric_cols)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_processed.index)
        
        poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()]
        
        df_processed = pd.concat([df_processed, poly_df], axis=1)
        
        self.feature_engineering['polynomial_features'] = list(poly_df.columns)
        
        return df_processed
    
    def _create_interaction_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        df_processed = df.copy()
        
        interaction_features = []
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                interaction_name = f"{col1}_x_{col2}"
                df_processed[interaction_name] = df_processed[col1] * df_processed[col2]
                interaction_features.append(interaction_name)
        
        self.feature_engineering['interaction_features'] = interaction_features
        
        return df_processed
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            df_processed[f'{col}_year'] = df_processed[col].dt.year
            df_processed[f'{col}_month'] = df_processed[col].dt.month
            df_processed[f'{col}_day'] = df_processed[col].dt.day
            df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
            df_processed[f'{col}_hour'] = df_processed[col].dt.hour
            
            self.feature_engineering['datetime_features'] = self.feature_engineering.get('datetime_features', []) + [
                f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek', f'{col}_hour'
            ]
        
        return df_processed
    
    def _create_aggregation_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        df_processed = df.copy()
        
        df_processed['feature_mean'] = df_processed[numeric_cols].mean(axis=1)
        df_processed['feature_std'] = df_processed[numeric_cols].std(axis=1)
        df_processed['feature_sum'] = df_processed[numeric_cols].sum(axis=1)
        df_processed['feature_max'] = df_processed[numeric_cols].max(axis=1)
        df_processed['feature_min'] = df_processed[numeric_cols].min(axis=1)
        
        self.feature_engineering['aggregation_features'] = [
            'feature_mean', 'feature_std', 'feature_sum', 'feature_max', 'feature_min'
        ]
        
        return df_processed
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        outliers = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100,
                    'method': 'IQR'
                }
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        for col in numeric_cols:
            if method == 'cap':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_processed[col] = np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
                df_processed[col] = np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col])
            
            elif method == 'remove':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
        
        return df_processed