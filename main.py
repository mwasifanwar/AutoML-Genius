# main.py
import streamlit as st
import pandas as pd
import numpy as np
from core.automl_engine import AutoMLEngine
from core.hyperparameter_optimizer import HyperparameterOptimizer
from core.model_explainer import ModelExplainer
from core.code_generator import CodeGenerator
from utils.data_processor import DataProcessor
from utils.config import load_config

st.set_page_config(
    page_title="AutoML Genius - Next Generation Automated Machine Learning",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'automl_engine' not in st.session_state:
        st.session_state.automl_engine = None
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}

def load_components():
    with st.spinner("🚀 Loading AutoML Engine..."):
        if st.session_state.automl_engine is None:
            st.session_state.automl_engine = AutoMLEngine()
        if st.session_state.data_processor is None:
            st.session_state.data_processor = DataProcessor()

def main():
    st.title("🚀 AutoML Genius - Next Generation Automated Machine Learning")
    st.markdown("Automated machine learning with model explanation, hyperparameter optimization, and deployment code generation")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        problem_type = st.selectbox(
            "Problem Type",
            ["Classification", "Regression", "Clustering", "Time Series"],
            help="Select the machine learning problem type"
        )
        
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Accuracy", "F1 Score", "Precision", "Recall", "AUC", "MSE", "MAE"],
            help="Select the primary optimization metric"
        )
        
        st.subheader("AutoML Parameters")
        max_training_time = st.slider("Max Training Time (minutes)", 1, 120, 30)
        enable_ensemble = st.checkbox("Enable Ensemble Learning", value=True)
        enable_feature_engineering = st.checkbox("Enable Feature Engineering", value=True)
        
        st.subheader("Advanced Options")
        enable_bayesian_optimization = st.checkbox("Bayesian Optimization", value=True)
        enable_model_explanation = st.checkbox("Model Explanation", value=True)
        generate_deployment_code = st.checkbox("Generate Deployment Code", value=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Upload", "🤖 AutoML Training", "⚡ Hyperparameter Tuning", "🔍 Model Explanation", "🚀 Deployment"])
    
    with tab1:
        st.header("Data Upload & Preprocessing")
        
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=['csv', 'parquet', 'xlsx', 'json'],
            help="Upload your dataset in CSV, Parquet, Excel, or JSON format"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.current_dataset = df
                
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head(10))
                
                with col2:
                    st.subheader("Dataset Info")
                    st.write(f"**Shape:** {df.shape}")
                    st.write(f"**Columns:** {len(df.columns)}")
                    st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    st.write(f"**Numeric Columns:** {len(numeric_cols)}")
                    st.write(f"**Categorical Columns:** {len(categorical_cols)}")
                
                st.subheader("Data Preprocessing")
                target_column = st.selectbox("Select Target Column", df.columns.tolist())
                
                if st.button("🔄 Preprocess Data"):
                    preprocess_data(df, target_column)
                    
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
    
    with tab2:
        st.header("AutoML Training")
        
        if st.session_state.current_dataset is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Start AutoML Training", type="primary"):
                    start_automl_training(problem_type, optimization_goal, max_training_time, enable_ensemble, enable_feature_engineering)
            
            with col2:
                if st.button("🔄 Train Multiple Models"):
                    train_multiple_models(problem_type, optimization_goal)
            
            with col3:
                if st.button("📊 Compare Models"):
                    compare_trained_models()
            
            if st.session_state.trained_models:
                display_training_results()
        else:
            st.info("📝 Please upload a dataset first to start training.")
    
    with tab3:
        st.header("Hyperparameter Optimization")
        
        if st.session_state.trained_models:
            st.subheader("Optimize Best Model")
            best_model_name = list(st.session_state.trained_models.keys())[0]
            selected_model = st.selectbox("Select Model to Optimize", list(st.session_state.trained_models.keys()))
            
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Bayesian Optimization", "Genetic Algorithm", "Random Search", "Grid Search"]
            )
            
            n_trials = st.slider("Number of Trials", 10, 200, 50)
            
            if st.button("⚡ Optimize Hyperparameters"):
                optimize_hyperparameters(selected_model, optimization_method, n_trials)
            
            if st.session_state.optimization_results:
                display_optimization_results()
        else:
            st.info("🤖 Train models first to optimize hyperparameters.")
    
    with tab4:
        st.header("Model Explanation & Interpretability")
        
        if st.session_state.trained_models:
            selected_model = st.selectbox("Select Model to Explain", list(st.session_state.trained_models.keys()))
            
            explanation_method = st.selectbox(
                "Explanation Method",
                ["SHAP", "LIME", "Partial Dependence", "Feature Importance"]
            )
            
            if st.button("🔍 Generate Explanations"):
                generate_model_explanations(selected_model, explanation_method)
        else:
            st.info("🤖 Train models first to generate explanations.")
    
    with tab5:
        st.header("Model Deployment")
        
        if st.session_state.trained_models:
            selected_model = st.selectbox("Select Model for Deployment", list(st.session_state.trained_models.keys()))
            
            deployment_framework = st.selectbox(
                "Deployment Framework",
                ["Flask API", "FastAPI", "Docker Container", "AWS Lambda", "Google Cloud Function"]
            )
            
            if st.button("🚀 Generate Deployment Code"):
                generate_deployment_code(selected_model, deployment_framework)
        else:
            st.info("🤖 Train models first to generate deployment code.")

def preprocess_data(df, target_column):
    load_components()
    
    with st.spinner("🔄 Preprocessing data..."):
        try:
            processed_data = st.session_state.data_processor.preprocess_data(df, target_column)
            st.session_state.current_dataset = processed_data
            st.success("✅ Data preprocessing completed successfully!")
            
            st.subheader("Preprocessing Summary")
            st.write(f"**Original Shape:** {df.shape}")
            st.write(f"**Processed Shape:** {processed_data.shape}")
            st.write(f"**Missing Values Handled:** Yes")
            st.write(f"**Categorical Encoding:** Yes")
            st.write(f"**Feature Scaling:** Yes")
            
        except Exception as e:
            st.error(f"❌ Data preprocessing failed: {str(e)}")

def start_automl_training(problem_type, optimization_goal, max_training_time, enable_ensemble, enable_feature_engineering):
    load_components()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🚀 Starting AutoML training...")
        
        X = st.session_state.current_dataset.drop(columns=[st.session_state.data_processor.target_column])
        y = st.session_state.current_dataset[st.session_state.data_processor.target_column]
        
        trained_models = st.session_state.automl_engine.train_models(
            X=X,
            y=y,
            problem_type=problem_type,
            optimization_metric=optimization_goal,
            max_training_time=max_training_time * 60,
            enable_ensemble=enable_ensemble,
            enable_feature_engineering=enable_feature_engineering
        )
        
        st.session_state.trained_models = trained_models
        
        progress_bar.progress(100)
        status_text.text("✅ AutoML training completed successfully!")
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ AutoML training failed: {str(e)}")

def train_multiple_models(problem_type, optimization_goal):
    load_components()
    
    with st.spinner("🤖 Training multiple models..."):
        try:
            X = st.session_state.current_dataset.drop(columns=[st.session_state.data_processor.target_column])
            y = st.session_state.current_dataset[st.session_state.data_processor.target_column]
            
            models = st.session_state.automl_engine.train_multiple_models(
                X=X,
                y=y,
                problem_type=problem_type,
                optimization_metric=optimization_goal
            )
            
            st.session_state.trained_models.update(models)
            st.success(f"✅ Trained {len(models)} additional models!")
            
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")

def compare_trained_models():
    if not st.session_state.trained_models:
        st.warning("No trained models to compare.")
        return
    
    comparison_df = st.session_state.automl_engine.compare_models(st.session_state.trained_models)
    
    st.subheader("Model Comparison")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
    
    best_model = comparison_df.iloc[0]['Model']
    st.success(f"🏆 Best Model: {best_model}")

def display_training_results():
    st.header("Training Results")
    
    for model_name, model_info in st.session_state.trained_models.items():
        with st.expander(f"📊 {model_name} - Score: {model_info['score']:.4f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Details**")
                st.write(f"**Algorithm:** {model_info['algorithm']}")
                st.write(f"**Training Time:** {model_info['training_time']:.2f}s")
                st.write(f"**Cross-Validation Score:** {model_info['cv_score']:.4f}")
                
                if 'feature_importance' in model_info:
                    st.write("**Top Features:**")
                    for feature, importance in list(model_info['feature_importance'].items())[:5]:
                        st.write(f"  - {feature}: {importance:.4f}")
            
            with col2:
                if 'learning_curve' in model_info:
                    st.plotly_chart(model_info['learning_curve'], use_container_width=True)

def optimize_hyperparameters(selected_model, optimization_method, n_trials):
    with st.spinner("⚡ Optimizing hyperparameters..."):
        try:
            model_info = st.session_state.trained_models[selected_model]
            X = st.session_state.current_dataset.drop(columns=[st.session_state.data_processor.target_column])
            y = st.session_state.current_dataset[st.session_state.data_processor.target_column]
            
            optimizer = HyperparameterOptimizer()
            optimization_results = optimizer.optimize(
                model=model_info['model'],
                X=X,
                y=y,
                method=optimization_method,
                n_trials=n_trials
            )
            
            st.session_state.optimization_results[selected_model] = optimization_results
            st.success(f"✅ Hyperparameter optimization completed! Best score: {optimization_results['best_score']:.4f}")
            
        except Exception as e:
            st.error(f"❌ Hyperparameter optimization failed: {str(e)}")

def display_optimization_results():
    st.subheader("Optimization Results")
    
    for model_name, results in st.session_state.optimization_results.items():
        with st.expander(f"⚡ {model_name} Optimization"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best Parameters**")
                for param, value in results['best_params'].items():
                    st.write(f"**{param}:** {value}")
                
                st.write(f"**Best Score:** {results['best_score']:.4f}")
                st.write(f"**Improvement:** {results['improvement']:.4f}")
            
            with col2:
                if 'optimization_history' in results:
                    st.plotly_chart(results['optimization_history'], use_container_width=True)

def generate_model_explanations(selected_model, explanation_method):
    with st.spinner("🔍 Generating model explanations..."):
        try:
            model_info = st.session_state.trained_models[selected_model]
            X = st.session_state.current_dataset.drop(columns=[st.session_state.data_processor.target_column])
            
            explainer = ModelExplainer()
            explanations = explainer.explain_model(
                model=model_info['model'],
                X=X,
                method=explanation_method
            )
            
            st.subheader(f"Model Explanations - {explanation_method}")
            
            if explanation_method == "SHAP":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(explanations['summary_plot'], use_container_width=True)
                
                with col2:
                    st.plotly_chart(explanations['feature_importance'], use_container_width=True)
            
            elif explanation_method == "Feature Importance":
                st.plotly_chart(explanations['feature_importance'], use_container_width=True)
            
            st.success("✅ Model explanations generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Model explanation failed: {str(e)}")

def generate_deployment_code(selected_model, deployment_framework):
    with st.spinner("🚀 Generating deployment code..."):
        try:
            model_info = st.session_state.trained_models[selected_model]
            
            code_generator = CodeGenerator()
            deployment_code = code_generator.generate_deployment_code(
                model=model_info['model'],
                model_name=selected_model,
                framework=deployment_framework
            )
            
            st.subheader(f"Deployment Code - {deployment_framework}")
            
            st.code(deployment_code['code'], language='python')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Code",
                    data=deployment_code['code'],
                    file_name=f"{selected_model}_{deployment_framework.lower()}.py",
                    mime="text/python"
                )
            
            with col2:
                if 'dockerfile' in deployment_code:
                    st.download_button(
                        label="🐳 Download Dockerfile",
                        data=deployment_code['dockerfile'],
                        file_name="Dockerfile",
                        mime="text/plain"
                    )
            
            st.success("✅ Deployment code generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Deployment code generation failed: {str(e)}")

if __name__ == "__main__":
    main()