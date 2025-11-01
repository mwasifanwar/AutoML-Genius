# core/code_generator.py
import inspect
import json
from typing import Dict, Any, List
import sklearn
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class CodeGenerator:
    def __init__(self):
        self.template_registry = {
            'Flask API': self._generate_flask_code,
            'FastAPI': self._generate_fastapi_code,
            'Docker Container': self._generate_docker_code,
            'AWS Lambda': self._generate_aws_lambda_code,
            'Google Cloud Function': self._generate_gcp_function_code
        }
    
    def generate_deployment_code(self, model, model_name: str, framework: str, 
                               dependencies: List[str] = None) -> Dict[str, Any]:
        
        if framework not in self.template_registry:
            raise ValueError(f"Unsupported framework: {framework}")
        
        template_func = self.template_registry[framework]
        deployment_code = template_func(model, model_name, dependencies)
        
        return deployment_code
    
    def _generate_flask_code(self, model, model_name: str, dependencies: List[str]) -> Dict[str, Any]:
        model_code = self._serialize_model(model, model_name)
        
        flask_code = f'''
import flask
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler

app = flask.Flask(__name__)

# Load the trained model
{model_code}

# Feature names (update with your actual feature names)
FEATURE_NAMES = {self._get_feature_names(model)}

@app.route('/')
def home():
    return "AutoML Genius Model Deployment - {model_name}"

@app.route('/health')
def health():
    return {{"status": "healthy", "model": "{model_name}"}}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = flask.request.get_json()
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure all features are present
        for feature in FEATURE_NAMES:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Reorder columns to match training
        input_data = input_data[FEATURE_NAMES]
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = getattr(model, 'predict_proba', lambda x: None)(input_data)
        
        # Prepare response
        response = {{
            'prediction': prediction[0],
            'prediction_probability': prediction_proba[0].tolist() if prediction_proba is not None else None,
            'model': '{model_name}',
            'status': 'success'
        }}
        
        return flask.jsonify(response)
    
    except Exception as e:
        return flask.jsonify({{'error': str(e), 'status': 'error'}}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = flask.request.get_json()
        input_data = pd.DataFrame(data)
        
        # Ensure all features are present
        for feature in FEATURE_NAMES:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        input_data = input_data[FEATURE_NAMES]
        
        predictions = model.predict(input_data)
        prediction_probas = getattr(model, 'predict_proba', lambda x: None)(input_data)
        
        response = {{
            'predictions': predictions.tolist(),
            'prediction_probabilities': prediction_probas.tolist() if prediction_probas is not None else None,
            'model': '{model_name}',
            'status': 'success'
        }}
        
        return flask.jsonify(response)
    
    except Exception as e:
        return flask.jsonify({{'error': str(e), 'status': 'error'}}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
        
        requirements = '''
flask==2.3.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
'''

        dockerfile = self._generate_basic_dockerfile('Flask')
        
        return {
            'code': flask_code.strip(),
            'requirements': requirements.strip(),
            'dockerfile': dockerfile,
            'endpoints': ['/', '/health', '/predict', '/batch_predict'],
            'port': 5000
        }
    
    def _generate_fastapi_code(self, model, model_name: str, dependencies: List[str]) -> Dict[str, Any]:
        model_code = self._serialize_model(model, model_name)
        
        fastapi_code = f'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from typing import List, Optional

app = FastAPI(title="AutoML Genius API", description="{model_name} Model Deployment")

# Load the trained model
{model_code}

# Feature names (update with your actual feature names)
FEATURE_NAMES = {self._get_feature_names(model)}

class PredictionInput(BaseModel):
    features: dict

class BatchPredictionInput(BaseModel):
    features_list: List[dict]

class PredictionResponse(BaseModel):
    prediction: float
    prediction_probability: Optional[List[float]]
    model: str
    status: str

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    prediction_probabilities: Optional[List[List[float]]]
    model: str
    status: str

@app.get("/")
async def root():
    return {{"message": "AutoML Genius Model Deployment - {model_name}"}}

@app.get("/health")
async def health():
    return {{"status": "healthy", "model": "{model_name}"}}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data.features])
        
        # Ensure all features are present
        for feature in FEATURE_NAMES:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[FEATURE_NAMES]
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = getattr(model, 'predict_proba', lambda x: None)(input_df)
        
        return PredictionResponse(
            prediction=float(prediction[0]),
            prediction_probability=prediction_proba[0].tolist() if prediction_proba is not None else None,
            model="{model_name}",
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(input_data: BatchPredictionInput):
    try:
        input_df = pd.DataFrame(input_data.features_list)
        
        # Ensure all features are present
        for feature in FEATURE_NAMES:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[FEATURE_NAMES]
        
        predictions = model.predict(input_df)
        prediction_probas = getattr(model, 'predict_proba', lambda x: None)(input_df)
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            prediction_probabilities=prediction_probas.tolist() if prediction_probas is not None else None,
            model="{model_name}",
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        requirements = '''
fastapi==0.100.0
uvicorn==0.23.0
pydantic==2.0.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
'''

        dockerfile = self._generate_basic_dockerfile('FastAPI')
        
        return {
            'code': fastapi_code.strip(),
            'requirements': requirements.strip(),
            'dockerfile': dockerfile,
            'endpoints': ['/', '/health', '/predict', '/batch_predict', '/docs'],
            'port': 8000
        }
    
    def _generate_docker_code(self, model, model_name: str, dependencies: List[str]) -> Dict[str, Any]:
        dockerfile = self._generate_comprehensive_dockerfile(model_name)
        
        return {
            'dockerfile': dockerfile,
            'docker_compose': self._generate_docker_compose(),
            'deployment_guide': self._generate_docker_deployment_guide()
        }
    
    def _generate_aws_lambda_code(self, model, model_name: str, dependencies: List[str]) -> Dict[str, Any]:
        model_code = self._serialize_model(model, model_name)
        
        lambda_code = f'''
import json
import numpy as np
import pandas as pd
import pickle
import boto3

# Load the trained model
{model_code}

# Feature names (update with your actual feature names)
FEATURE_NAMES = {self._get_feature_names(model)}

def lambda_handler(event, context):
    try:
        # Parse input data
        if 'body' in event:
            if event['body']:
                data = json.loads(event['body'])
            else:
                return {{
                    'statusCode': 400,
                    'body': json.dumps({{'error': 'Empty body'}})
                }}
        else:
            data = event
        
        # Check if it's batch prediction
        if 'features_list' in data:
            # Batch prediction
            input_df = pd.DataFrame(data['features_list'])
            for feature in FEATURE_NAMES:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[FEATURE_NAMES]
            
            predictions = model.predict(input_df)
            prediction_probas = getattr(model, 'predict_proba', lambda x: None)(input_df)
            
            response = {{
                'predictions': predictions.tolist(),
                'prediction_probabilities': prediction_probas.tolist() if prediction_probas is not None else None,
                'model': '{model_name}',
                'status': 'success'
            }}
        else:
            # Single prediction
            input_df = pd.DataFrame([data['features']])
            for feature in FEATURE_NAMES:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[FEATURE_NAMES]
            
            prediction = model.predict(input_df)
            prediction_proba = getattr(model, 'predict_proba', lambda x: None)(input_df)
            
            response = {{
                'prediction': float(prediction[0]),
                'prediction_probability': prediction_proba[0].tolist() if prediction_proba is not None else None,
                'model': '{model_name}',
                'status': 'success'
            }}
        
        return {{
            'statusCode': 200,
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }},
            'body': json.dumps(response)
        }}
    
    except Exception as e:
        return {{
            'statusCode': 400,
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }},
            'body': json.dumps({{'error': str(e), 'status': 'error'}})
        }}
'''
        
        requirements = '''
boto3==1.28.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
scipy==1.10.0
joblib==1.3.0
'''
        
        return {
            'code': lambda_code.strip(),
            'requirements': requirements.strip(),
            'handler': 'lambda_function.lambda_handler',
            'runtime': 'python3.9',
            'timeout': 30,
            'memory_size': 512
        }
    
    def _generate_gcp_function_code(self, model, model_name: str, dependencies: List[str]) -> Dict[str, Any]:
        model_code = self._serialize_model(model, model_name)
        
        gcp_code = f'''
import json
import numpy as np
import pandas as pd
import pickle

# Load the trained model
{model_code}

# Feature names (update with your actual feature names)
FEATURE_NAMES = {self._get_feature_names(model)}

def predict(request):
    """HTTP Cloud Function for model prediction"""
    
    # Set CORS headers
    headers = {{
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }}
    
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    try:
        # Parse request data
        request_json = request.get_json(silent=True)
        
        if not request_json:
            return (json.dumps({{'error': 'No JSON data provided'}}), 400, headers)
        
        # Check if it's batch prediction
        if 'features_list' in request_json:
            # Batch prediction
            input_df = pd.DataFrame(request_json['features_list'])
            for feature in FEATURE_NAMES:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[FEATURE_NAMES]
            
            predictions = model.predict(input_df)
            prediction_probas = getattr(model, 'predict_proba', lambda x: None)(input_df)
            
            response = {{
                'predictions': predictions.tolist(),
                'prediction_probabilities': prediction_probas.tolist() if prediction_probas is not None else None,
                'model': '{model_name}',
                'status': 'success'
            }}
        else:
            # Single prediction
            input_df = pd.DataFrame([request_json['features']])
            for feature in FEATURE_NAMES:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[FEATURE_NAMES]
            
            prediction = model.predict(input_df)
            prediction_proba = getattr(model, 'predict_proba', lambda x: None)(input_df)
            
            response = {{
                'prediction': float(prediction[0]),
                'prediction_probability': prediction_proba[0].tolist() if prediction_proba is not None else None,
                'model': '{model_name}',
                'status': 'success'
            }}
        
        return (json.dumps(response), 200, headers)
    
    except Exception as e:
        return (json.dumps({{'error': str(e), 'status': 'error'}}), 400, headers)
'''
        
        requirements = '''
functions-framework==3.0.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
'''
        
        return {
            'code': gcp_code.strip(),
            'requirements': requirements.strip(),
            'entry_point': 'predict',
            'runtime': 'python39',
            'trigger': 'http'
        }
    
    def _serialize_model(self, model, model_name: str) -> str:
        model_type = type(model).__name__
        
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor']:
            return f'''
import pickle
model = pickle.loads({pickle.dumps(model)})
'''
        
        elif model_type in ['XGBClassifier', 'XGBRegressor']:
            return f'''
import xgboost as xgb
model = xgb.{model_type}()
model.load_model("{model_name}_model.json")
'''
        
        elif model_type in ['LGBMClassifier', 'LGBMRegressor']:
            return f'''
import lightgbm as lgb
model = lgb.Booster(model_file='{model_name}_model.txt')
'''
        
        else:
            return f'''
import pickle
model = pickle.loads({pickle.dumps(model)})
'''
    
    def _get_feature_names(self, model) -> List[str]:
        try:
            if hasattr(model, 'feature_names_in_'):
                return model.feature_names_in_.tolist()
            elif hasattr(model, 'feature_name_'):
                return model.feature_name_
            else:
                return ['feature_1', 'feature_2', 'feature_3']
        except:
            return ['feature_1', 'feature_2', 'feature_3']
    
    def _generate_basic_dockerfile(self, framework: str) -> str:
        return f'''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
'''
    
    def _generate_comprehensive_dockerfile(self, model_name: str) -> str:
        return f'''
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY {model_name}_model.* ./

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app.py"]
'''
    
    def _generate_docker_compose(self) -> str:
        return '''
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ml-api
'''
    
    def _generate_docker_deployment_guide(self) -> str:
        return '''
# Docker Deployment Guide

## 1. Build the Docker image
docker build -t automl-genius-api .

## 2. Run the container
docker run -p 5000:5000 automl-genius-api

## 3. Test the API
curl http://localhost:5000/health

## 4. For production with Docker Compose
docker-compose up -d
'''

class AdvancedCodeGenerator(CodeGenerator):
    def __init__(self):
        super().__init__()
        self.template_registry.update({
            'Kubernetes': self._generate_kubernetes_code,
            'AWS SageMaker': self._generate_sagemaker_code
        })
    
    def _generate_kubernetes_code(self, model, model_name: str, dependencies: List[str]) -> Dict[str, Any]:
        deployment_yaml = f'''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {model_name}
  template:
    metadata:
      labels:
        app: {model_name}
    spec:
      containers:
      - name: {model_name}
        image: {model_name}:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-service
spec:
  selector:
    app: {model_name}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
'''
        
        return {
            'deployment_yaml': deployment_yaml.strip(),
            'service_yaml': '',
            'config_map': '',
            'deployment_guide': self._generate_kubernetes_guide(model_name)
        }
    
    def _generate_sagemaker_code(self, model, model_name: str, dependencies: List[str]) -> Dict[str, Any]:
        sagemaker_code = f'''
import boto3
import sagemaker
from sagemaker import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

class {model_name}Predictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super().__init__(
            endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )

# Initialize SageMaker session
session = sagemaker.Session()
role = 'arn:aws:iam::ACCOUNT-NUMBER:role/SageMakerRole'

# Create model
model = Model(
    model_data='s3://your-bucket/{model_name}/model.tar.gz',
    image_uri='YOUR-ECR-IMAGE-URI',
    role=role,
    name='{model_name}',
    sagemaker_session=session
)

# Deploy model
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='{model_name}-endpoint'
)

print(f"Endpoint deployed: {{predictor.endpoint_name}}")
'''
        
        return {
            'sagemaker_code': sagemaker_code.strip(),
            'inference_script': self._generate_sagemaker_inference_script(model, model_name),
            'requirements': 'sagemaker>=2.0.0',
            'deployment_guide': self._generate_sagemaker_guide()
        }
    
    def _generate_sagemaker_inference_script(self, model, model_name: str) -> str:
        return f'''
import json
import numpy as np
import pandas as pd
import pickle
import os

def model_fn(model_dir):
    """Load the model from the model_dir"""
    with open(os.path.join(model_dir, '{model_name}_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {{request_content_type}}")

def predict_fn(input_data, model):
    """Make predictions"""
    if 'features_list' in input_data:
        # Batch prediction
        input_df = pd.DataFrame(input_data['features_list'])
        predictions = model.predict(input_df)
        prediction_probas = getattr(model, 'predict_proba', lambda x: None)(input_df)
        
        return {{
            'predictions': predictions.tolist(),
            'prediction_probabilities': prediction_probas.tolist() if prediction_probas is not None else None
        }}
    else:
        # Single prediction
        input_df = pd.DataFrame([input_data['features']])
        prediction = model.predict(input_df)
        prediction_proba = getattr(model, 'predict_proba', lambda x: None)(input_df)
        
        return {{
            'prediction': float(prediction[0]),
            'prediction_probability': prediction_proba[0].tolist() if prediction_proba is not None else None
        }}

def output_fn(prediction, content_type):
    """Format the prediction output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {{content_type}}")
'''
    
    def _generate_kubernetes_guide(self, model_name: str) -> str:
        return f'''
# Kubernetes Deployment Guide for {model_name}

## 1. Build and push Docker image
docker build -t your-registry/{model_name}:latest .
docker push your-registry/{model_name}:latest

## 2. Apply Kubernetes manifests
kubectl apply -f deployment.yaml

## 3. Verify deployment
kubectl get pods -l app={model_name}
kubectl get service {model_name}-service

## 4. Access the service
kubectl port-forward service/{model_name}-service 8080:80
'''
    
    def _generate_sagemaker_guide(self) -> str:
        return '''
# AWS SageMaker Deployment Guide

## 1. Package the model
tar -czf model.tar.gz model.pkl inference_script.py

## 2. Upload to S3
aws s3 cp model.tar.gz s3://your-bucket/models/

## 3. Create and deploy model
python sagemaker_deploy.py

## 4. Test the endpoint
aws sagemaker-runtime invoke-endpoint \\
    --endpoint-name your-endpoint \\
    --body '{"features": {...}}' \\
    --content-type application/json \\
    output.json
'''