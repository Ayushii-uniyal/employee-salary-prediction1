import pandas as pd
import numpy as np
import joblib
import json
from lightgbm import LGBMRegressor

# Load artifacts
try:
    model = joblib.load('salary_predictor.pkl')
    model.set_params(predict_disable_shape_check=True)  # <- Add this line

    ct = joblib.load('column_transformer.pkl')
    le_exp = joblib.load('label_encoder_exp.pkl')
    le_size = joblib.load('label_encoder_size.pkl')
    
    with open('category_orders.json') as f:
        categories = json.load(f)
        VALID_EXPERIENCE = categories['experience_order']
        VALID_COMPANY_SIZES = categories['company_size_order']

except Exception as e:
    raise RuntimeError(f"Error loading artifacts: {e}")

# Job title mapping based on your dataset categories
JOB_CATEGORY_MAPPING = {
    # Data roles
    'data scientist': 'Data Science & ML',
    'data analyst': 'Data Analytics', 
    'machine learning engineer': 'Data Science & ML',
    'data engineer': 'Data Engineering',
    'business analyst': 'Business Intelligence',
    'business intelligence analyst': 'Business Intelligence',
    
    # Engineering roles  
    'software engineer': 'Data Engineering',
    'software developer': 'Data Engineering',
    'full stack developer': 'Data Engineering',
    'backend developer': 'Data Engineering',
    'frontend developer': 'Data Engineering',
    'devops engineer': 'DevOps & Cloud',
    'cloud engineer': 'DevOps & Cloud',
    
    # Business roles
    'sales manager': 'Associate Roles',
    'marketing manager': 'Associate Roles',
    'product manager': 'Associate Roles',
    'project manager': 'Associate Roles',
    'consultant': 'Consulting',
    
    # Healthcare/Other
    'doctor': 'Clinical & Healthcare',
    'nurse': 'Clinical & Healthcare',
    'pharmacist': 'Clinical & Healthcare',
}

# Valid options (update if needed)
VALID_EMPLOYMENT = ['FT', 'PT', 'CT', 'FL']  # Full-time, Part-time, etc.

def validate_input(input_data: dict) -> None:
    """Validate all input fields"""
    errors = []
    
    if input_data['experience_level'] not in VALID_EXPERIENCE:
        errors.append(f"Experience level must be one of {VALID_EXPERIENCE}")
    if input_data['company_size'] not in VALID_COMPANY_SIZES:
        errors.append(f"Company size must be one of {VALID_COMPANY_SIZES}")
    if input_data['employment_type'] not in VALID_EMPLOYMENT:
        errors.append(f"Employment type must be one of {VALID_EMPLOYMENT}")
    if not 0 <= input_data['remote_ratio'] <= 100:
        errors.append("Remote ratio must be 0-100")
    
    if errors:
        raise ValueError(" | ".join(errors))

def smart_job_title_mapping(job_title: str) -> str:
    """Maps job titles to known categories using fuzzy matching"""
    job_title_lower = job_title.lower().strip()
    
    # 1. Direct mapping
    if job_title_lower in JOB_CATEGORY_MAPPING:
        return JOB_CATEGORY_MAPPING[job_title_lower]
    
    # 2. Keyword matching
    if any(keyword in job_title_lower for keyword in ['data', 'analyst', 'analytics']):
        return 'Data Analytics'
    elif any(keyword in job_title_lower for keyword in ['scientist', 'ml', 'machine learning']):
        return 'Data Science & ML'  
    elif any(keyword in job_title_lower for keyword in ['engineer', 'developer', 'programming']):
        return 'Data Engineering'
    elif any(keyword in job_title_lower for keyword in ['sales', 'account', 'business development']):
        return 'Associate Roles'
    elif any(keyword in job_title_lower for keyword in ['marketing', 'growth', 'digital']):
        return 'Associate Roles'
    elif any(keyword in job_title_lower for keyword in ['manager', 'director', 'lead']):
        return 'Associate Roles'
    elif any(keyword in job_title_lower for keyword in ['consultant', 'advisor']):
        return 'Consulting'
    elif any(keyword in job_title_lower for keyword in ['cloud', 'devops', 'infrastructure']):
        return 'DevOps & Cloud'
    elif any(keyword in job_title_lower for keyword in ['doctor', 'physician', 'medical']):
        return 'Clinical & Healthcare'
    
    # 3. Default fallback
    return 'Data Analytics'

def prepare_features(input_data: dict) -> pd.DataFrame:
    """Convert raw input to DataFrame matching training structure"""
    # Create DataFrame with ALL original columns
    original_job = input_data['job_title']
    mapped_job = smart_job_title_mapping(original_job)

    input_data_mapped = input_data.copy()
    input_data_mapped['job_title'] = mapped_job

    input_df = pd.DataFrame(columns=[
        'experience_level', 
        'employment_type',
        'job_title',
        'remote_ratio',
        'company_location',
        'company_size'
    ])
    
    # Add current input (other fields will be NaN)
    input_df = pd.concat([input_df, pd.DataFrame([input_data_mapped])], ignore_index=True)
    
    # Ordinal encoding
    input_df['experience_level_encoded'] = le_exp.transform(
        pd.Categorical(input_df['experience_level'], categories=VALID_EXPERIENCE)
    )
    input_df['company_size_encoded'] = le_size.transform(
        pd.Categorical(input_df['company_size'], categories=VALID_COMPANY_SIZES)
    )
    
    return input_df.fillna(0)  # Fill missing with 0

def predict_salary(input_data: dict) -> dict:
    validate_input(input_data)
    input_df = prepare_features(input_data)
    
    # Get EXPECTED feature names from training
    expected_features = ct.get_feature_names_out()
    
    # Transform current input
    features = ct.transform(input_df)
    
    # Convert to DataFrame with CORRECT columns
    features_df = pd.DataFrame(
        features.toarray() if hasattr(features, "toarray") else features,
        columns=ct.get_feature_names_out()
    )
    
    # Add missing features (set to 0)
    for feat in expected_features:
        if feat not in features_df.columns:
            features_df[feat] = 0
    
    # Reorder columns to match training
    features_df = features_df[expected_features]
    
    # Predict
    log_pred = model.predict(features_df)[0]
    return {
        'prediction': round(np.exp(log_pred), 2),
        'range': [
            round(np.exp(log_pred - 0.2)), 
            round(np.exp(log_pred + 0.2))
        ],
        'mapped_job_title': smart_job_title_mapping(input_data['job_title'])

    }