import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    """
    Data preprocessing module for UDISE+ dataset
    Handles feature engineering, encoding, and train-test splitting
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'dropout_risk'
        
    def load_data(self, filepath='data/udise_data.csv'):
        """Load UDISE+ dataset"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def engineer_features(self, df):
        """
        Create additional features from existing data
        """
        df = df.copy()
        
        df['enrollment_trend'] = df.groupby('school_id')['total_enrollment'].pct_change().fillna(0)
        
        df['teacher_adequacy'] = df['pupil_teacher_ratio'].apply(
            lambda x: 1 if x <= 30 else (0.5 if x <= 40 else 0)
        )
        
        df['girls_ratio'] = df['enrollment_girls'] / (df['total_enrollment'] + 1)
        
        df['classroom_adequacy'] = df['total_enrollment'] / (df['num_classrooms'] * 40 + 1)
        
        df['basic_infrastructure'] = (
            df['has_electricity'] + 
            df['has_drinking_water'] + 
            df['has_toilet_boys'] + 
            df['has_toilet_girls']
        ) / 4.0
        
        df['advanced_infrastructure'] = (
            df['has_library'] + 
            df['has_computer_lab'] + 
            df['has_playground'] + 
            df['has_boundary_wall']
        ) / 4.0
        
        df['gender_gap'] = abs(df['enrollment_boys'] - df['enrollment_girls']) / (df['total_enrollment'] + 1)
        
        df['infrastructure_deficit'] = 1 - df['infrastructure_score']
        
        df['high_ptr_flag'] = (df['pupil_teacher_ratio'] > 35).astype(int)
        
        df['low_enrollment_flag'] = (df['total_enrollment'] < 100).astype(int)
        
        return df
    
    def prepare_features(self, df, fit=True):
        """
        Prepare features for modeling
        - Encode categorical variables
        - Scale numerical features
        - Select relevant features
        """
        df = df.copy()
        
        categorical_features = ['state', 'district', 'school_type', 'location', 'management']
        
        for col in categorical_features:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        numerical_features = [
            'year',
            'enrollment_boys',
            'enrollment_girls',
            'total_enrollment',
            'num_teachers',
            'pupil_teacher_ratio',
            'num_classrooms',
            'infrastructure_score',
            'gender_parity_index',
            'enrollment_trend',
            'teacher_adequacy',
            'girls_ratio',
            'classroom_adequacy',
            'basic_infrastructure',
            'advanced_infrastructure',
            'gender_gap',
            'infrastructure_deficit'
        ]
        
        binary_features = [
            'has_electricity',
            'has_drinking_water',
            'has_toilet_boys',
            'has_toilet_girls',
            'has_library',
            'has_computer_lab',
            'has_playground',
            'has_boundary_wall',
            'high_ptr_flag',
            'low_enrollment_flag'
        ]
        
        encoded_categorical = [f'{col}_encoded' for col in categorical_features if col in df.columns]
        
        self.feature_columns = numerical_features + binary_features + encoded_categorical
        
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if fit:
            df[available_features] = self.scaler.fit_transform(df[available_features].fillna(0))
        else:
            df[available_features] = self.scaler.transform(df[available_features].fillna(0))
        
        return df, available_features
    
    def encode_target(self, df, target_col='dropout_risk'):
        """
        Encode target variable (dropout risk)
        High = 2, Medium = 1, Low = 0
        """
        risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df['dropout_risk_encoded'] = df[target_col].map(risk_mapping)
        return df
    
    def prepare_train_test_split(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        Use most recent year for testing to simulate real-world prediction
        """
        df = df.copy()
        
        df = self.engineer_features(df)
        
        df = self.encode_target(df)
        
        df, feature_cols = self.prepare_features(df, fit=True)
        
        latest_year = df['year'].max()
        train_df = df[df['year'] < latest_year]
        test_df = df[df['year'] == latest_year]
        
        if len(test_df) < 50:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['dropout_risk_encoded'])
        
        X_train = train_df[feature_cols]
        y_train = train_df['dropout_risk_encoded']
        X_test = test_df[feature_cols]
        y_test = test_df['dropout_risk_encoded']
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, feature_cols, train_df, test_df
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor for later use"""
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load saved preprocessor"""
        data = joblib.load(filepath)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"Preprocessor loaded from {filepath}")


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    
    df = preprocessor.load_data('data/udise_data.csv')
    
    print("\nOriginal data shape:", df.shape)
    print("\nDropout risk distribution:")
    print(df['dropout_risk'].value_counts())
    
    X_train, X_test, y_train, y_test, features, train_df, test_df = preprocessor.prepare_train_test_split(df)
    
    preprocessor.save_preprocessor()
    
    print("\nPreprocessing complete!")
