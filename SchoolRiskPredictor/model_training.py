import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available, skipping LightGBM model")
from data_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns

class DropoutRiskModel:
    """
    Model training module for dropout risk prediction
    Supports XGBoost, RandomForest, and LightGBM models
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize model
        
        Parameters:
        - model_type: 'xgboost', 'randomforest', or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.class_names = ['Low', 'Medium', 'High']
        
    def build_model(self):
        """Build the selected model"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM is not available")
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Built {self.model_type} model")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = X_train.columns.tolist()
        
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {len(self.feature_names)}")
        
        if self.model_type == 'xgboost' and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("Training complete!")
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        Returns metrics dictionary
        """
        y_pred, y_proba = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print(f"\n{'='*50}")
        print(f"Model Evaluation: {self.model_type.upper()}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        }
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance scores
        """
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'randomforest':
            importance = self.model.feature_importances_
        elif self.model_type == 'lightgbm':
            importance = self.model.feature_importances_
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def plot_confusion_matrix(self, cm, save_path='models/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, top_n=15, save_path='models/feature_importance.png'):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importances - {self.model_type.upper()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {save_path}")
    
    def save_model(self, filepath='models/dropout_model.pkl'):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/dropout_model.pkl'):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_names = data['feature_names']
        self.class_names = data['class_names']
        print(f"Model loaded from {filepath}")


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all available models and compare performance
    """
    results = {}
    
    model_types = ['xgboost', 'randomforest']
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    
    for model_type in model_types:
        print(f"\n{'#'*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'#'*60}")
        
        model = DropoutRiskModel(model_type=model_type)
        model.build_model()
        model.train(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        model.plot_confusion_matrix(metrics['confusion_matrix'], 
                                    save_path=f'models/confusion_matrix_{model_type}.png')
        model.plot_feature_importance(top_n=15, 
                                     save_path=f'models/feature_importance_{model_type}.png')
        
        model.save_model(f'models/dropout_model_{model_type}.pkl')
        
        results[model_type] = {
            'model': model,
            'metrics': metrics
        }
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    comparison_df = pd.DataFrame({
        model_type: {
            'Accuracy': results[model_type]['metrics']['accuracy'],
            'Precision': results[model_type]['metrics']['precision'],
            'Recall': results[model_type]['metrics']['recall'],
            'F1-Score': results[model_type]['metrics']['f1_score']
        }
        for model_type in results.keys()
    }).T
    
    print(comparison_df)
    comparison_df.to_csv('models/model_comparison.csv')
    
    best_model_type = comparison_df['F1-Score'].idxmax()
    print(f"\nBest model: {best_model_type.upper()} (F1-Score: {comparison_df.loc[best_model_type, 'F1-Score']:.4f})")
    
    results[best_model_type]['model'].save_model('models/dropout_model_best.pkl')
    
    return results, best_model_type


if __name__ == '__main__':
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/udise_data.csv')
    
    X_train, X_test, y_train, y_test, features, train_df, test_df = preprocessor.prepare_train_test_split(df)
    
    preprocessor.save_preprocessor()
    
    results, best_model = train_all_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: {best_model}")
    print("Models saved in 'models/' directory")
