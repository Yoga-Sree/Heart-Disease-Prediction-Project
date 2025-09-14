# heart_disease_prediction.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.model_performance = {}

    def load_data(self, file_path=None):
        """Load heart disease dataset"""
        if file_path:
            self.data = pd.read_csv(file_name)
        else:
            # Create synthetic data if no file is provided
            self._create_synthetic_data()

        print("Dataset loaded successfully!")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst 5 rows:")
        print(self.data.head())

        # Separate features and target
        self.y = self.data['target']
        self.X = self.data.drop('target', axis=1)

    def _create_synthetic_data(self):
        """Create synthetic heart disease data for demonstration"""
        np.random.seed(42)
        n_samples = 1000

        # Generate synthetic features
        age = np.random.normal(55, 9, n_samples)
        sex = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.45, 0.25, 0.2, 0.1])
        trestbps = np.random.normal(130, 18, n_samples)
        chol = np.random.normal(240, 45, n_samples)
        fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        restecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1])
        thalach = np.random.normal(150, 23, n_samples)
        exang = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        oldpeak = np.abs(np.random.normal(1.0, 1.2, n_samples))
        slope = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.45, 0.15])
        ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.55, 0.3, 0.1, 0.05])
        thal = np.random.choice([0, 1, 2, 3], n_samples, p=[0.05, 0.7, 0.2, 0.05])

        # Create a target variable based on feature relationships
        z = (age - 55)/9 + (chol - 240)/45 - (thalach - 150)/23 + oldpeak
        prob = 1 / (1 + np.exp(-z))
        target = np.random.binomial(1, prob)

        # Create DataFrame
        self.data = pd.DataFrame({
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
            'thal': thal, 'target': target
        })

        # Ensure values are within reasonable ranges
        self.data['age'] = self.data['age'].clip(29, 80)
        self.data['trestbps'] = self.data['trestbps'].clip(90, 200)
        self.data['chol'] = self.data['chol'].clip(120, 570)
        self.data['thalach'] = self.data['thalach'].clip(70, 220)
        self.data['oldpeak'] = self.data['oldpeak'].clip(0, 6.2)

    def explore_data(self):
        """Explore the dataset with visualizations"""
        print("\n=== Dataset Exploration ===\n")

        # Basic info
        print("Dataset Info:")
        print(self.data.info())

        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())

        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())

        # Class distribution
        print("\nClass Distribution:")
        print(self.data['target'].value_counts())
        print(f"Positive Rate: {self.data['target'].mean():.2%}")

        # Visualizations
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Heart Disease Dataset Exploration', fontsize=16)

        # Target distribution
        self.data['target'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].set_xticklabels(['No Disease', 'Disease'], rotation=0)

        # Age distribution by target
        sns.histplot(data=self.data, x='age', hue='target', kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Target')

        # Cholesterol distribution by target
        sns.boxplot(data=self.data, x='target', y='chol', ax=axes[1, 0])
        axes[1, 0].set_title('Cholesterol by Target')
        axes[1, 0].set_xticklabels(['No Disease', 'Disease'])

        # Max heart rate by target
        sns.boxplot(data=self.data, x='target', y='thalach', ax=axes[1, 1])
        axes[1, 1].set_title('Max Heart Rate by Target')
        axes[1, 1].set_xticklabels(['No Disease', 'Disease'])

        # Correlation heatmap
        corr = self.data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=axes[2, 0])
        axes[2, 0].set_title('Correlation Heatmap')

        # Chest pain type by target
        cp_data = pd.crosstab(self.data['cp'], self.data['target'])
        cp_data.plot(kind='bar', ax=axes[2, 1])
        axes[2, 1].set_title('Chest Pain Type by Target')
        axes[2, 1].set_xticklabels(['Typical Angina', 'Atypical Angina',
                                   'Non-anginal Pain', 'Asymptomatic'], rotation=45)
        axes[2, 1].legend(['No Disease', 'Disease'])

        plt.tight_layout()
        plt.savefig('heart_disease_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("\nExploratory visualizations saved as 'heart_disease_exploration.png'")

    def preprocess_data(self):
        """Split and preprocess the data"""
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Define preprocessing for numeric and categorical features
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Apply preprocessing
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)

        print("\n=== Data Preprocessing ===\n")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")

    def train_models(self):
        """Train multiple classification models"""
        print("\n=== Training Models ===\n")

        # Define models to train with hyperparameter grids
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.05]}
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            }
        }

        # Train each model with GridSearchCV
        for name, model_info in models.items():
            print(f"Training {name}...")
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)

            # Store the best model
            self.models[name] = grid_search.best_estimator_

            # Cross-validation score
            cv_scores = cross_val_score(
                grid_search.best_estimator_,
                self.X_train,
                self.y_train,
                cv=5,
                scoring='accuracy'
            )

            # Store performance metrics
            self.model_performance[name] = {
                'best_params': grid_search.best_params_,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'test_accuracy': grid_search.best_estimator_.score(self.X_test, self.y_test)
            }

            print(f"  Best params: {grid_search.best_params_}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"  Test Accuracy: {self.model_performance[name]['test_accuracy']:.4f}")

        # Select the best model based on test accuracy
        self.best_model = max(
            self.models.items(),
            key=lambda x: self.model_performance[x[0]]['test_accuracy']
        )
        print(f"\nBest Model: {self.best_model[0]} with test accuracy: "
              f"{self.model_performance[self.best_model[0]]['test_accuracy']:.4f}")

        # Calculate feature importance for tree-based models
        if hasattr(self.best_model[1], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self._get_feature_names(),
                'importance': self.best_model[1].feature_importances_
            }).sort_values('importance', ascending=False)

    def _get_feature_names(self):
        """Get feature names after preprocessing"""
        # Numeric features
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        # Categorical features
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        categorical_transformer = self.preprocessor.named_transformers_['cat']
        cat_feature_names = categorical_transformer.named_steps['onehot'].get_feature_names_out(categorical_features)

        # Combine all feature names
        all_features = numeric_features + list(cat_feature_names)
        return all_features

    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n=== Model Evaluation ===\n")

        # Create evaluation report
        evaluation_df = pd.DataFrame.from_dict(self.model_performance, orient='index')
        evaluation_df = evaluation_df[['cv_mean_accuracy', 'cv_std_accuracy', 'test_accuracy']]
        evaluation_df.columns = ['CV Accuracy', 'CV Std', 'Test Accuracy']
        evaluation_df = evaluation_df.sort_values('Test Accuracy', ascending=False)

        print("Model Performance Comparison:")
        print(evaluation_df.round(4))

        # Plot model comparison
        plt.figure(figsize=(10, 6))
        evaluation_df['Test Accuracy'].plot(kind='bar', color='skyblue')
        plt.title('Model Comparison - Test Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Evaluate the best model in detail
        self.evaluate_model(self.best_model[0])

    def evaluate_model(self, model_name):
        """Evaluate a specific model in detail"""
        model = self.models[model_name]

        print(f"\n=== Detailed Evaluation for {model_name} ===\n")

        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred,
                                   target_names=['No Disease', 'Disease']))

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved as 'confusion_matrix_{model_name.lower().replace(' ', '_')}.png'")

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved as 'roc_curve_{model_name.lower().replace(' ', '_')}.png'")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f'pr_curve_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Precision-Recall curve saved as 'pr_curve_{model_name.lower().replace(' ', '_')}.png'")


# Instantiate the class and run the workflow
predictor = HeartDiseasePredictor()
predictor.load_data()
predictor.explore_data()
predictor.preprocess_data()
predictor.train_models()
predictor.evaluate_models()
