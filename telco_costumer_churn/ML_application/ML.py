import pandas as pd
import os
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# Defining File paths
file_path = r'file_path\pre_processed.xlsx'
output_dir = r'output_path\ML_application'

# 1: Load dataset
def load_dataset(file_path):
    print('[INFO] Loading dataset...')
    try:
        df = pd.read_excel(file_path)
        print(f'[SUCCESS] File successfully loaded from: {file_path}')
        return df
    except Exception as e:
        print(f'[ERROR] Failed to load file: {e}')
        return None


# 2: Preprocess dataset
def preprocess_data(df):
    print('[INFO] Starting data preprocessing...')
    try:
        X = df.drop(columns=['Churn'])
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ]
        )

        print('[SUCCESS] Preprocessing completed.')
        return X, y, preprocessor
    except Exception as e:
        print(f'[ERROR] Data preprocessing failed: {e}')
        return None, None, None


# 3: Split data
def split_data(X, y, test_size=0.3, random_state=42):
    print('[INFO] Splitting data into training and testing sets...')
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f'[SUCCESS] Data split completed: {len(X_train)} training samples, {len(X_test)} testing samples.')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f'[ERROR] Failed to split data: {e}')
        return None, None, None, None


# 4: Train SVM model
def train_svm(X_train, y_train, preprocessor):
    print('[INFO] Training SVM model...')
    try:
        svm_params = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }

        svm_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=51))
        ])

        svm_grid = GridSearchCV(
            estimator=svm_pipeline,
            param_grid=svm_params,
            scoring='accuracy',
            cv=5,
            n_jobs=-1
        )

        start_time = time.time()
        svm_grid.fit(X_train, y_train)
        training_time = time.time() - start_time

        print(f'[SUCCESS] Training completed in {training_time:.2f} seconds.')
        return svm_grid, training_time
    except Exception as e:
        print(f'[ERROR] Failed to train SVM model: {e}')
        return None, None


# 5:Evaluate model
def evaluate_model(model, X_test, y_test):
    print('[INFO] Evaluating the SVM model...')
    try:
        y_pred = model.predict(X_test)

        metrics = {
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }

        print('[SUCCESS] Model evaluation completed.')
        return y_pred, metrics
    except Exception as e:
        print(f'[ERROR] Failed to evaluate model: {e}')
        return None, None


# 6:Save outputs
def save_outputs(X_test, y_pred, metrics, output_dir):
    print('[INFO] Saving outputs...')
    try:
        # Save predictions
        X_test_with_predictions = X_test.copy()
        X_test_with_predictions['Predicted Labels'] = y_pred
        predictions_file = os.path.join(output_dir, 'y-pred.xlsx')
        X_test_with_predictions.to_excel(predictions_file, index=False)
        print(f'[SUCCESS] Predictions saved at: {predictions_file}')

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_file = os.path.join(output_dir, 'metrics.xlsx')
        metrics_df.to_excel(metrics_file, index=False)
        print(f'[SUCCESS] Metrics saved at: {metrics_file}')
    except Exception as e:
        print(f'[ERROR] Failed to save outputs: {e}')


# 7: Main function
def main(file_path, output_dir):
    print('[INFO] Starting the ML pipeline...')
    df = load_dataset(file_path)
    if df is None:
        return

    X, y, preprocessor = preprocess_data(df)
    if X is None or y is None or preprocessor is None:
        return

    X_train, X_test, y_train, y_test = split_data(X, y)
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return

    model, training_time = train_svm(X_train, y_train, preprocessor)
    if model is None:
        return

    y_pred, metrics = evaluate_model(model, X_test, y_test)
    if y_pred is None or metrics is None:
        return

    metrics['Model'] = 'SVM'
    metrics['Best Parameters'] = model.best_params_
    metrics['Training Time (s)'] = training_time

    save_outputs(X_test, y_pred, metrics, output_dir)
    print('[INFO] ML pipeline execution completed successfully.')


# Applying the function
main(file_path, output_dir)
