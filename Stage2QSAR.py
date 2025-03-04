# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.decomposition import PCA
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import ast
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Function to calculate Q² (q-squared) for validation
def calculate_q_squared(y_true, y_pred, y_train_mean):
    """
    Calculate Q² (predictive r-squared) which accounts for overfitting

    Parameters:
    y_true: True target values
    y_pred: Predicted target values
    y_train_mean: Mean of the training target values

    Returns:
    q_squared: Q² value
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_train_mean) ** 2)

    q_squared = 1 - (numerator / denominator)
    return q_squared

# Load the Stage 1 model outputs (merged data with features)
print("Loading datasets...")
merged_data = pd.read_csv("/content/merged_data.csv")  # Modify path as needed

# --- Data Preparation ---
print("Preparing data for modeling...")

# 1. Identify target variable and features
TARGET = 'standard_value'
NON_FEATURES = ['cell_line_display_name', TARGET, 'molecule_chembl_id', 'canonical_smiles',
               'assay_chembl_id', 'assay_type', 'standard_type', 'standard_relation',
               'target_chembl_id', 'target_pref_name']
FEATURES = [col for col in merged_data.columns if col not in NON_FEATURES]

# 2. Separate features (X) and target (y)
X = merged_data[FEATURES].copy()
y = merged_data[TARGET].copy()

# Apply data transformation for skewed target variable (often helps with drug response data)
y_transformed = np.log1p(y)  # log(1+y) transformation for positively skewed data
print(f"Applied log1p transformation to target variable. Original range: {y.min()}-{y.max()}, Transformed range: {y_transformed.min()}-{y_transformed.max()}")

# Function to handle dictionary-like strings in case they exist
def safe_parse_dict(value):
    """Safely parse dictionary-like strings to numeric values"""
    if not isinstance(value, str):
        return value

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict) and len(parsed) > 0:
            return float(list(parsed.values())[0])
        return np.nan
    except:
        return np.nan

# Process non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Found {len(non_numeric_cols)} non-numeric columns")

# Convert dictionary-like strings to floats
for col in non_numeric_cols:
    X[col] = X[col].apply(safe_parse_dict)

# Verify all columns are now numeric
remaining_non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if remaining_non_numeric:
    print(f"WARNING: {len(remaining_non_numeric)} columns still non-numeric: {remaining_non_numeric}")
    # Drop columns that couldn't be converted
    X = X.drop(columns=remaining_non_numeric)
    print(f"Dropped non-numeric columns. New shape: {X.shape}")

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_transformed, test_size=0.2, random_state=42
)

# Store the training mean for Q² calculation
y_train_mean = y_train.mean()

# Check for data leakage between train and test sets
if 'cell_line_display_name' in merged_data.columns:
    train_cell_lines = set(merged_data.loc[X_train.index, 'cell_line_display_name'])
    test_cell_lines = set(merged_data.loc[X_test.index, 'cell_line_display_name'])
    overlap = train_cell_lines.intersection(test_cell_lines)
    print(f"Cell line overlap between train and test: {len(overlap)} out of {len(test_cell_lines)} test cell lines")

# --- Advanced Feature Selection ---
print("\nPerforming advanced feature selection...")

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.columns)
    ]
)

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# 1. PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train_preprocessed)
X_test_pca = pca.transform(X_test_preprocessed)
print(f"PCA reduced dimensions from {X_train_preprocessed.shape[1]} to {X_train_pca.shape[1]} features")

# 2. Recursive Feature Elimination with Cross-Validation
print("Performing Recursive Feature Elimination with Cross-Validation...")
rfecv = RFECV(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    step=10,  # Remove 10 features at a time
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit RFECV on a subset to speed up the process if dataset is large
X_train_rfecv = rfecv.fit_transform(X_train_preprocessed, y_train)
X_test_rfecv = rfecv.transform(X_test_preprocessed)
print(f"RFECV selected {X_train_rfecv.shape[1]} features")

# Plot RFECV results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of features selected')
plt.ylabel('Cross-validation score (neg MSE)')
plt.title('Recursive Feature Elimination with Cross-Validation')
plt.tight_layout()
plt.savefig('rfecv_results.png')
plt.close()

# --- Define Base Models ---
print("\nDefining base models...")

base_models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'SVR': SVR(),
    'NeuralNetwork': MLPRegressor(random_state=42, max_iter=1000),
    'ElasticNet': ElasticNet(random_state=42)
}

# Define parameter grids for each base model
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'num_leaves': [20, 40, 60],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    },
    'NeuralNetwork': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1]
    },
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1],
        'l1_ratio': [0.2, 0.5, 0.8]
    }
}

# Train base models and evaluate
trained_models = {}
for model_name, model in base_models.items():
    print(f"\nTraining {model_name} model...")
    param_grid = param_grids[model_name]
    
    # Create GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit the model
    grid_search.fit(X_train_rfecv, y_train)
    
    # Get the best estimator
    best_model = grid_search.best_estimator_
    trained_models[model_name] = best_model
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test_rfecv)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    q2 = calculate_q_squared(y_test, y_pred, y_train_mean)
    
    print(f"{model_name} Results:")
    print(f"R² = {r2:.4f}, Q² = {q2:.4f}")
    print(f"RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# --- Stacking Ensemble ---
print("\nCreating Stacking Ensemble...")

# Define base estimators
level0 = list(trained_models.items())  # List of (name, model) tuples

# Define meta learner
level1 = Ridge()  # Simple linear model

# Define the stacking ensemble
stacking_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)

# Train the stacking ensemble
stacking_model.fit(X_train_rfecv, y_train)

# Make predictions on the test set
y_pred_stacking = stacking_model.predict(X_test_rfecv)

# Evaluate performance
r2_stacking = r2_score(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mean_squared_error(y_test, y_pred_stacking))
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
q2_stacking = calculate_q_squared(y_test, y_pred_stacking, y_train_mean)

print("Stacking Ensemble Results:")
print(f"R² = {r2_stacking:.4f}, Q² = {q2_stacking:.4f}")
print(f"RMSE = {rmse_stacking:.4f}, MAE = {mae_stacking:.4f}")

# --- Model Evaluation and Comparison ---
print("\nModel Evaluation and Comparison...")
results = []
for model_name, model in trained_models.items():
    # Make predictions
    y_pred = model.predict(X_test_rfecv)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    q2 = calculate_q_squared(y_test, y_pred, y_train_mean)
    
    # Add results to the list
    results.append([model_name, r2, q2, rmse, mae])

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'R²', 'Q²', 'RMSE', 'MAE'])
results_df = results_df.sort_values(by='R²', ascending=False).reset_index(drop=True)
print("Base Model Comparison:")
print(results_df)

# Add Stacking to results
stacking_results = ['Stacking', r2_stacking, q2_stacking, rmse_stacking, mae_stacking]
results_df.loc[len(results_df)] = stacking_results
print("\nAll Results (including Stacking):")
print(results_df)

# --- Conversion to Binary Classification and Evaluation ---
print("\nConverting to Binary Classification and Evaluating...")

# Define threshold (median of the target variable) to create binary classes
threshold = y_transformed.median()

# Convert to binary classes
y_train_binary = (y_train > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

# Recalculate metrics for base models and stacking using binary outcomes
binary_results = []
for model_name, model in trained_models.items():
    # Make predictions
    y_pred = model.predict(X_test_rfecv)

    # Convert predictions to binary values using the same threshold
    y_pred_binary = (y_pred > threshold).astype(int)

    # Calculate binary classification metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)

    # Add binary classification metrics to results
    binary_results.append([model_name, accuracy, precision, recall, f1])

# Evaluate stacking model's binary predictions
y_pred_stacking_binary = (y_pred_stacking > threshold).astype(int)

# Calculate binary classification metrics for stacking model
stacking_accuracy = accuracy_score(y_test_binary, y_pred_stacking_binary)
stacking_precision = precision_score(y_test_binary, y_pred_stacking_binary)
stacking_recall = recall_score(y_test_binary, y_pred_stacking_binary)
stacking_f1 = f1_score(y_test_binary, y_pred_stacking_binary)

# Add stacking results to the table
binary_results.append(['Stacking', stacking_accuracy, stacking_precision, stacking_recall, stacking_f1])

# Create DataFrame for comparison
binary_df = pd.DataFrame(binary_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
binary_df = binary_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print("\nBinary Classification Results:")
print(binary_df)

# --- Identify Novel Inhibitors ---
print("\nIdentifying potential novel inhibitors...")

# 1. Load a database of potential inhibitors.
# For example, ZINC database, ChEMBL, etc.
# Load example dataset (replace with your actual data)
new_compounds = pd.DataFrame({
    'compound_id': ['CMP1', 'CMP2', 'CMP3'],
    'feature1': [0.1, 0.2, 0.3],
    'feature2': [0.4, 0.5, 0.6]
})
print("Loading and preprocessing new compound data...")

# 2. Preprocess new compound data in the same way as training data
new_compounds_preprocessed = preprocessor.transform(new_compounds.drop('compound_id', axis=1, errors='ignore'))

# Apply feature selection using RFECV to the new compounds
new_compounds_selected = rfecv.transform(new_compounds_preprocessed)

# 3. Predict activity using the stacking model
predicted_activities = stacking_model.predict(new_compounds_selected)

# 4. Apply activity threshold (choose an appropriate threshold)
activity_threshold = np.log1p(1000)  # Example: Predicted activity > 1000

# 5. Identify potential novel inhibitors
potential_inhibitors = new_compounds[predicted_activities > activity_threshold]

# Print Results
print("\nPotential Novel Inhibitors:")
print(potential_inhibitors)

# --- Save Everything ---
print("\nSaving models, predictions, and results...")
# Save Models
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(rfecv, 'rfecv.pkl')
joblib.dump(stacking_model, 'stacking_model.pkl')

# Save Predictions and Results (example)
np.save('y_pred_stacking.npy', y_pred_stacking)
results_df.to_csv('model_comparison.csv', index=False)
binary_df.to_csv('binary_classification_results.csv', index=False)

print("All done!")
