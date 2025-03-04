# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform
import ast
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Load datasets
combined_features = pd.read_csv("/content/combined_selected_features.csv")
molecular_descriptors = pd.read_csv("/content/molecular_descriptors_with_activity.csv")

# Print column names for verification
print("Columns in combined_features:", combined_features.columns.tolist())
print("Columns in molecular_descriptors:", molecular_descriptors.columns.tolist())

# Rename the column in molecular_descriptors to match combined_features
if "cell_line" in molecular_descriptors.columns:
    molecular_descriptors.rename(columns={"cell_line": "cell_line_display_name"}, inplace=True)
else:
    raise KeyError("'cell_line' column not found in molecular_descriptors.")

# Merge datasets on the standardized column
merged_data = pd.merge(combined_features, molecular_descriptors, on="cell_line_display_name", how="inner")

# Check the merged dataset
print(f"Merged Data Shape: {merged_data.shape}")
print(f"Number of unique cell lines: {merged_data['cell_line_display_name'].nunique()}")

# --- Data Preparation ---

# 1. Identify target variable and features
TARGET = 'standard_value'
NON_FEATURES = ['cell_line_display_name', TARGET, 'molecule_chembl_id', 'canonical_smiles', 
               'assay_chembl_id', 'assay_type', 'standard_type', 'standard_relation', 
               'target_chembl_id', 'target_pref_name']
FEATURES = [col for col in merged_data.columns if col not in NON_FEATURES]

# 2. Separate features (X) and target (y)
X = merged_data[FEATURES].copy()  # Use copy to avoid SettingWithCopyWarning
y = merged_data[TARGET].copy()

# Function to handle dictionary-like strings
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

# 3. Split into training and test sets (with stratification if possible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check for data leakage between train and test sets
train_cell_lines = set(merged_data.loc[X_train.index, 'cell_line_display_name'])
test_cell_lines = set(merged_data.loc[X_test.index, 'cell_line_display_name'])
overlap = train_cell_lines.intersection(test_cell_lines)
print(f"Cell line overlap between train and test: {len(overlap)} out of {len(test_cell_lines)} test cell lines")

# --- Preprocessing Pipeline ---

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

# --- Model Building ---

# 1. Random Forest with Hyperparameter Tuning
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))),
    ('regressor', RandomForestRegressor(random_state=42))
])

rf_param_dist = {
    'regressor__n_estimators': randint(100, 300),
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': randint(2, 10),
    'regressor__min_samples_leaf': randint(1, 10),
    'regressor__max_features': ['sqrt', 'log2', None]
}

# 2. Gradient Boosting with Hyperparameter Tuning
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

gb_param_dist = {
    'regressor__n_estimators': randint(100, 300),
    'regressor__learning_rate': uniform(0.01, 0.2),
    'regressor__max_depth': randint(3, 10),
    'regressor__min_samples_split': randint(2, 10),
    'regressor__min_samples_leaf': randint(1, 10),
    'regressor__subsample': uniform(0.7, 0.3)
}

# 3. ElasticNet with Hyperparameter Tuning
en_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(random_state=42))
])

en_param_dist = {
    'regressor__alpha': uniform(0.01, 1.0),
    'regressor__l1_ratio': uniform(0.0, 1.0)
}

# 4. SVR with Hyperparameter Tuning
svr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

svr_param_dist = {
    'regressor__C': uniform(0.1, 10),
    'regressor__gamma': ['scale', 'auto'],
    'regressor__kernel': ['linear', 'rbf', 'poly']
}

# Dictionary to store models and their parameters
models = {
    'Random Forest': (rf_pipeline, rf_param_dist),
    'Gradient Boosting': (gb_pipeline, gb_param_dist),
    'ElasticNet': (en_pipeline, en_param_dist),
    'SVR': (svr_pipeline, svr_param_dist)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for model_name, (pipeline, param_dist) in models.items():
    print(f"\n{'-'*50}\nTraining {model_name}...\n{'-'*50}")
    
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,  # Reduced for faster execution
        cv=5,
        scoring='r2',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        'model': best_model,
        'best_params': random_search.best_params_,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"R² = {r2:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE = {mae:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Compare models
print("\n--- Model Comparison ---")
model_comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'R²': [results[model]['r2'] for model in results],
    'RMSE': [results[model]['rmse'] for model in results],
    'MAE': [results[model]['mae'] for model in results]
}).sort_values('R²', ascending=False)

print(model_comparison)

# Get the best model based on R²
best_model_name = model_comparison.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")

# Feature importance for the best model (if applicable)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get the feature selector step to find which features were selected
    if best_model_name == 'Random Forest' and 'feature_selection' in best_model.named_steps:
        selected = best_model.named_steps['feature_selection']
        selected_indices = selected.get_support(indices=True)
        
        # Get column names after preprocessing
        feature_names = X.columns[selected_indices]
        
        # Get importance values
        importances = best_model.named_steps['regressor'].feature_importances_
        
    else:  # For Gradient Boosting
        feature_names = X.columns
        importances = best_model.named_steps['regressor'].feature_importances_
    
    # Check lengths match
    if len(importances) == len(feature_names):
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_n = min(20, len(importance_df))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"WARNING: Feature importance visualization skipped due to length mismatch.")
        print(f"  Features: {len(feature_names)}, Importances: {len(importances)}")
        
# Final Ensemble (Optional)
print("\n--- Creating Ensemble Model ---")
# Make predictions with all models
ensemble_predictions = np.column_stack([
    results[model]['model'].predict(X_test) for model in results
])

# Simple averaging ensemble
ensemble_pred = np.mean(ensemble_predictions, axis=1)
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

print(f"Ensemble Model Performance:")
print(f"R² = {ensemble_r2:.4f}")
print(f"RMSE = {ensemble_rmse:.4f}")
print(f"MAE = {ensemble_mae:.4f}")

# Plot ensemble results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ensemble_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Actual vs Predicted Values - Ensemble Model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Update model comparison with ensemble
final_comparison = pd.concat([
    model_comparison,
    pd.DataFrame({
        'Model': ['Ensemble'],
        'R²': [ensemble_r2],
        'RMSE': [ensemble_rmse],
        'MAE': [ensemble_mae]
    })
]).sort_values('R²', ascending=False)

print("\nFinal Model Comparison:")
print(final_comparison)
