import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If CUDA is available, print some additional information
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Custom transformer for calculating the delivery margin
class DeliveryMarginTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['delivered_margin'] = np.where((X_['line_product_revenue'] + X_['line_freight_revenue']) != 0, 
                                          ((X_['line_product_revenue'] + X_['line_freight_revenue'] - 
                                            X_['line_product_cost'] - X_['line_freight_cost']) / 
                                           (X_['line_product_revenue'] + X_['line_freight_revenue'])) * 100, 
                                          0)
        return X_

# Load data
df = pd.read_csv('G:\\My Drive\\marginwell\\marginwell_sample_lightbulb_output_dataset_1.csv')

# Split data into features and target
X = df.drop(['order_number', 'order_date', 'unique_identifier_for_this_line', 'customer_name', 'customer_agreement'], axis=1)
y = None  # We'll create this later using our transformer

# Perform train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# Create full preprocessing pipeline
full_pipeline = Pipeline([
    ('margin', DeliveryMarginTransformer()),
    ('preprocessor', preprocessor)
])

# Fit the pipeline on training data and transform both train and test
X_train_processed = full_pipeline.fit_transform(X_train)
X_test_processed = full_pipeline.transform(X_test)

# Extract the target variable
y_train = X_train_processed[:, 0]  # Assuming 'delivered_margin' is the first column
y_test = X_test_processed[:, 0]

# Remove the target variable from the feature set
X_train_processed = X_train_processed[:, 1:]
X_test_processed = X_test_processed[:, 1:]

# Convert to PyTorch tensors and move to GPU
X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test_processed).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)


# Define the model
class MarginPredictor(nn.Module):
    def __init__(self, input_size):
        super(MarginPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze()

# Function to train and evaluate the model
def train_and_evaluate(model, train_loader, val_tensor, val_target, criterion, optimizer, device):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_tensor)
        val_loss = criterion(val_outputs, val_target).item()
        val_mse = mean_squared_error(val_target.cpu().numpy(), val_outputs.cpu().numpy())
        val_mae = mean_absolute_error(val_target.cpu().numpy(), val_outputs.cpu().numpy())
        val_r2 = r2_score(val_target.cpu().numpy(), val_outputs.cpu().numpy())
    
    return val_loss, val_mse, val_mae, val_r2

# Set up cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Training parameters
input_size = X_train_tensor.shape[1]
batch_size = 64
num_epochs = 100

# Lists to store results
cv_scores = []
cv_models = []

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tensor), 1):
    print(f"Fold {fold}")
    
    # Split data
    X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
    y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]
    
    # Create data loader
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, criterion, and optimizer
    model = MarginPredictor(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        val_loss, val_mse, val_mae, val_r2 = train_and_evaluate(model, train_loader, X_val_fold, y_val_fold, criterion, optimizer, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
    
    # Store best model and score
    cv_models.append(best_model)
    cv_scores.append(val_r2)
    print(f"Fold {fold} Validation R2 Score: {val_r2:.4f}")

print(f"\nMean CV R2 Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train final model on entire training set
final_model = MarginPredictor(input_size).to(device)
final_criterion = nn.MSELoss()
final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)

final_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    train_and_evaluate(final_model, final_train_loader, X_train_tensor, y_train_tensor, final_criterion, final_optimizer, device)

# Evaluate on test set
final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_tensor).cpu().numpy()

mse = mean_squared_error(y_test, test_outputs)
mae = mean_absolute_error(y_test, test_outputs)
r2 = r2_score(y_test, test_outputs)

print("\nFinal Test Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_outputs, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Margin')
plt.ylabel('Predicted Margin')
plt.title('Predicted vs Actual Delivery Margin')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
plt.close()

# Feature importance (using gradient-based method)
def feature_importance_gradient(model, X):
    model.eval()
    X.requires_grad_(True)
    outputs = model(X)
    outputs.sum().backward()
    feature_importance = X.grad.abs().mean(0)
    return feature_importance.detach().cpu().numpy()

feature_importance = feature_importance_gradient(final_model, X_test_tensor)

# Get feature names from the preprocessor
feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Ensure feature_importance and feature_names have the same length
if len(feature_importance) != len(feature_names):
    print(f"Warning: Feature importance length ({len(feature_importance)}) "
          f"does not match feature names length ({len(feature_names)})")
    # Use the shorter length to avoid errors
    min_length = min(len(feature_importance), len(feature_names))
    feature_importance = feature_importance[:min_length]
    feature_names = feature_names[:min_length]

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features (Gradient-based):")
print(feature_importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance_df.head(20).plot(x='feature', y='importance', kind='bar')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Print all feature names and their corresponding importance
print("\nAll Features and Their Importance:")
for feature, importance in zip(feature_names, feature_importance):
    print(f"{feature}: {importance:.6f}")

# Additional debugging information
print("\nDebugging Information:")
print(f"Number of features in the model: {input_size}")
print(f"Number of feature names from preprocessor: {len(full_pipeline.named_steps['preprocessor'].get_feature_names_out())}")
print(f"Number of importance values: {len(feature_importance)}")

# Check if any feature was dropped during preprocessing
original_features = list(X.columns)
preprocessed_features = list(full_pipeline.named_steps['preprocessor'].get_feature_names_out())
dropped_features = set(original_features) - set([f.split('__')[1] for f in preprocessed_features if '__' in f])
if dropped_features:
    print("\nWarning: The following features were dropped during preprocessing:")
    for feature in dropped_features:
        print(feature)