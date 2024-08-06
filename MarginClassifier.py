'''import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# Load and preprocess data
df = pd.read_csv('G:\\My Drive\\marginwell\\marginwell_sample_lightbulb_output_dataset_1.csv')

# Calculate the delivery margin
df['delivered_margin'] = np.where((df['line_product_revenue'] + df['line_freight_revenue']) != 0, 
                                  ((df['line_product_revenue'] + df['line_freight_revenue'] - df['line_product_cost'] - df['line_freight_cost']) / 
                                   (df['line_product_revenue'] + df['line_freight_revenue'])) * 100, 
                                  0)

# Feature Engineering
def engineer_features(df):
    # Extract day of week and month from order_date
    df['order_day_of_week'] = pd.to_datetime(df['order_date']).dt.dayofweek
    df['order_month'] = pd.to_datetime(df['order_date']).dt.month
    
    # Calculate total revenue and total cost per line
    df['total_revenue'] = df['line_product_revenue'] + df['line_freight_revenue']
    df['total_cost'] = df['line_product_cost'] + df['line_freight_cost']
    
    # Calculate profit per item (handle division by zero)
    df['profit_per_item'] = np.where(df['line_quantity'] != 0, 
                                     (df['total_revenue'] - df['total_cost']) / df['line_quantity'], 
                                     0)
    
    # Calculate revenue per CBM (handle division by zero)
    df['revenue_per_cbm'] = np.where(df['line_volume_cbm'] != 0, 
                                     df['total_revenue'] / df['line_volume_cbm'], 
                                     0)
    
    # Create a feature for whether the order has freight revenue
    df['has_freight_revenue'] = (df['line_freight_revenue'] > 0).astype(int)
    
    # Group by order_number and create aggregated features
    order_features = df.groupby('order_number').agg({
        'line_quantity': 'sum',
        'line_volume_cbm': 'sum',
        'total_revenue': 'sum',
        'total_cost': 'sum',
        'unique_identifier_for_this_line': 'count'
    }).rename(columns={'unique_identifier_for_this_line': 'items_in_order'})
    
    # Merge these features back to the original dataframe
    df = df.merge(order_features, on='order_number', suffixes=('', '_order_total'))
    
    # Calculate the percentage of order's total represented by this line item (handle division by zero)
    df['line_percent_of_order_revenue'] = np.where(df['total_revenue_order_total'] != 0, 
                                                   (df['total_revenue'] / df['total_revenue_order_total']) * 100, 
                                                   0)
    df['line_percent_of_order_volume'] = np.where(df['line_volume_cbm_order_total'] != 0, 
                                                  (df['line_volume_cbm'] / df['line_volume_cbm_order_total']) * 100, 
                                                  0)
    
    return df

# Apply feature engineering
df_engineered = engineer_features(df)
# Modify the target variable to be binary
# Modify the target variable to be binary
df['positive_margin'] = (df['delivered_margin'] > 0).astype(int)

# Prepare features and target
X = df.drop(['delivered_margin', 'positive_margin', 'order_number', 'order_date', 'unique_identifier_for_this_line', 'customer_name', 'customer_agreement'], axis=1)
y = df['positive_margin']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

# Fit the preprocessor and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.LongTensor(y_test.values)

# Calculate class weights
class_counts = y_train.value_counts()
total_samples = len(y_train)
class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts])

# Create weighted sampler
sample_weights = [class_weights[t] for t in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Create DataLoader with weighted sampler
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

# Define the model
class DeepMarginPredictor(nn.Module):
    def __init__(self, input_size):
        super(DeepMarginPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.dropout(self.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout(self.relu(self.batch_norm2(self.fc2(x))))
        x = self.dropout(self.relu(self.batch_norm3(self.fc3(x))))
        x = self.fc4(x)
        return x

# Initialize the model
input_size = X_train_tensor.shape[1]
model = DeepMarginPredictor(input_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        epoch_accuracy += (predicted == batch_y).sum().item()
    
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(y_train)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_accuracy = accuracy_score(y_test_tensor, val_predicted)
        val_accuracies.append(val_accuracy)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print()

# Final evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = torch.argmax(test_outputs, dim=1).numpy()
    test_probs = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
    
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    conf_matrix = confusion_matrix(y_test, test_preds)

print("Final Test Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history_improved.png')
plt.close()

# Feature importance (using gradient-based method)
def feature_importance_gradient(model, X):
    model.eval()
    X.requires_grad_(True)
    outputs = model(X)
    probabilities = torch.softmax(outputs, dim=1)
    target_probabilities = probabilities[:, 1]  # Probability of positive margin
    target_probabilities.sum().backward()
    feature_importance = X.grad.abs().mean(0)
    return feature_importance.detach().numpy()

feature_importance = feature_importance_gradient(model, X_test_tensor)
feature_names = preprocessor.get_feature_names_out()

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features (Gradient-based):")
print(feature_importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance_df.head(20).plot(x='feature', y='importance', kind='bar')
plt.title("Feature Importance (Improved Classification Model)")
plt.tight_layout()
plt.savefig('feature_importance_improved.png')
plt.close()'''



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