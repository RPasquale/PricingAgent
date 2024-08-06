import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

class FreightCostPredictor(nn.Module):
    def __init__(self, input_size):
        super(FreightCostPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))  
        x = self.relu(self.fc4(x))
        return x

class MarginPredictor(nn.Module):
    def __init__(self, input_size):
        super(MarginPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size + 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x, freight_revenue):
        x = torch.cat([x, freight_revenue], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class MemoryEfficientDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_freight_cost, y_margin):
        self.X = X.cpu().numpy()
        self.y_freight_cost = y_freight_cost.cpu().numpy()
        self.y_margin = y_margin.cpu().numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).float(),
                torch.from_numpy(self.y_freight_cost[idx]).float(),
                torch.from_numpy(self.y_margin[idx]).float())

def train_models(freight_cost_model, margin_model, X, y_freight_cost, y_margin, 
                 freight_cost_optimizer, margin_optimizer, criterion, device,
                 num_epochs=100, batch_size=32, accumulation_steps=8):
    dataset = MemoryEfficientDataset(X, y_freight_cost, y_margin)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        freight_cost_losses = []
        margin_losses = []
        
        freight_cost_model.train()
        margin_model.train()
        
        for i, (batch_X, batch_y_freight_cost, batch_y_margin) in enumerate(dataloader):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y_freight_cost = batch_y_freight_cost.to(device, non_blocking=True)
            batch_y_margin = batch_y_margin.to(device, non_blocking=True)
            
            # Train freight cost model
            with autocast():
                freight_cost_pred = freight_cost_model(batch_X)
                freight_cost_loss = criterion(freight_cost_pred, batch_y_freight_cost)
            
            scaler.scale(freight_cost_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(freight_cost_optimizer)
                torch.nn.utils.clip_grad_norm_(freight_cost_model.parameters(), max_norm=1.0)
                scaler.step(freight_cost_optimizer)
                scaler.update()
                freight_cost_optimizer.zero_grad(set_to_none=True)
            
            # Train margin model
            with autocast():
                freight_revenue = torch.rand(batch_X.size(0), 1, device=device) * 1000
                margin_pred = margin_model(batch_X, freight_revenue)
                margin_loss = criterion(margin_pred, batch_y_margin)
            
            scaler.scale(margin_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(margin_optimizer)
                torch.nn.utils.clip_grad_norm_(margin_model.parameters(), max_norm=1.0)
                scaler.step(margin_optimizer)
                scaler.update()
                margin_optimizer.zero_grad(set_to_none=True)
            
            freight_cost_losses.append(freight_cost_loss.item())
            margin_losses.append(margin_loss.item())
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Freight Cost Loss: {np.mean(freight_cost_losses):.4f}, '
              f'Margin Loss: {np.mean(margin_losses):.4f}')
        
        if np.mean(margin_losses) < 0.0001:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.cuda.empty_cache()

def optimize_freight_revenue(order_features, line_product_cost, line_product_revenue, freight_cost_model, margin_model, TARGET_PRODUCT_MARGIN, TARGET_FREIGHT_MARGIN, TARGET_DELIVERED_MARGIN, device):
    def objective(freight_revenue):
        freight_revenue_tensor = torch.FloatTensor([freight_revenue]).to(device)
        with torch.no_grad():
            predicted_margin = margin_model(order_features, freight_revenue_tensor).item()
        
        product_margin = (line_product_revenue - line_product_cost) / line_product_revenue
        freight_margin = (freight_revenue - predicted_freight_cost) / freight_revenue if freight_revenue > 0 else -1
        delivered_margin = ((line_product_revenue + freight_revenue - line_product_cost - predicted_freight_cost) / 
                            (line_product_revenue + freight_revenue)) if (line_product_revenue + freight_revenue) > 0 else -1
        
        margin_diff = (
            (product_margin - TARGET_PRODUCT_MARGIN)**2 +
            (freight_margin - TARGET_FREIGHT_MARGIN)**2 +
            (delivered_margin - TARGET_DELIVERED_MARGIN)**2
        )
        
        return margin_diff

    with torch.no_grad():
        predicted_freight_cost = freight_cost_model(order_features).item()
    initial_guess = max(predicted_freight_cost / (1 - TARGET_FREIGHT_MARGIN), 0.01)
    
    result = minimize(objective, initial_guess, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    return max(result.x[0], 0.01), predicted_freight_cost

def preprocess_margins(margins):
    margins = np.clip(margins, -1, 1)
    margins = np.nan_to_num(margins, nan=0.0, posinf=1.0, neginf=-1.0)
    return margins

def load_and_preprocess_data():
    df = pd.read_csv('G:\\My Drive\\marginwell\\marginwell_sample_lightbulb_output_dataset_1.csv')

    TARGET_PRODUCT_MARGIN = 0.492
    TARGET_FREIGHT_MARGIN = -0.1501
    TARGET_DELIVERED_MARGIN = 0.4541

    df['product_margin'] = (df['line_product_revenue'] - df['line_product_cost']) / df['line_product_revenue']
    df['freight_margin'] = (df['line_freight_revenue'] - df['line_freight_cost']) / df['line_freight_revenue']
    df['delivered_margin'] = ((df['line_product_revenue'] + df['line_freight_revenue'] - 
                               df['line_product_cost'] - df['line_freight_cost']) / 
                              (df['line_product_revenue'] + df['line_freight_revenue']))

    features = ['category_1', 'category_2', 'category_3', 'item_number', 'line_volume_cbm', 
                'line_quantity', 'line_product_cost', 'order_channel', 'order_region', 
                'customer_agreement']

    X = df[features]
    y_freight_cost = df['line_freight_cost']
    y_delivered_margin = df['delivered_margin']

    X_train, X_test, y_freight_cost_train, y_freight_cost_test, y_margin_train, y_margin_test = train_test_split(
        X, y_freight_cost, y_delivered_margin, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

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

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (X_train_processed, X_test_processed, y_freight_cost_train, y_freight_cost_test, 
            y_margin_train, y_margin_test, df, X_test, TARGET_PRODUCT_MARGIN, TARGET_FREIGHT_MARGIN, TARGET_DELIVERED_MARGIN)

def main():
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    (X_train_processed, X_test_processed, y_freight_cost_train, y_freight_cost_test, 
     y_margin_train, y_margin_test, df, X_test, TARGET_PRODUCT_MARGIN, TARGET_FREIGHT_MARGIN, 
     TARGET_DELIVERED_MARGIN) = load_and_preprocess_data()

    X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
    y_freight_cost_train_tensor = torch.FloatTensor(y_freight_cost_train.values).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_processed).to(device)
    y_freight_cost_test_tensor = torch.FloatTensor(y_freight_cost_test.values).unsqueeze(1).to(device)
    y_margin_train_tensor = torch.FloatTensor(preprocess_margins(y_margin_train)).unsqueeze(1).to(device)
    y_margin_test_tensor = torch.FloatTensor(preprocess_margins(y_margin_test)).unsqueeze(1).to(device)

    input_size = X_train_tensor.shape[1]
    freight_cost_model = FreightCostPredictor(input_size).to(device)
    margin_model = MarginPredictor(input_size).to(device)

    criterion = nn.MSELoss()
    freight_cost_optimizer = optim.Adam(freight_cost_model.parameters(), lr=0.001)
    margin_optimizer = optim.Adam(margin_model.parameters(), lr=0.001)

    train_models(freight_cost_model, margin_model, X_train_tensor, y_freight_cost_train_tensor, 
                 y_margin_train_tensor, freight_cost_optimizer, margin_optimizer, criterion, device)

    # Evaluate models on test set
    freight_cost_model.eval()
    margin_model.eval()
    with torch.no_grad():
        freight_cost_pred = freight_cost_model(X_test_tensor)
        freight_cost_mse = criterion(freight_cost_pred, y_freight_cost_test_tensor)
        print(f"Freight Cost Test MSE: {freight_cost_mse.item():.4f}")

        # For margin model, we need to provide a random freight revenue
        freight_revenue = torch.rand(X_test_tensor.size(0), 1, device=device) * 1000
        margin_pred = margin_model(X_test_tensor, freight_revenue)
        margin_mse = criterion(margin_pred, y_margin_test_tensor)
        print(f"Margin Test MSE: {margin_mse.item():.4f}")

    sample_order = X_test_tensor[0].unsqueeze(0)
    sample_product_cost = df.loc[X_test.index[0], 'line_product_cost']
    sample_product_revenue = sample_product_cost / (1 - TARGET_PRODUCT_MARGIN)

    optimized_freight_revenue, predicted_freight_cost = optimize_freight_revenue(
        sample_order, sample_product_cost, sample_product_revenue, freight_cost_model, margin_model, 
        TARGET_PRODUCT_MARGIN, TARGET_FREIGHT_MARGIN, TARGET_DELIVERED_MARGIN, device)

    print(f"\nDetailed Results:")
    print(f"Sample Product Cost: ${sample_product_cost:.2f}")
    print(f"Sample Product Revenue: ${sample_product_revenue:.2f}")
    print(f"Predicted Freight Cost: ${predicted_freight_cost:.2f}")
    print(f"Optimized Freight Revenue: ${optimized_freight_revenue:.2f}")
    print(f"Recommended Delivery Price: ${optimized_freight_revenue:.2f}")

    product_margin = (sample_product_revenue - sample_product_cost) / sample_product_revenue
    freight_margin = (optimized_freight_revenue - predicted_freight_cost) / optimized_freight_revenue if optimized_freight_revenue != 0 else 0
    delivered_margin = ((sample_product_revenue + optimized_freight_revenue - sample_product_cost - predicted_freight_cost) / 
                        (sample_product_revenue + optimized_freight_revenue)) if (sample_product_revenue + optimized_freight_revenue) != 0 else 0

    print(f"\nPredicted Margins:")
    print(f"Product Margin: {product_margin:.2%}")
    print(f"Freight Margin: {freight_margin:.2%}")
    print(f"Delivered Margin: {delivered_margin:.2%}")

    print(f"\nTarget Margins:")
    print(f"Product Margin: {TARGET_PRODUCT_MARGIN:.2%}")
    print(f"Freight Margin: {TARGET_FREIGHT_MARGIN:.2%}")
    print(f"Delivered Margin: {TARGET_DELIVERED_MARGIN:.2%}")

    print(f"\nDifferences from Targets:")
    print(f"Product Margin Diff: {(product_margin - TARGET_PRODUCT_MARGIN):.2%}")
    print(f"Freight Margin Diff: {(freight_margin - TARGET_FREIGHT_MARGIN):.2%}")
    print(f"Delivered Margin Diff: {(delivered_margin - TARGET_DELIVERED_MARGIN):.2%}")

if __name__ == '__main__':
    main()