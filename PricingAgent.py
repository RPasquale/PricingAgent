import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque
import random
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import math
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
import os
import torch
import joblib
from scipy.stats import linregress

MODEL_DIR = 'DRL_MODELS'
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to save PyTorch models
def save_model(model, name):
    path = os.path.join(MODEL_DIR, name)
    torch.save(model.state_dict(), path)
    print(f"Saved model: {name}")

# Function to load PyTorch models
def load_model(model, name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"Loaded model: {name}")
    return model

# Function to save scikit-learn models
def save_sklearn_model(model, name):
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, path)
    print(f"Saved scikit-learn model: {name}")

# Function to load scikit-learn models
def load_sklearn_model(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"Loaded scikit-learn model: {name}")
        return model
    return None

# Function to save preprocessing pipelines or transformers
def save_preprocessor(preprocessor, name):
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(preprocessor, path)
    print(f"Saved preprocessor: {name}")

# Function to load preprocessing pipelines or transformers
def load_preprocessor(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        preprocessor = joblib.load(path)
        print(f"Loaded preprocessor: {name}")
        return preprocessor
    return None


# Function to save the entire agent
def save_agent(agent, name):
    path = os.path.join(MODEL_DIR, name)
    save_dict = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'memory': list(agent.memory),
        'entropy_weight': agent.entropy_weight,
        'scheduler_state_dict': agent.scheduler.state_dict()
    }
    torch.save(save_dict, path)
    print(f"Agent saved: {name}")

# Function to load the entire agent
def load_agent(agent, name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.memory = deque(checkpoint['memory'], maxlen=100000)
        agent.entropy_weight = checkpoint['entropy_weight']
        agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Agent loaded: {name}")
    return agent

# Function to save MCTS
def save_mcts(mcts, name):
    path = os.path.join(MODEL_DIR, name)
    save_dict = {
        'num_simulations': mcts.num_simulations,
        'c_puct': mcts.c_puct,
        'progressive_widening_constant': mcts.progressive_widening_constant
    }
    torch.save(save_dict, path)
    print(f"MCTS saved: {name}")

# Function to load MCTS
def load_mcts(mcts, name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        mcts.num_simulations = checkpoint['num_simulations']
        mcts.c_puct = checkpoint['c_puct']
        mcts.progressive_widening_constant = checkpoint['progressive_widening_constant']
        print(f"MCTS loaded: {name}")
    return mcts

# Function to save a model with state_dict (like ValueNetwork, PolicyNetwork, CombinedNetwork)
def save_model_state_dict(model, name):
    path = os.path.join(MODEL_DIR, name)
    torch.save(model.state_dict(), path)
    print(f"Model state_dict saved: {name}")

# Function to load a model with state_dict
def load_model_state_dict(model, name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"Model state_dict loaded: {name}")
    return model



def compute_dataset_stats(df):
    stats = {
        'avg_profit': df['profit'].mean(),
        'std_profit': df['profit'].std(),
        'avg_quantity': df['line_quantity'].mean(),
        'avg_freight_price': df['line_freight_cost'].mean(),
        'price_elasticity': -1.5,  
        'segment_multipliers': df.groupby('customer_segment')['delivered_margin'].mean().to_dict(),
        'monthly_multipliers': df.groupby('month')['delivered_margin'].mean() / df['delivered_margin'].mean()
    }
    return stats

def preprocess_data(df):
    # Convert date and extract features
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Feature engineering for numeric columns
    if all(col in df.columns for col in ['line_product_revenue', 'line_product_cost']):
        df['revenue_to_cost_ratio'] = np.where(df['line_product_cost'] != 0,
                                               df['line_product_revenue'] / df['line_product_cost'],
                                               0)
        df['revenue_to_cost_ratio'] = np.clip(df['revenue_to_cost_ratio'], 0, 10)

    if all(col in df.columns for col in ['line_freight_cost', 'line_product_cost']):
        df['freight_to_product_cost_ratio'] = np.where(df['line_product_cost'] != 0,
                                                       df['line_freight_cost'] / df['line_product_cost'],
                                                       0)
        df['freight_to_product_cost_ratio'] = np.clip(df['freight_to_product_cost_ratio'], 0, 10)


    # Moving averages with error handling
    for col in ['line_product_revenue', 'line_freight_cost']:
        if col in df.columns:
            df[f'avg_{col}_7d'] = df.groupby('customer_name')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )

    # Margin calculations
    if all(col in df.columns for col in ['line_product_revenue', 'line_product_cost']):
        df['product_margin'] = np.where(df['line_product_revenue'] != 0,
                                        ((df['line_product_revenue'] - df['line_product_cost']) / df['line_product_revenue']) * 100,
                                        0)

    if all(col in df.columns for col in ['line_freight_revenue', 'line_freight_cost']):
        df['freight_margin'] = np.where(df['line_freight_revenue'] != 0,
                                        ((df['line_freight_revenue'] - df['line_freight_cost']) / df['line_freight_revenue']) * 100,
                                        0)

    if all(col in df.columns for col in ['line_product_revenue', 'line_freight_revenue', 'line_product_cost', 'line_freight_cost']):
        df['delivered_margin'] = np.where((df['line_product_revenue'] + df['line_freight_revenue']) != 0,
                                          ((df['line_product_revenue'] + df['line_freight_revenue'] -
                                            df['line_product_cost'] - df['line_freight_cost']) /
                                           (df['line_product_revenue'] + df['line_freight_revenue'])) * 100,
                                          0)

    # Calculate total revenue and total cost
    if all(col in df.columns for col in ['line_product_revenue', 'line_freight_revenue']):
        df['total_revenue'] = df['line_product_revenue'] + df['line_freight_revenue']
    if all(col in df.columns for col in ['line_product_cost', 'line_freight_cost']):
        df['total_cost'] = df['line_product_cost'] + df['line_freight_cost']

    # Feature: line_volume_cbm_to_delivery_margin_ratio
    if all(col in df.columns for col in ['line_volume_cbm', 'delivered_margin']):
        df['line_volume_cbm_to_delivery_margin_ratio'] = np.where(df['delivered_margin'] != 0,
                                                                  df['line_volume_cbm'] / df['delivered_margin'],
                                                                  0)

    # Identify numerical and categorical columns
    all_features = ['line_volume_cbm', 'line_quantity', 'line_product_revenue', 'line_product_cost',
                    'line_freight_revenue', 'line_freight_cost', 'month', 'day_of_week',
                    'revenue_to_cost_ratio', 'freight_to_product_cost_ratio',
                    'product_margin', 'freight_margin', 'delivered_margin',
                    'line_volume_cbm_to_delivery_margin_ratio',
                    'category_1', 'category_2', 'category_3', 'item_number', 'order_channel', 'order_region', 'customer_agreement']

    # Only include features that are present in the DataFrame
    available_features = [col for col in all_features if col in df.columns]

    numerical_features = [col for col in available_features if df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in available_features if df[col].dtype == 'object']

    # Ensure 'day_of_week' is in numerical_features
    if 'day_of_week' not in numerical_features:
        numerical_features.append('day_of_week')

    # Encode categorical features
    for cat_feature in categorical_features:
        df[cat_feature] = df[cat_feature].astype('category').cat.codes

    # Prepare data for prediction models
    X = df[numerical_features + categorical_features]
    y = df['delivered_margin'] if 'delivered_margin' in df.columns else df['line_freight_revenue'] - df['line_freight_cost']

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Fit the preprocessor and transform the data
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y, preprocessor, df, numerical_features, categorical_features

# Training function with learning rate scheduler
def train_reward_model(model, X, y, optimizer, criterion, scheduler, epochs=10, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def calculate_reward(input_data, preprocessor, profit_model, quantity_model, nn_model, dataset_stats, categorical_features, input_size):
    # Prepare input for models
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required columns are present
    for column in preprocessor.feature_names_in_:
        if column not in input_df.columns:
            input_df[column] = 0  # or some appropriate default value

    # Ensure categorical features are properly encoded
    for feature in categorical_features:
        if feature in input_df.columns:
            input_df[feature] = input_df[feature].astype('category')
    
    input_array = preprocessor.transform(input_df)
    
    # Ensure input size matches the model's expected input size
    if input_array.shape[1] != input_size:
        raise ValueError(f"Input size mismatch. Expected {input_size}, got {input_array.shape[1]}")
    
    # Predict profit and quantity
    predicted_profit = profit_model(torch.FloatTensor(input_array).to(device)).item()
    predicted_quantity = quantity_model(torch.FloatTensor(input_array).to(device)).item()
    
    # Get NN prediction
    nn_prediction = nn_model(torch.FloatTensor(input_array).to(device)).item()
    
    # Calculate the difference between the average prediction and the actual margin
    margin_difference = nn_prediction - input_data['new_freight_price']
    
    # Calculate base reward
    base_reward = (predicted_profit / dataset_stats['avg_profit']) * (predicted_quantity / dataset_stats['avg_quantity'])
    
    # Adjust reward based on margin difference
    margin_adjustment = 1 + (margin_difference / input_data['new_freight_price'])
    
    # Adjust reward based on customer segment
    segment_multiplier = dataset_stats['segment_multipliers'].get(input_data.get('customer_segment', 0), 1)
    
    # Adjust reward based on seasonality (month)
    month_multiplier = dataset_stats['monthly_multipliers'][input_data.get('month', 1)]
    
    # Calculate final reward
    final_reward = base_reward * margin_adjustment * segment_multiplier * month_multiplier
    
    return torch.tensor(final_reward, device=device)

class EnhancedNet(nn.Module):
    def __init__(self, input_size):
        super(EnhancedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class CombinedNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CombinedNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        ).to(device)
        self.policy_head = nn.Linear(128, action_size).to(device)
        self.value_head = nn.Linear(128, 1).to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.shared_layers(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class MCTS:
    def __init__(self, model, env, num_simulations=500, c_puct=10.0):
        self.model = model
        self.env = env  # Store the environment
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.progressive_widening_constant = 1.0

    def search(self, state, n_parallel=10):
        root = Node(state, c_puct=self.c_puct)
        
        for _ in range(self.num_simulations // n_parallel):
            leaves = []
            search_paths = []
            
            # Selection
            for _ in range(n_parallel):
                node = root
                search_path = [node]
                while node.expanded():
                    action, node = node.select_child()
                    # Use the environment to get the next state
                    next_state, reward, done = self.env.step(action)
                    node.state = next_state
                    node.reward = reward
                    node.done = done
                    search_path.append(node)
                    if done:
                        break
                leaves.append(node)
                search_paths.append(search_path)
            
            # Expansion and Evaluation
            leaf_states = torch.stack([leaf.state for leaf in leaves])
            policy_logits, values = self.model(leaf_states)
            policies = F.softmax(policy_logits, dim=1)
            
            for leaf, policy, value, search_path in zip(leaves, policies, values, search_paths):
                if not leaf.expanded() and not leaf.done:
                    leaf.expand(leaf.state, policy)
                self.backpropagate(search_path, value.item())

        return self.select_action(root)

    def select_action(self, root):
        visit_counts = torch.tensor([child.visit_count for child in root.children.values()])
        selected_action = visit_counts.argmax().item()
        print(f"Selected Action: {selected_action}")
        return selected_action

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

class Node:
    def __init__(self, state, parent=None, action=None, c_puct=1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.policy = None
        self.c_puct = c_puct
        self.reward = 0
        self.done = False

    def expanded(self):
        return len(self.children) > 0

    def select_child(self):
        visit_counts = torch.tensor([child.visit_count for child in self.children.values()], dtype=torch.float32, device=device)
        total_sqrt_n = torch.sqrt(visit_counts.sum())
        
        actions = list(self.children.keys())
        policy_scores = self.policy[actions]
        
        uct_scores = (
            torch.tensor([child.value() for child in self.children.values()], dtype=torch.float32, device=device) +
            self.c_puct * policy_scores * total_sqrt_n / (1 + visit_counts)
        )
        
        best_action = actions[uct_scores.argmax().item()]
        return best_action, self.children[best_action]

    def expand(self, state, policy):
        self.policy = policy
        for action in range(len(policy)):
            if policy[action] > 0:
                self.children[action] = Node(state, parent=self, action=action, c_puct=self.c_puct)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        query = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.input_dim)
        output = self.output_linear(context)

        return output

class AttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(AttentionModule, self).__init__()
        self.multi_head_attention = MultiHeadSelfAttention(input_dim, num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        attention_output = self.multi_head_attention(x)
        output = self.layer_norm(x + attention_output)

        if output.size(1) == 1:
            output = output.squeeze(1)

        return output

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention = AttentionModule(128, num_heads=4)
        self.mean = nn.Linear(128, action_size)
        self.log_std = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = torch.relu(x)

        x = self.attention(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self.__init__)
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention = AttentionModule(128, num_heads=4)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = torch.relu(x)

        x = self.attention(x)
        return self.value(x)

# AlphaZero-PPO Agent Definition
class CombinedAlphaZeroPPOAgent:
    def __init__(self, state_size, action_size, env, clip_epsilon=0.2, c1=0.5, c2=0.01):
        self.model = CombinedNetwork(state_size, action_size).to(device)
        self.env = env
        self.mcts = MCTS(self.model, self.env, num_simulations=500, c_puct=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = deque(maxlen=100000) 
        self.batch_size = 256
        self.action_size = action_size
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.scaler = GradScaler()
        self.entropy_weight = 0.05
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)

    def get_action(self, state, training=True, episode=0):
        exploration_rate = max(0.05, min(1, 1.0 - math.log(1 + episode) / math.log(1000)))
        mcts_probability = min(0.8, 0.2 + episode / 1000)
        
        if training and random.random() < exploration_rate:
            chosen_action = random.randint(0, self.action_size - 1)
            print(f"Random Action Chosen: {chosen_action}")
            return chosen_action
        elif training and random.random() < mcts_probability:
            action = self.mcts.search(state)
            print(f"MCTS Chosen Action: {action}")
            return action
        else:
            with torch.no_grad():
                policy_logits, _ = self.model(state.unsqueeze(0))
                temperature = max(0.5, 1.0 - episode / 1000)
                action = torch.multinomial(F.softmax(policy_logits / temperature, dim=1), 1).item()
                print(f"Policy Chosen Action: {action}")
            return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).float().to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones).float().to(device)

        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for states, actions, rewards, next_states, dones in dataloader:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    old_policy_logits, _ = self.model(states)
                    old_policy = F.softmax(old_policy_logits, dim=1)
                    old_log_policy = F.log_softmax(old_policy_logits, dim=1)
                    old_log_policy_actions = old_log_policy.gather(1, actions.unsqueeze(1)).squeeze()
                    
                    _, next_values = self.model(next_states)
                    targets = rewards + (1 - dones) * 0.99 * next_values.squeeze()

            for _ in range(5):  # PPO update iterations
                with autocast():
                    policy_logits, values = self.model(states)
                    
                    new_policy = F.softmax(policy_logits, dim=1)
                    new_log_policy = F.log_softmax(policy_logits, dim=1)
                    new_log_policy_actions = new_log_policy.gather(1, actions.unsqueeze(1)).squeeze()

                    ratio = torch.exp(new_log_policy_actions - old_log_policy_actions)
                    advantages = targets - values.squeeze().detach()

                    surrogate1 = ratio * advantages
                    surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    value_loss = F.mse_loss(values.squeeze(), targets)
                    entropy = -(new_policy * new_log_policy).sum(dim=1).mean()

                    loss = policy_loss + self.c1 * value_loss - self.c2 * entropy + self.entropy_weight * entropy

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

        self.memory = []

# Environment Definition
class AdvancedFreightPricingEnvironment:
    def __init__(self, data, action_size, original_df, feature_names, categorical_features, curriculum_factor=1.0):
        self.data = torch.tensor(data, device=device, dtype=torch.float32)
        self.original_df = original_df
        self.feature_names = feature_names
        self.categorical_features = categorical_features

        self.column_indices = {name: i for i, name in enumerate(feature_names)}
        self.action_size = action_size
        self.current_index = 0
        self.reset_metrics()
        self.data_length = len(self.data)
        self.curriculum_factor = curriculum_factor
        self.dataset_stats = compute_dataset_stats(original_df)
        self.preprocessor = None
        self.profit_model = None
        self.quantity_model = None
        self.nn_model = None
        self.price_elasticity = self.calculate_price_elasticity()
        self.lower_price, self.upper_price, self.best_action_size = self.determine_price_range_and_action_size()


    def set_column_indices(self, column_indices):
        self.column_indices = {col: i for i, col in enumerate(column_indices)}

    def set_models(self, preprocessor, profit_model, quantity_model, nn_model):
        self.preprocessor = preprocessor
        self.feature_names_in_ = preprocessor.feature_names_in_
        self.input_size = profit_model.fc1.in_features  # Add this line
        self.profit_model = profit_model
        self.quantity_model = quantity_model
        self.nn_model = nn_model

    def reset_metrics(self):
        self.total_reward = 0
        self.total_quantity = 0
        self.total_margin = 0

    def reset(self):
        self.current_index = 0
        self.reset_metrics()
        return self.data[self.current_index]

    def calculate_price_elasticity(self):
        # Use the original DataFrame for price elasticity calculation
        df = self.original_df.copy()
        df['freight_revenue'] = np.where(
            df['line_freight_cost'] != 0, df['line_freight_revenue'], df['line_product_revenue'] * 0.15
        )
        df['price'] = df['freight_revenue'] / df['line_quantity']
        grouped_data = df.groupby(['month', 'day_of_week', 'order_channel', 'order_region']).agg(
            avg_price=('price', 'mean'),
            avg_quantity=('line_quantity', 'mean')
        ).reset_index()
        grouped_data['log_price'] = np.log(grouped_data['avg_price'] + 0.01)
        grouped_data['log_quantity'] = np.log(grouped_data['avg_quantity'] + 0.01)
        slope, _, _, _, _ = linregress(grouped_data['log_price'], grouped_data['log_quantity'])
        return torch.tensor(slope, device=device)

    def determine_price_range_and_action_size(self):
        df = self.original_df.copy()  # Use the original DataFrame
        df['freight_revenue'] = np.where(
            df['line_freight_cost'] != 0, df['line_freight_revenue'], df['line_product_revenue'] * 0.15
        )
        df['price'] = df['freight_revenue'] / df['line_quantity']

        lower_price = torch.tensor(df['price'].quantile(0.05), device=device)
        upper_price = torch.tensor(df['price'].quantile(0.95), device=device)
        price_range = upper_price - lower_price
        action_size_suggestions = [50, 100, 200, 500]
        best_action_size = None
        lowest_std_dev = float('inf')

        for size in action_size_suggestions:
            simulated_prices = torch.linspace(lower_price, upper_price, steps=size, device=device)
            simulated_rewards = self.simulate_rewards(simulated_prices)
            std_dev = torch.std(simulated_rewards)
            if std_dev < lowest_std_dev:
                lowest_std_dev = std_dev
                best_action_size = size

        return lower_price, upper_price, best_action_size

    def simulate_rewards(self, prices):
        simulated_rewards = []
        for price in prices:
            demand = torch.exp(-self.price_elasticity * price)
            revenue = price * demand
            # Use the correct index for line_freight_cost
            cost_index = self.column_indices['line_freight_cost']
            profit = revenue - self.data[self.current_index][cost_index] * demand
            simulated_rewards.append(profit)
        return torch.tensor(simulated_rewards, device=device)

    def step(self, action):
        print(f"Environment step: Action={action}")

        current_state = self.data[self.current_index]

        normalized_action = action / (self.action_size - 1)
        price_multiplier = self.lower_price + normalized_action * (self.upper_price - self.lower_price)
        new_freight_price = torch.max(torch.tensor(0.01, device=device), current_state[self.column_indices['line_freight_cost']] * price_multiplier)
        print(f"New freight price: {new_freight_price.item():.2f}")

        base_quantity = current_state[self.column_indices['line_quantity']]
        price_sensitivity = torch.exp(-self.price_elasticity * new_freight_price)
        new_quantity = torch.max(torch.tensor(1, device=device), (base_quantity * price_sensitivity).int())

        # Prepare input for the reward calculation
        input_data = {name: current_state[i].item() for name, i in self.column_indices.items()}
        input_data.update({
            'new_freight_price': new_freight_price.item(),
            'new_quantity': new_quantity.item(),
        })

        # Calculate reward
        reward = calculate_reward(
            input_data,
            self.preprocessor,
            self.profit_model,
            self.quantity_model,
            self.nn_model,
            self.dataset_stats,
            self.categorical_features,
            self.input_size  # Add this line
        )

        # Apply curriculum factor to reward
        reward *= self.curriculum_factor

        # Clip the reward to avoid extreme values
        reward = torch.clamp(reward, min=-10.0, max=10.0)
            
        self.total_reward += reward
        self.total_quantity += new_quantity
        self.total_margin += (new_freight_price - current_state[self.column_indices['line_freight_cost']]) / new_freight_price

        self.current_index = (self.current_index + 1) % self.data_length
        next_state = self.data[self.current_index]

        done = (self.current_index == 0)
        print(f"Step result - Reward: {reward.item():.2f}, Done: {done}")

        return next_state, reward, done

    def increase_difficulty(self, factor=1.1):
        self.curriculum_factor *= factor

    def adjust_curriculum(self, avg_reward):
        if avg_reward > self.target_reward:
            self.curriculum_factor *= 1.1
        else:
            self.curriculum_factor *= 0.9
        self.curriculum_factor = max(0.5, min(2.0, self.curriculum_factor))

    def get_metrics(self):
        steps = self.current_index if self.current_index > 0 else self.data_length
        return {
            "avg_reward": self.total_reward / steps,
            "total_quantity": int(self.total_quantity),
            "avg_margin": self.total_margin / steps
        }



# Training Function
def train_agent(env, agent, episodes, writer=None, patience=50):
    best_avg_reward = float('-inf')
    episodes_without_improvement = 0
    
    for e in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset().to(device)  # Move state to GPU
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.get_action(state, training=True, episode=e)
            next_state, reward, done = env.step(action)
            next_state = next_state.to(device)  # Move next_state to GPU

            episode_reward += reward.item()

            agent.store_transition(state, action, reward, next_state, done)

            state = next_state

            if len(agent.memory) >= agent.batch_size:
                agent.update()

        if (e + 1) % 10 == 0:
            metrics = env.get_metrics()
            current_avg_reward = metrics['avg_reward'].item()
            
            print(f"\nEpisode: {e+1}/{episodes}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Average Reward: {current_avg_reward:.2f}")
            print(f"Total Quantity: {metrics['total_quantity']}")
            print(f"Average Margin: {metrics['avg_margin'].item():.2f}")

            if writer:
                writer.add_scalar('Reward/episode', episode_reward, e)
                writer.add_scalar('Reward/average', current_avg_reward, e)
                writer.add_scalar('Quantity/total', metrics['total_quantity'], e)
                writer.add_scalar('Margin/average', metrics['avg_margin'].item(), e)

            if current_avg_reward > best_avg_reward:
                best_avg_reward = current_avg_reward
                torch.save(agent.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with average reward: {best_avg_reward:.2f}")
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 10

            if episodes_without_improvement >= patience:
                print(f"Early stopping triggered after {e+1} episodes")
                break

        if (e + 1) % 50 == 0:
            env.increase_difficulty(factor=1.05)
            print(f"Increased environment difficulty. New factor: {env.curriculum_factor:.2f}")

        agent.entropy_weight = max(0.01, agent.entropy_weight * 0.995)

    print("Training completed.")
    return best_avg_reward


if __name__ == "__main__":
    # Define NeuralNetwork class outside of the conditional block
    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Flag to control whether to train or load models
    train_models = False  # Set to True if you want to train models, False to load existing models

    print("Loading and preprocessing data")
    df = pd.read_csv('marginwell_sample_lightbulb_output_dataset_1.csv')
    X_preprocessed, y, preprocessor, df, numerical_features, categorical_features = preprocess_data(df)

    # Save and load preprocessor
    save_preprocessor(preprocessor, 'preprocessor.pkl')
    preprocessor = load_preprocessor('preprocessor.pkl')

    # Calculate profit if not already present
    if 'profit' not in df.columns:
        df['profit'] = df['total_revenue'] - df['total_cost']

    # Ensure all necessary columns are present
    missing_cols = set(numerical_features + categorical_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    print("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    if train_models:
        print("Training neural network models...")

        # Ensure 'delivered_margin' is calculated
        if 'delivered_margin' not in df.columns:
            df['delivered_margin'] = np.where((df['line_product_revenue'] + df['line_freight_revenue']) != 0,
                                            ((df['line_product_revenue'] + df['line_freight_revenue'] -
                                                df['line_product_cost'] - df['line_freight_cost']) /
                                            (df['line_product_revenue'] + df['line_freight_revenue'])) * 100,
                                            0)

        # Convert data to tensors
        X_train_tensor = torch.tensor(X_train).float().to(device)
        y_train_tensor = torch.tensor(y_train.values).float().to(device)

        y_profit = df['profit'].values
        y_quantity = df['line_quantity'].values
        y_delivered_margin = df['delivered_margin'].values

        # Split the data
        X_train, X_test, y_profit_train, y_profit_test, y_quantity_train, y_quantity_test, y_deliv_margin_train, y_deliv_margin_test = train_test_split(
            X_preprocessed, y_profit, y_quantity, y_delivered_margin, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_profit_train_tensor = torch.FloatTensor(y_profit_train).to(device)
        y_profit_test_tensor = torch.FloatTensor(y_profit_test).to(device)
        y_quantity_train_tensor = torch.FloatTensor(y_quantity_train).to(device)
        y_quantity_test_tensor = torch.FloatTensor(y_quantity_test).to(device)
        y_deliv_margin_train_tensor = torch.FloatTensor(y_deliv_margin_train).to(device)
        y_deliv_margin_test_tensor = torch.FloatTensor(y_deliv_margin_test).to(device)

        # Initialize neural networks
        profit_model = EnhancedNet(X_train.shape[1]).to(device)
        quantity_model = EnhancedNet(X_train.shape[1]).to(device)
        deliv_margin_model = NeuralNetwork(X_train.shape[1]).to(device)
        
        # Define loss function and optimizer
        criterion = nn.SmoothL1Loss()  # Huber loss
        profit_optimizer = optim.AdamW(profit_model.parameters(), lr=0.001, weight_decay=0.01)
        quantity_optimizer = optim.AdamW(quantity_model.parameters(), lr=0.001, weight_decay=0.01)
        deliv_margin_optimizer = optim.AdamW(deliv_margin_model.parameters(), lr=0.001, weight_decay=0.01)

        # Learning rate scheduler
        profit_scheduler = optim.lr_scheduler.ReduceLROnPlateau(profit_optimizer, mode='min', factor=0.5, patience=5)
        quantity_scheduler = optim.lr_scheduler.ReduceLROnPlateau(quantity_optimizer, mode='min', factor=0.5, patience=5)
        deliv_margin_scheduler = optim.lr_scheduler.ReduceLROnPlateau(deliv_margin_optimizer, mode='min', factor=0.5, patience=5)

        # Creating datasets and dataloaders
        profit_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_profit_train_tensor)
        profit_dataloader = DataLoader(profit_dataset, batch_size=64, shuffle=True)

        quantity_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_quantity_train_tensor)
        quantity_dataloader = DataLoader(quantity_dataset, batch_size=64, shuffle=True)

        deliv_margin_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_deliv_margin_train_tensor)
        deliv_dataloader = DataLoader(deliv_margin_dataset, batch_size=64, shuffle=True)
        
        # Training loop
        epochs = 11
        for epoch in range(epochs):
            for batch_X, batch_y in profit_dataloader:
                profit_optimizer.zero_grad()
                outputs = profit_model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                profit_optimizer.step()
            
            profit_scheduler.step(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                # Save the model after every 10 epochs
        save_model(profit_model, 'profit_model.pth')
        del profit_optimizer, profit_scheduler, profit_dataset, profit_dataloader
        torch.cuda.empty_cache()

        for epoch in range(epochs):
            for batch_X, batch_y in quantity_dataloader:
                quantity_optimizer.zero_grad()
                outputs = quantity_model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                quantity_optimizer.step()
            
            quantity_scheduler.step(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                # Save the model after every 10 epochs
        save_model(quantity_model, 'quantity_model.pth')
        del quantity_optimizer, quantity_scheduler, quantity_dataset, quantity_dataloader
        torch.cuda.empty_cache()

        for epoch in range(epochs):
            for batch_X, batch_y in deliv_dataloader:
                deliv_margin_optimizer.zero_grad()
                outputs = deliv_margin_model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                deliv_margin_optimizer.step()
            
            deliv_margin_scheduler.step(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                # Save the model after every 10 epochs
        save_model(deliv_margin_model, 'nn_delivery_margin_model.pth')
        del deliv_margin_optimizer, deliv_margin_scheduler, deliv_margin_dataset, deliv_dataloader
        torch.cuda.empty_cache()

    else:
        print("Loading saved models")
        profit_model = load_model(EnhancedNet(X_train.shape[1]).to(device), 'profit_model.pth')
        quantity_model = load_model(EnhancedNet(X_train.shape[1]).to(device), 'quantity_model.pth')
        deliv_margin_model = load_model(NeuralNetwork(X_train.shape[1]).to(device), 'nn_delivery_margin_model.pth')

        # Predict using the loaded nn model
        X_tensor = torch.tensor(X_preprocessed).float().to(device)
        nn_predictions = deliv_margin_model(X_tensor).detach().cpu().numpy()
        df['nn_delivery_margin_pred'] = nn_predictions

    # Predict using the loaded nn model
    X_tensor = torch.tensor(X_preprocessed).float().to(device)
    nn_predictions = deliv_margin_model(X_tensor).detach().cpu().numpy()
    df['nn_delivery_margin_pred'] = nn_predictions
    # Ensure the 'nn_delivery_margin_pred' column is available
    if 'nn_delivery_margin_pred' not in df.columns:
        raise ValueError("'nn_delivery_margin_pred' column is missing. Ensure the model prediction step was successful.")

    # Customer segmentation
    customer_features = df.groupby('customer_name').agg({
        'total_revenue': 'sum',
        'line_quantity': 'sum',
        'delivered_margin': 'mean',
        'nn_delivery_margin_pred': 'mean'
    }).reset_index()

    scaler = StandardScaler()
    customer_features_scaled = scaler.fit_transform(customer_features.drop('customer_name', axis=1))

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_features['segment'] = kmeans.fit_predict(customer_features_scaled)

    # Map customer segments and average predictions back to the original dataframe
    customer_segment_map = dict(zip(customer_features['customer_name'], customer_features['segment']))
    customer_nn_pred_map = dict(zip(customer_features['customer_name'], customer_features['nn_delivery_margin_pred']))

    df['customer_segment'] = df['customer_name'].map(customer_segment_map)
    df['customer_avg_nn_pred'] = df['customer_name'].map(customer_nn_pred_map)

    # Prepare final data array
    final_numerical_features = numerical_features + [
        'customer_segment', 'nn_delivery_margin_pred', 'customer_avg_nn_pred',
    ]
    final_categorical_features = categorical_features

    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), final_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_categorical_features)
        ])

    final_data_array = final_preprocessor.fit_transform(df[final_numerical_features + final_categorical_features])

    print(f"Data shape after preprocessing: {final_data_array.shape}")

    print("Saving final preprocessor and customer segmentation maps")
    save_preprocessor(final_preprocessor, 'final_preprocessor.pkl')
    save_preprocessor(customer_segment_map, 'customer_segment_map.pkl')
    save_preprocessor(customer_nn_pred_map, 'customer_nn_pred_map.pkl')

    # Splitting final data into train and test sets for RL environment
    train_data, test_data = train_test_split(final_data_array, test_size=0.2, random_state=42)

    print("Initializing environment and agent")
    state_size = train_data.shape[1]
    action_size = 100
    final_feature_names = final_numerical_features + final_categorical_features
    env = AdvancedFreightPricingEnvironment(train_data, action_size, df, final_feature_names, categorical_features)
    env.set_column_indices({name: i for i, name in enumerate(final_feature_names)})
    env.set_models(preprocessor, profit_model, quantity_model, deliv_margin_model)

    agent = CombinedAlphaZeroPPOAgent(state_size, action_size, env)

    if not os.path.exists(os.path.join(MODEL_DIR, 'agent_model.pth')):
        writer = SummaryWriter('runs/freight_pricing_experiment_alphazero')
        best_reward = train_agent(env, agent, episodes=500, writer=writer, patience=50)
        save_agent(agent, 'combined_alphazero_ppo_agent.pth')
        save_model_state_dict(agent.model, 'combined_network.pth')
        save_model_state_dict(agent.model.policy_head, 'policy_network.pth')
        save_model_state_dict(agent.model.value_head, 'value_network.pth')
        save_mcts(agent.mcts, 'mcts.pth')
        writer.close()
    else:
        # Load models and agent
        agent = load_agent(agent, 'combined_alphazero_ppo_agent.pth')
        agent.model = load_model_state_dict(agent.model, 'combined_network.pth')
        agent.model.policy_head = load_model_state_dict(agent.model.policy_head, 'policy_network.pth')
        agent.model.value_head = load_model_state_dict(agent.model.value_head, 'value_network.pth')
        agent.mcts = load_mcts(agent.mcts, 'mcts.pth')

    print(f"Best average reward achieved: {best_reward:.2f}")
    print("Training completed. The agent is now ready for freight price predictions.")

    print("Starting evaluation on test data")
    test_env = AdvancedFreightPricingEnvironment(test_data, action_size, df, final_feature_names, categorical_features)
    test_env.set_column_indices({name: i for i, name in enumerate(final_feature_names)})
    test_env.set_models(final_preprocessor, profit_model, quantity_model, deliv_margin_model)
    
    state = test_env.reset()
    done = False
    test_rewards = []

    while not done:
        action = agent.get_action(state, training=False)
        state, reward, done = test_env.step(action)
        test_rewards.append(reward)

    print(f"Evaluation completed. Average test reward: {np.mean(test_rewards):.2f}")

    print("Saving trained model")
    save_agent(agent, 'combined_alphazero_ppo_agent.pth')
    save_model_state_dict(agent.model, 'trained_model.pth')
    print("Model saved. Process complete.")


