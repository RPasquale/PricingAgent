print("PART 1")
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load your dataset
df = pd.read_csv('marginwell_sample_lightbulb_output_dataset_1.csv')

# Calculate margin columns
df['product_margin'] = np.where(df['line_product_revenue'] != 0,
                               ((df['line_product_revenue'] - df['line_product_cost']) / df['line_product_revenue']) * 100,
                               0)
df['freight_margin'] = np.where(df['line_freight_revenue'] != 0,
                               ((df['line_freight_revenue'] - df['line_freight_cost']) / df['line_freight_revenue']) * 100,
                               0)
df['delivered_margin'] = np.where((df['line_product_revenue'] + df['line_freight_revenue']) != 0,
                                 ((df['line_product_revenue'] + df['line_freight_revenue'] -
                                   df['line_product_cost'] - df['line_freight_cost']) /
                                  (df['line_product_revenue'] + df['line_freight_revenue'])) * 100,
                                 0)

# Filter orders with negative delivery margins
negative_margin_orders = df[df['delivered_margin'] < 0]['order_number'].unique()

# Create a new dataset containing only the rows of orders with negative delivery margins
negative_margin_data = df[df['order_number'].isin(negative_margin_orders)]

# Aggregate data at the order level
order_agg = negative_margin_data.groupby('order_number').agg({
    'line_quantity': 'sum',
    'line_volume_cbm': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_revenue': 'sum',
    'line_freight_cost': 'sum',
    'delivered_margin': 'mean',  # You might want to consider other aggregation functions here (e.g., min, max)
    'category_1': lambda x: ', '.join(x.unique()),  # Concatenate unique categories
    'category_2': lambda x: ', '.join(x.unique()),
    'category_3': lambda x: ', '.join(x.unique()),
    'item_number': lambda x: ', '.join(x.unique()),
}).reset_index()

# Show results 
print(f'The number of orders with negative delivery margins: {len(negative_margin_orders)}')
print(order_agg.head().to_markdown(index=False, numalign="left", stralign="left"))
print(order_agg.describe().to_markdown(numalign="left", stralign="left"))

# -------------------------------------------------------
print("PART 2")
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('marginwell_sample_lightbulb_output_dataset_1.csv')

# Recalculate margin columns
df['product_margin'] = np.where(df['line_product_revenue'] != 0,
                               ((df['line_product_revenue'] - df['line_product_cost']) / df['line_product_revenue']) * 100,
                               0)
df['freight_margin'] = np.where(df['line_freight_revenue'] != 0,
                               ((df['line_freight_revenue'] - df['line_freight_cost']) / df['line_freight_revenue']) * 100,
                               0)
df['delivered_margin'] = np.where((df['line_product_revenue'] + df['line_freight_revenue']) != 0,
                                 ((df['line_product_revenue'] + df['line_freight_revenue'] -
                                   df['line_product_cost'] - df['line_freight_cost']) /
                                  (df['line_product_revenue'] + df['line_freight_revenue'])) * 100,
                                 0)

# Target Encoding for each categorical variable
for column in ['customer_agreement', 'category_1', 'category_2', 'category_3']:
    mean_target = df.groupby(column)['delivered_margin'].mean()
    df[column + '_encoded'] = df[column].map(mean_target)

# Prepare features and target
X = df[['customer_agreement_encoded', 'category_1_encoded', 'category_2_encoded', 'category_3_encoded', 
        'line_quantity', 'line_volume_cbm', 'line_product_revenue', 
        'line_product_cost', 'line_freight_revenue', 'line_freight_cost']]
y = df['delivered_margin']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=20)
model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_

# Map feature importance to feature names
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})

# Sort by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))  # Increased figure size
plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Delivered Margin')
plt.xticks(fontsize=10)  # Adjust font size for x-axis
plt.yticks(fontsize=10)  # Adjust font size for y-axis
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Display specific feature importance for customer agreements
customer_agreement_importance = feature_importance[feature_importance['feature'] == 'customer_agreement_encoded']
print(customer_agreement_importance)


# Calculate mean delivered margin for each customer agreement
mean_delivered_margin_by_agreement = df.groupby('customer_agreement')['delivered_margin'].mean()

# Sort values for better visualization
mean_delivered_margin_by_agreement = mean_delivered_margin_by_agreement.sort_values(ascending=False)

# Plot the results
plt.figure(figsize=(14, 10))  # Increased figure size for better readability
mean_delivered_margin_by_agreement.plot(kind='barh', color='skyblue')
plt.xlabel('Mean Delivered Margin')
plt.ylabel('Customer Agreement')
plt.title('Impact of Each Customer Agreement on Delivered Margin')
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and adjust font size of x-axis labels
plt.yticks(fontsize=10)  # Adjust font size for y-axis labels
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
plt.show()



# Function to calculate and plot mean delivered margin for any categorical column
def plot_bottom5_mean_delivered_margin(df, category_col):
    mean_delivered_margin_by_category = df.groupby(category_col)['delivered_margin'].mean()
    mean_delivered_margin_by_category = mean_delivered_margin_by_category.sort_values(ascending=True).head(5)
    print(mean_delivered_margin_by_category)
    plt.figure(figsize=(14, 10))  # Increased figure size for better readability
    mean_delivered_margin_by_category.plot(kind='barh', color='skyblue')
    plt.xlabel('Mean Delivered Margin')
    plt.ylabel(category_col)
    plt.title(f'Bottom 5 {category_col} Categories Impacting Delivered Margin')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better fit
    plt.yticks(fontsize=10)  # Adjust font size of y-axis labels
    plt.tight_layout()  # Automatically adjust subplot params for better fit
    plt.show()

# Plot for Category 1
plot_bottom5_mean_delivered_margin(df, 'category_1')

# Plot for Category 2 (Bottom 5)
plot_bottom5_mean_delivered_margin(df, 'category_2')

# Plot for Category 3 (Bottom 5)
plot_bottom5_mean_delivered_margin(df, 'category_3')

# -------------------------------------------------------------

print("PART 3")

import pandas as pd
import numpy as np
from tabulate import tabulate
import os

# Create a directory to store the output files
output_dir = "category_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Load your dataset
df = pd.read_csv('marginwell_sample_lightbulb_output_dataset_1.csv')

# Recalculate delivered_margin
df['delivered_margin'] = np.where((df['line_product_revenue'] + df['line_freight_revenue']) != 0,
                                 ((df['line_product_revenue'] + df['line_freight_revenue'] -
                                   df['line_product_cost'] - df['line_freight_cost']) /
                                  (df['line_product_revenue'] + df['line_freight_revenue'])) * 100,
                                 0)

def analyze_category(df, category_column, category_value):
    # Filter the dataframe for the specific category
    category_df = df[df[category_column] == category_value]
    
    # Group by item_number and calculate relevant metrics
    item_analysis = category_df.groupby('item_number').agg({
        'delivered_margin': 'mean',
        'line_volume_cbm': 'mean',
        'order_region': lambda x: ', '.join(x.unique()),
        'customer_agreement': lambda x: ', '.join(x.unique()),
        'customer_name': lambda x: ', '.join(x.unique()),
        'line_product_revenue': 'sum',
        'line_product_cost': 'sum',
        'line_freight_revenue': 'sum',
        'line_freight_cost': 'sum'
    }).reset_index()
    
    # Sort by delivered margin (most negative first)
    item_analysis = item_analysis.sort_values('delivered_margin', ascending=True)
    
    # Save results to CSV
    filename = f"{output_dir}/{category_column}_{category_value}_item_analysis.csv"
    item_analysis.to_csv(filename, index=False)
    print(f"Item analysis for {category_column}: {category_value} saved to {filename}")
    
    # Save overall delivered margin
    with open(f"{output_dir}/{category_column}_{category_value}_overall_margin.txt", 'w') as f:
        f.write(f"Overall delivered margin: {category_df['delivered_margin'].mean():.2f}%")

def analyze_customer_agreements(df, category_column, category_value):
    category_df = df[df[category_column] == category_value]
    
    agreement_analysis = category_df.groupby('customer_agreement').agg({
        'delivered_margin': 'mean',
        'line_volume_cbm': 'mean',
        'item_number': lambda x: ', '.join(x.unique()),
        'line_product_revenue': 'sum',
        'line_product_cost': 'sum',
        'line_freight_revenue': 'sum',
        'line_freight_cost': 'sum'
    }).reset_index()
    
    # Sort by delivered margin (most negative first)
    agreement_analysis = agreement_analysis.sort_values('delivered_margin', ascending=True)
    
    # Save results to CSV
    filename = f"{output_dir}/{category_column}_{category_value}_customer_agreement_analysis.csv"
    agreement_analysis.to_csv(filename, index=False)
    print(f"Customer agreement analysis for {category_column}: {category_value} saved to {filename}")

# Analyze specific categories
categories_to_analyze = {
    'category_1': ['293fad55b73916bd94f52218a8859fa4', '52e208fd2ce775bda27a4e9a68e96a85', 'e61a42146db4c000c33b75592288b2ff', 'f8762460f4735a774ba593d36db8074c','7171673eb2464a236a83af415b0e2253'],
    'category_2': ['b7d4d528cfcfd4192294dea72944f949', '1ecbee1a8ea3797f2693cb24738bd378', '4edc26d50d15b6d2954d97c07bfad864', 'a63c6cbd86775420a8c66cff3d079958','f8762460f4735a774ba593d36db8074c'],
    'category_3': ['a3005ea40b27e13db1100d61cb3fece9', 'e2e86bc73e52e175d0227c3ccd6e429a','8ef466b230630daebe4c8825164cf989','6ebb0644ea8dc0f2ad24efca4ff97eb6','801be76cbf87b731f93bf16a51d642bb']
}

for category, values in categories_to_analyze.items():
    for value in values:
        analyze_category(df, category, value)
        analyze_customer_agreements(df, category, value)

print(f"All analysis results have been saved to the '{output_dir}' directory.")

