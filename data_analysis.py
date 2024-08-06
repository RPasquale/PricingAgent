# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the cleaned data
#df = pd.read_csv('cleaned_ecommerce_data.csv')
df = pd.read_csv('G:\\My Drive\\marginwell\\marginwell_sample_lightbulb_output_dataset_1.csv')

df['order_date'] = pd.to_datetime(df['order_date'])

# Calculate margins and total revenue
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
df['total_revenue'] = df['line_product_revenue'] + df['line_freight_revenue']
df['avg_price'] = df['line_product_revenue'] / df['line_quantity']

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Section 1: Descriptive Analysis and Visualizations
# ----------------------------------------

# Summary Statistics for Key Metrics
key_metrics = ['line_volume_cbm', 'line_quantity', 'line_product_revenue', 'line_product_cost', 
               'line_freight_revenue', 'line_freight_cost', 'product_margin', 'freight_margin', 'delivered_margin']
summary_stats = df[key_metrics].describe()
print("Summary Statistics for Key Metrics:")
print(summary_stats)

# Distribution of Orders Across Categories
category_counts = {
    'Category 1': df['category_1'].value_counts(),
    'Category 2': df['category_2'].value_counts(),
    'Category 3': df['category_3'].value_counts()
}
for category, counts in category_counts.items():
    print(f"\nTop 10 {category} by order count:")
    print(counts.head(10))

# Distribution of Orders Across Channels and Regions
channel_distribution = df['order_channel'].value_counts()
region_distribution = df['order_region'].value_counts()
print("\nDistribution of orders across channels:")
print(channel_distribution)
print("\nTop 10 regions by order count:")
print(region_distribution.head(10))

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['line_volume_cbm'], bins=50, kde=True)
plt.title('Distribution of Line Volume (CBM)')
plt.xlabel('Volume (CBM)')
plt.savefig('line_volume_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
channel_distribution.plot(kind='bar')
plt.title('Distribution of Orders Across Channels')
plt.xlabel('Channel')
plt.ylabel('Number of Orders')
plt.savefig('order_channel_distribution.png')
plt.close()

plt.figure(figsize=(12, 6))
region_distribution.head(10).plot(kind='bar')
plt.title('Top 10 Regions by Order Count')
plt.xlabel('Region')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_10_regions.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(df['line_product_cost'], df['line_product_revenue'], alpha=0.5)
plt.title('Product Revenue vs. Cost')
plt.xlabel('Product Cost')
plt.ylabel('Product Revenue')
plt.savefig('revenue_vs_cost_scatter.png')
plt.close()

print("\nVisualization images have been saved.")

# Section 2: Time Series Analysis
# ----------------------------------------

# Resample data to daily frequency
daily_data = df.groupby('order_date').agg({
    'line_volume_cbm': 'sum',
    'line_product_revenue': 'sum',
    'line_freight_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_cost': 'sum',
    'product_margin': 'mean',
    'freight_margin': 'mean',
    'delivered_margin': 'mean'
}).reset_index()

daily_data['total_revenue'] = daily_data['line_product_revenue'] + daily_data['line_freight_revenue']
daily_data['total_cost'] = daily_data['line_product_cost'] + daily_data['line_freight_cost']
daily_data.set_index('order_date', inplace=True)
daily_data.sort_index(inplace=True)

# Analyze Trends
plt.figure(figsize=(15, 15))
plt.subplot(5, 1, 1)
daily_data['line_volume_cbm'].plot()
plt.title('Daily Order Volume')
plt.ylabel('Volume (CBM)')

plt.subplot(5, 1, 2)
daily_data['total_revenue'].plot()
plt.title('Daily Revenue')
plt.ylabel('Revenue')

plt.subplot(5, 1, 3)
daily_data['product_margin'].plot()
plt.title('Daily Product Margin')
plt.ylabel('Margin (%)')

plt.subplot(5, 1, 4)
daily_data['freight_margin'].plot()
plt.title('Daily Freight Margin')
plt.ylabel('Margin (%)')

plt.subplot(5, 1, 5)
daily_data['delivered_margin'].plot()
plt.title('Daily Delivered Margin')
plt.ylabel('Margin (%)')

plt.tight_layout()
plt.savefig('daily_trends_with_margins.png')
plt.close()

# Identify Seasonality Patterns
def plot_seasonal_decompose(series, title):
    result = seasonal_decompose(series, model='additive', period=7)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
    result.observed.plot(ax=ax1)
    ax1.set_title(f'{title} - Observed')
    result.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_decomposition.png')
    plt.close()

for column in ['line_volume_cbm', 'total_revenue', 'product_margin', 'freight_margin', 'delivered_margin']:
    plot_seasonal_decompose(daily_data[column], column.replace('_', ' ').title())

# Forecast Future Sales and Demand
def fit_sarima(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    return results

def plot_forecast(series, forecast, title):
    plt.figure(figsize=(15, 7))
    plt.plot(series.index, series, label='Observed')
    
    last_date = series.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast.predicted_mean))
    
    plt.plot(forecast_index, forecast.predicted_mean, color='red', label='Forecast')
    plt.fill_between(forecast_index, 
                     forecast.conf_int().iloc[:, 0],
                     forecast.conf_int().iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title(f'{title} Forecast')
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}_forecast.png')
    plt.close()

metrics = ['total_revenue', 'product_margin', 'freight_margin', 'delivered_margin']
forecast_steps = 30

for metric in metrics:
    model = fit_sarima(daily_data[metric], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    forecast = model.get_forecast(steps=forecast_steps)
    plot_forecast(daily_data[metric], forecast, metric.replace('_', ' ').title())
    
    print(f"\n{metric.replace('_', ' ').title()} Forecast for next 30 days:")
    print(forecast.predicted_mean)
    print(f"\n{metric.replace('_', ' ').title()} SARIMA Model Summary:")
    print(model.summary())

print("\nTime Series Analysis Summary:")
print("\nDaily statistics:")
print(daily_data.describe())

print("\nVisualization images have been saved.")

# Section 3: Clustering and Customer Analysis
# ----------------------------------------

# Aggregate data by customer
customer_data = df.groupby('customer_name').agg({
    'order_number': 'count',
    'line_product_revenue': 'sum',
    'line_freight_revenue': 'sum',
    'line_volume_cbm': 'sum',
    'order_date': lambda x: (pd.to_datetime(x.max()) - pd.to_datetime(x.min())).days + 1
})

customer_data.columns = ['order_count', 'total_product_revenue', 'total_freight_revenue', 'total_volume', 'customer_lifetime_days']

# Calculate average order value and purchase frequency
customer_data['avg_order_value'] = (customer_data['total_product_revenue'] + customer_data['total_freight_revenue']) / customer_data['order_count']
customer_data['purchase_frequency'] = customer_data['order_count'] / customer_data['customer_lifetime_days']

# Improve CLV calculation
customer_data['clv'] = customer_data['avg_order_value'] * customer_data['purchase_frequency'] * customer_data['customer_lifetime_days']
customer_data['clv'] = customer_data['clv'].clip(lower=0)  # Ensure CLV is not negative

# Normalize the features
features = ['order_count', 'total_product_revenue', 'total_freight_revenue', 'total_volume', 'avg_order_value', 'purchase_frequency', 'clv']
scaler = StandardScaler()
normalized_features = scaler.fit_transform(customer_data[features])

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(normalized_features)

# Analyze cluster characteristics
cluster_analysis = customer_data.groupby('cluster').agg({
    'order_count': 'mean',
    'total_product_revenue': 'mean',
    'total_freight_revenue': 'mean',
    'total_volume': 'mean',
    'avg_order_value': 'mean',
    'purchase_frequency': 'mean',
    'clv': 'mean'
}).round(2)

# Count customers in each cluster
cluster_customer_count = customer_data['cluster'].value_counts().sort_index()
cluster_analysis['customer_count'] = cluster_customer_count
cluster_analysis['customer_percentage'] = (cluster_analysis['customer_count'] / cluster_analysis['customer_count'].sum() * 100).round(2)

print("Cluster Analysis:")
print(cluster_analysis)

# Identify high-value and low-value clusters
high_value_cluster = cluster_analysis['clv'].idxmax()
low_value_cluster = cluster_analysis['clv'].idxmin()

print("\nHigh-value Cluster:", high_value_cluster)
print("Low-value Cluster:", low_value_cluster)

# Detect Outliers
def detect_outliers(cluster_data, columns):
    outliers = pd.DataFrame()
    for column in columns:
        z_scores = np.abs(stats.zscore(cluster_data[column]))
        outliers[column] = cluster_data[column][z_scores > 3]
    return outliers.dropna(how='all')

# Analyze high-value customers
high_value_customers = customer_data[customer_data['cluster'] == high_value_cluster]
print("\nHigh-value Customer Characteristics:")
print(high_value_customers.describe())

print("\nPotential Outliers in High-value Cluster:")
high_value_outliers = detect_outliers(high_value_customers, features)
print(high_value_outliers)

# Analyze low-value customers
low_value_customers = customer_data[customer_data['cluster'] == low_value_cluster]
print("\nLow-value Customer Characteristics:")
print(low_value_customers.describe())

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=customer_data.reset_index(), x='total_product_revenue', y='order_count', hue='cluster', palette='viridis')
plt.title('Customer Clusters: Total Revenue vs Order Count')
plt.xlabel('Total Product Revenue')
plt.ylabel('Order Count')
plt.savefig('customer_clusters.png')
plt.close()

# Visualize feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(kmeans.cluster_centers_.mean(axis=0))
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Clustering')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importance.png')
plt.close()

# Analyze CLV distribution in low-value cluster
plt.figure(figsize=(10, 6))
sns.histplot(data=low_value_customers, x='clv', bins=50, kde=True)
plt.title('CLV Distribution in Low-value Cluster')
plt.xlabel('Customer Lifetime Value')
plt.savefig('low_value_clv_distribution.png')
plt.close()

# Print additional metrics for each cluster
for cluster in range(5):
    cluster_data = customer_data[customer_data['cluster'] == cluster]
    print(f"\nCluster {cluster} Additional Metrics:")
    print(f"Avg Freight Revenue: {cluster_data['total_freight_revenue'].mean():.2f}")
    print(f"Avg Volume: {cluster_data['total_volume'].mean():.2f}")
    print(f"Avg Purchase Frequency: {cluster_data['purchase_frequency'].mean():.4f}")
    print(f"Median CLV: {cluster_data['clv'].median():.2f}")

# Section 4: Product Analysis
# ----------------------------------------

# Calculate product performance metrics
product_performance = df.groupby('item_number').agg({
    'line_quantity': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_revenue': 'sum',
    'line_freight_cost': 'sum',
    'line_volume_cbm': 'sum',
    'order_number': 'nunique'
}).reset_index()

product_performance['product_profit'] = product_performance['line_product_revenue'] - product_performance['line_product_cost']
product_performance['freight_profit'] = product_performance['line_freight_revenue'] - product_performance['line_freight_cost']
product_performance['total_profit'] = product_performance['product_profit'] + product_performance['freight_profit']

product_performance['product_margin'] = np.where(product_performance['line_product_revenue'] != 0,
                                                 product_performance['product_profit'] / product_performance['line_product_revenue'] * 100,
                                                 0)
product_performance['freight_margin'] = np.where(product_performance['line_freight_revenue'] != 0,
                                                 product_performance['freight_profit'] / product_performance['line_freight_revenue'] * 100,
                                                 0)
product_performance['delivered_margin'] = np.where((product_performance['line_product_revenue'] + product_performance['line_freight_revenue']) != 0,
                                                   product_performance['total_profit'] / (product_performance['line_product_revenue'] + product_performance['line_freight_revenue']) * 100,
                                                   0)

product_performance['avg_price'] = np.where(product_performance['line_quantity'] != 0,
                                            product_performance['line_product_revenue'] / product_performance['line_quantity'],
                                            0)

# Identify top and bottom performing products
top_products = product_performance.nlargest(10, 'total_profit')
bottom_products = product_performance.nsmallest(10, 'total_profit')

print("Top 10 Performing Products:")
print(top_products[['item_number', 'line_quantity', 'line_product_revenue', 'total_profit', 'delivered_margin']])

print("\nBottom 10 Performing Products:")
print(bottom_products[['item_number', 'line_quantity', 'line_product_revenue', 'total_profit', 'delivered_margin']])

# Analyze product mix impact on profitability
product_performance['profit_contribution'] = product_performance['total_profit'] / product_performance['total_profit'].sum() * 100
product_performance['revenue_contribution'] = (product_performance['line_product_revenue'] + product_performance['line_freight_revenue']) / (product_performance['line_product_revenue'] + product_performance['line_freight_revenue']).sum() * 100

plt.figure(figsize=(12, 6))
plt.scatter(product_performance['revenue_contribution'], product_performance['profit_contribution'])
plt.xlabel('Revenue Contribution (%)')
plt.ylabel('Profit Contribution (%)')
plt.title('Revenue vs Profit Contribution by Product')
plt.savefig('revenue_vs_profit_contribution.png')
plt.close()

# Sum of delivery margins for Category 1, sorted in ascending order
sum_margin_category_1 = df.groupby('category_1')['delivered_margin'].sum().sort_values()

# Sum of delivery margins for Category 2, sorted in ascending order
sum_margin_category_2 = df.groupby('category_2')['delivered_margin'].sum().sort_values()

# Sum of delivery margins for Category 3, sorted in ascending order
sum_margin_category_3 = df.groupby('category_3')['delivered_margin'].sum().sort_values()

# Display the results
print("Sum of delivery margins for Category 1 (ascending order):")
print(sum_margin_category_1)

print("\nSum of delivery margins for Category 2 (ascending order):")
print(sum_margin_category_2)

print("\nSum of delivery margins for Category 3 (ascending order):")
print(sum_margin_category_3)


# Pricing Strategy Analysis
def calculate_price_elasticity(df, product):
    product_data = df[df['item_number'] == product].copy()
    product_data['price'] = np.where(product_data['line_quantity'] != 0,
                                     product_data['line_product_revenue'] / product_data['line_quantity'],
                                     0)
    product_data = product_data.groupby('price').agg({'line_quantity': 'sum'}).reset_index()
    product_data = product_data.sort_values('price')
    
    price_pct_change = product_data['price'].pct_change()
    quantity_pct_change = product_data['line_quantity'].pct_change()
    
    valid_changes = (price_pct_change != 0) & (quantity_pct_change != 0) & (~np.isinf(price_pct_change)) & (~np.isinf(quantity_pct_change))
    elasticity = -quantity_pct_change[valid_changes] / price_pct_change[valid_changes]
    
    return elasticity.median() if len(elasticity) > 0 else np.nan

top_products_by_revenue = product_performance.nlargest(10, 'line_product_revenue')
elasticities = []

for product in top_products_by_revenue['item_number']:
    elasticity = calculate_price_elasticity(df, product)
    elasticities.append({'item_number': product, 'price_elasticity': elasticity})

elasticities_df = pd.DataFrame(elasticities)
print("\nPrice Elasticity for Top 10 Products by Revenue:")
print(elasticities_df)

for product in df['item_number'].unique():
    elasticity = calculate_price_elasticity(df, product)
    elasticities.append({'item_number': product, 'price_elasticity': elasticity})

elasticities_df = pd.DataFrame(elasticities)
elasticities_df.dropna(subset=['price_elasticity'], inplace=True)

# Adding delivered margin to elasticities_df
delivered_margins = df.groupby('item_number').agg({'delivered_margin': 'mean'}).reset_index()
elasticities_df = elasticities_df.merge(delivered_margins, on='item_number')

# Printing top 10 and bottom 10 price elasticities by delivered margin
top_10_by_margin = elasticities_df.nlargest(10, 'delivered_margin')
bottom_10_by_margin = elasticities_df.nsmallest(10, 'delivered_margin')

print("\nTop 10 Price Elasticities by Delivered Margin:")
print(top_10_by_margin)

print("\nBottom 10 Price Elasticities by Delivered Margin:")
print(bottom_10_by_margin)

def calculate_freight_revenue_elasticity(df, product):
    product_data = df[df['item_number'] == product].copy()
    product_data['freight_revenue'] = np.where(product_data['line_quantity'] != 0,
                                               product_data['line_freight_revenue'] / product_data['line_quantity'],
                                               0)
    product_data = product_data.groupby('freight_revenue').agg({'line_quantity': 'sum'}).reset_index()
    product_data = product_data.sort_values('freight_revenue')
    
    revenue_pct_change = product_data['freight_revenue'].pct_change()
    quantity_pct_change = product_data['line_quantity'].pct_change()
    
    valid_changes = (revenue_pct_change != 0) & (quantity_pct_change != 0) & (~np.isinf(revenue_pct_change)) & (~np.isinf(quantity_pct_change))
    elasticity = -quantity_pct_change[valid_changes] / revenue_pct_change[valid_changes]
    
    return elasticity.median() if len(elasticity) > 0 else np.nan


# Calculate freight revenue elasticity for each product
freight_elasticities = []
for product in df['item_number'].unique():
    elasticity = calculate_freight_revenue_elasticity(df, product)
    freight_elasticities.append({'item_number': product, 'freight_revenue_elasticity': elasticity})

freight_elasticities_df = pd.DataFrame(freight_elasticities)
freight_elasticities_df.dropna(subset=['freight_revenue_elasticity'], inplace=True)

# Adding delivered margin to elasticities_df
delivered_margins = df.groupby('item_number').agg({'delivered_margin': 'mean'}).reset_index()
freight_elasticities_df = freight_elasticities_df.merge(delivered_margins, on='item_number')

# Sorting elasticities by delivered margin in ascending order
sorted_freight_elasticities_df = freight_elasticities_df.sort_values('delivered_margin', ascending=True)
bottom_10_freight_by_margin = freight_elasticities_df.nsmallest(10, 'delivered_margin')
print("\nBottom 10 Freight Price Elasticities by Delivered Margin:")
print(bottom_10_freight_by_margin)

# Analyze relationship between price and sales volume
plt.figure(figsize=(12, 6))
sns.scatterplot(data=product_performance, x='avg_price', y='line_quantity')
plt.xlabel('Average Price')
plt.ylabel('Sales Volume')
plt.title('Price vs Sales Volume')
plt.savefig('price_vs_sales_volume.png')
plt.close()

# Dynamic pricing analysis
def calculate_price_variance(group):
    prices = group['line_product_revenue'] / group['line_quantity']
    return prices.var() / prices.mean() if len(prices) > 1 and prices.mean() != 0 else 0

product_performance['price_variance'] = df.groupby('item_number').apply(calculate_price_variance)
product_performance['volume_price_score'] = product_performance['price_variance'] * product_performance['line_quantity']

dynamic_pricing_opportunities = product_performance[np.isfinite(product_performance['volume_price_score'])].nlargest(10, 'volume_price_score')

print("\nTop 10 Products with Highest Volume-Adjusted Price Variance (Potential for Dynamic Pricing):")
print(dynamic_pricing_opportunities[['item_number', 'avg_price', 'price_variance', 'line_quantity', 'volume_price_score', 'total_profit']])

# Time-based analysis
df['month'] = df['order_date'].dt.to_period('M')
monthly_product_performance = df.groupby(['month', 'item_number']).agg({
    'line_quantity': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_revenue': 'sum',
    'line_freight_cost': 'sum'
}).reset_index()

monthly_product_performance['total_profit'] = (monthly_product_performance['line_product_revenue'] - monthly_product_performance['line_product_cost']) + \
                                              (monthly_product_performance['line_freight_revenue'] - monthly_product_performance['line_freight_cost'])
monthly_product_performance['delivered_margin'] = monthly_product_performance['total_profit'] / (monthly_product_performance['line_product_revenue'] + monthly_product_performance['line_freight_revenue']) * 100

# Plot monthly trends for top 5 products
top_5_products = product_performance.nlargest(5, 'total_profit')['item_number']

plt.figure(figsize=(15, 10))
for product in top_5_products:
    product_data = monthly_product_performance[monthly_product_performance['item_number'] == product]
    plt.plot(product_data['month'].astype(str), product_data['total_profit'], label=product)

plt.xlabel('Month')
plt.ylabel('Total Profit')
plt.title('Monthly Total Profit Trends for Top 5 Products')
plt.legend(title='Product')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_profit_trends.png')
plt.close()

# Seasonal analysis adjusted for Northern Europe
df['season'] = df['order_date'].dt.month.map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                              3: 'Spring', 4: 'Spring', 5: 'Spring',
                                              6: 'Summer', 7: 'Summer', 8: 'Summer',
                                              9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})

seasonal_performance = df.groupby('season').agg({
    'line_quantity': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_revenue': 'sum',
    'line_freight_cost': 'sum'
}).reset_index()

seasonal_performance['total_revenue'] = seasonal_performance['line_product_revenue'] + seasonal_performance['line_freight_revenue']
seasonal_performance['total_cost'] = seasonal_performance['line_product_cost'] + seasonal_performance['line_freight_cost']
seasonal_performance['total_profit'] = seasonal_performance['total_revenue'] - seasonal_performance['total_cost']
seasonal_performance['delivered_margin'] = seasonal_performance['total_profit'] / seasonal_performance['total_revenue'] * 100

print("\nSeasonal Performance (Adjusted for Northern Europe):")
print(seasonal_performance)

plt.figure(figsize=(12, 6))
sns.barplot(x='season', y='total_profit', data=seasonal_performance)
plt.title('Seasonal Total Profit (Northern Europe)')
plt.ylabel('Total Profit')
plt.savefig('seasonal_profit_northern_europe.png')
plt.close()

# Analyze top products by season
top_products_by_season = df.groupby(['season', 'item_number']).agg({
    'line_quantity': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_revenue': 'sum',
    'line_freight_cost': 'sum'
}).reset_index()

top_products_by_season['total_profit'] = (top_products_by_season['line_product_revenue'] - top_products_by_season['line_product_cost']) + \
                                         (top_products_by_season['line_freight_revenue'] - top_products_by_season['line_freight_cost'])
top_products_by_season['delivered_margin'] = top_products_by_season['total_profit'] / (top_products_by_season['line_product_revenue'] + top_products_by_season['line_freight_revenue']) * 100
top_products_by_season = top_products_by_season.sort_values(['season', 'total_profit'], ascending=[True, False])

print("\nTop 3 Products by Season:")
for season in top_products_by_season['season'].unique():
    print(f"\n{season}:")
    print(top_products_by_season[top_products_by_season['season'] == season].head(3)[['item_number', 'line_quantity', 'total_profit', 'delivered_margin']])

print("\nVisualization images have been saved.")

# Section 5: Channel and Agreement Analysis
# ----------------------------------------

# Overall Channel Performance
channel_performance = df.groupby('order_channel').agg({
    'order_number': 'count',
    'line_product_revenue': 'sum',
    'line_freight_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_cost': 'sum',
    'product_margin': 'mean',
    'freight_margin': 'mean',
    'delivered_margin': 'mean',
    'customer_name': 'nunique'
}).reset_index()

channel_performance['total_revenue'] = channel_performance['line_product_revenue'] + channel_performance['line_freight_revenue']
channel_performance['total_profit'] = (channel_performance['line_product_revenue'] - channel_performance['line_product_cost']) + \
                                      (channel_performance['line_freight_revenue'] - channel_performance['line_freight_cost'])
channel_performance['avg_order_value'] = channel_performance['total_revenue'] / channel_performance['order_number']

print("Overall Channel Performance:")
print(channel_performance)

# Visualize channel performance
plt.figure(figsize=(12, 6))
sns.barplot(x='order_channel', y='total_revenue', data=channel_performance)
plt.title('Total Revenue by Order Channel')
plt.savefig('channel_revenue.png')
plt.close()

# Customer Group Analysis
df['order_value'] = df['line_product_revenue'] + df['line_freight_revenue']
df['customer_total_value'] = df.groupby('customer_name')['order_value'].transform('sum')
df['customer_group'] = pd.qcut(df['customer_total_value'], q=3, labels=['Low Value', 'Medium Value', 'High Value'])

customer_channel_performance = df.groupby(['customer_group', 'order_channel']).agg({
    'order_number': 'count',
    'order_value': 'mean',
    'delivered_margin': 'mean'
}).reset_index()

print("\nCustomer Group Performance by Channel:")
print(customer_channel_performance)

# Visualize customer group performance by channel
plt.figure(figsize=(12, 6))
sns.barplot(x='order_channel', y='order_value', hue='customer_group', data=customer_channel_performance)
plt.title('Average Order Value by Channel and Customer Group')
plt.savefig('channel_customer_group_performance.png')
plt.close()

# Regional Analysis
regional_channel_performance = df.groupby(['order_region', 'order_channel']).agg({
    'order_number': 'count',
    'order_value': 'mean',
    'delivered_margin': 'mean'
}).reset_index()

print("\nTop 5 Regions Performance by Channel:")
print(regional_channel_performance.sort_values('order_number', ascending=False).head())

# Visualize top 5 regions by order count
top_5_regions = regional_channel_performance.groupby('order_region')['order_number'].sum().nlargest(5).index
plt.figure(figsize=(12, 6))
sns.barplot(x='order_region', y='order_number', hue='order_channel', 
            data=regional_channel_performance[regional_channel_performance['order_region'].isin(top_5_regions)])
plt.title('Order Count by Channel for Top 5 Regions')
plt.xticks(rotation=45)
plt.savefig('top_5_regions_channel_performance.png')
plt.close()

# Product Category Analysis
category_channel_performance = df.groupby(['category_1', 'order_channel']).agg({
    'order_number': 'count',
    'order_value': 'mean',
    'product_margin': 'mean',
    'freight_margin': 'mean',
    'delivered_margin': 'mean'
}).reset_index()

print("\nProduct Category Performance by Channel:")
print(category_channel_performance)

# Visualize category performance by channel
plt.figure(figsize=(12, 6))
sns.barplot(x='category_1', y='delivered_margin', hue='order_channel', data=category_channel_performance)
plt.title('Delivered Margin by Product Category and Channel')
plt.xticks(rotation=45)
plt.savefig('category_channel_performance.png')
plt.close()

# Time-based Analysis
df['month'] = df['order_date'].dt.to_period('M')
monthly_channel_performance = df.groupby(['month', 'order_channel']).agg({
    'order_number': 'count',
    'order_value': 'mean',
    'delivered_margin': 'mean'
}).reset_index()

print("\nMonthly Channel Performance (Last 3 months):")
print(monthly_channel_performance.sort_values('month', ascending=False).head(12))

# Visualize monthly trends
plt.figure(figsize=(15, 6))
for channel in df['order_channel'].unique():
    channel_data = monthly_channel_performance[monthly_channel_performance['order_channel'] == channel]
    plt.plot(channel_data['month'].astype(str), channel_data['order_number'], label=channel)
plt.title('Monthly Order Count by Channel')
plt.xlabel('Month')
plt.ylabel('Order Count')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('monthly_channel_trends.png')
plt.close()

# Channel Efficiency Analysis
channel_efficiency = df.groupby('order_channel').agg({
    'order_number': 'count',
    'line_quantity': 'mean',
    'line_volume_cbm': 'mean',
    'delivered_margin': 'mean'
}).reset_index()

channel_efficiency['orders_per_volume'] = channel_efficiency['order_number'] / channel_efficiency['line_volume_cbm']

print("\nChannel Efficiency:")
print(channel_efficiency)

# Visualize channel efficiency
plt.figure(figsize=(12, 6))
sns.scatterplot(x='orders_per_volume', y='delivered_margin', hue='order_channel', size='order_number', data=channel_efficiency)
plt.title('Channel Efficiency: Orders per Volume vs Delivered Margin')
plt.savefig('channel_efficiency.png')
plt.close()

print("\nVisualization images have been saved.")

# Identify Areas for Improvement
print("\nPotential Areas for Improvement:")
for channel in channel_performance['order_channel']:
    channel_data = channel_performance[channel_performance['order_channel'] == channel].iloc[0]
    if channel_data['delivered_margin'] < channel_performance['delivered_margin'].mean():
        print(f"- {channel}: Below average delivered margin ({channel_data['delivered_margin']:.2f}%). "
              f"Consider optimizing costs or pricing strategy.")
    if channel_data['avg_order_value'] < channel_performance['avg_order_value'].mean():
        print(f"- {channel}: Below average order value (${channel_data['avg_order_value']:.2f}). "
              f"Consider upselling or cross-selling strategies.")

# Statistical Tests
def perform_anova(df, group_col, value_col):
    groups = [group for _, group in df.groupby(group_col)[value_col]]
    f_value, p_value = stats.f_oneway(*groups)
    return f_value, p_value

f_value, p_value = perform_anova(df, 'order_channel', 'delivered_margin')
print(f"\nANOVA Test for Delivered Margin across Channels:")
print(f"F-value: {f_value}, p-value: {p_value}")
if p_value < 0.05:
    print("There is a statistically significant difference in delivered margins across channels.")
else:
    print("There is no statistically significant difference in delivered margins across channels.")

# Agreement Analysis
agreement_analysis = df.groupby('customer_agreement').agg({
    'customer_name': 'nunique',
    'order_number': 'count',
    'total_revenue': 'sum',
    'product_margin': 'mean',
    'freight_margin': 'mean',
    'delivered_margin': 'mean'
}).reset_index()

agreement_analysis['avg_order_value'] = agreement_analysis['total_revenue'] / agreement_analysis['order_number']
agreement_analysis['revenue_per_customer'] = agreement_analysis['total_revenue'] / agreement_analysis['customer_name']

print("Overall Agreement Analysis:")
print(agreement_analysis.sort_values('total_revenue', ascending=False))

# Visualize agreement performance
plt.figure(figsize=(12, 6))
sns.scatterplot(x='revenue_per_customer', y='delivered_margin', 
                size='customer_name', hue='customer_name', 
                data=agreement_analysis)
plt.title('Agreement Performance: Revenue per Customer vs Delivered Margin')
plt.xlabel('Revenue per Customer')
plt.ylabel('Delivered Margin (%)')
plt.savefig('agreement_performance.png')
plt.close()

# Pricing Analysis by Agreement
pricing_analysis = df.groupby('customer_agreement').agg({
    'line_product_revenue': 'sum',
    'line_quantity': 'sum'
}).reset_index()

pricing_analysis['avg_price'] = pricing_analysis['line_product_revenue'] / pricing_analysis['line_quantity']

print("\nPricing Analysis by Agreement:")
print(pricing_analysis.sort_values('avg_price', ascending=False))

# Visualize pricing by agreement
plt.figure(figsize=(12, 6))
sns.barplot(x='customer_agreement', y='avg_price', data=pricing_analysis)
plt.title('Average Price by Customer Agreement')
plt.xticks(rotation=45)
plt.savefig('agreement_pricing.png')
plt.close()

# Profitability Analysis by Agreement
profitability_analysis = agreement_analysis[['customer_agreement', 'total_revenue', 'delivered_margin']]
profitability_analysis['profit'] = profitability_analysis['total_revenue'] * profitability_analysis['delivered_margin'] / 100

print("\nProfitability Analysis by Agreement:")
print(profitability_analysis.sort_values('profit', ascending=False))

# Visualize profitability by agreement
plt.figure(figsize=(12, 6))
sns.scatterplot(x='total_revenue', y='delivered_margin', 
                size='profit', hue='customer_agreement', 
                data=profitability_analysis)
plt.title('Agreement Profitability: Revenue vs Margin')
plt.xlabel('Total Revenue')
plt.ylabel('Delivered Margin (%)')
plt.savefig('agreement_profitability.png')
plt.close()

# Time-based Analysis of Agreement Performance
df['month'] = df['order_date'].dt.to_period('M')
monthly_agreement_performance = df.groupby(['month', 'customer_agreement']).agg({
    'total_revenue': 'sum',
    'delivered_margin': 'mean'
}).reset_index()

print("\nMonthly Agreement Performance (Last 3 months):")
print(monthly_agreement_performance.sort_values('month', ascending=False).head(30))

# Visualize monthly trends for top 5 agreements by revenue
top_5_agreements = agreement_analysis.nlargest(5, 'total_revenue')['customer_agreement']
plt.figure(figsize=(15, 6))
for agreement in top_5_agreements:
    agreement_data = monthly_agreement_performance[monthly_agreement_performance['customer_agreement'] == agreement]
    plt.plot(agreement_data['month'].astype(str), agreement_data['delivered_margin'], label=agreement)
plt.title('Monthly Delivered Margin Trends for Top 5 Agreements')
plt.xlabel('Month')
plt.ylabel('Delivered Margin (%)')
plt.legend(title='Customer Agreement')
plt.xticks(rotation=45)
plt.savefig('monthly_agreement_trends.png')
plt.close()

# Identify Agreements for Potential Renegotiation
renegotiation_candidates = agreement_analysis.copy()
renegotiation_candidates['revenue_rank'] = renegotiation_candidates['total_revenue'].rank(ascending=False)
renegotiation_candidates['margin_rank'] = renegotiation_candidates['delivered_margin'].rank(ascending=False)
renegotiation_candidates['renegotiation_score'] = renegotiation_candidates['revenue_rank'] - renegotiation_candidates['margin_rank']

print("\nTop Agreements for Potential Renegotiation (High Revenue, Low Margin):")
print(renegotiation_candidates.nlargest(10, 'renegotiation_score'))

# Statistical Analysis of Agreement Impact
f_value, p_value = perform_anova(df, 'customer_agreement', 'delivered_margin')
print(f"\nANOVA Test for Delivered Margin across Agreements:")
print(f"F-value: {f_value}, p-value: {p_value}")
if p_value < 0.05:
    print("There is a statistically significant difference in delivered margins across agreements.")
else:
    print("There is no statistically significant difference in delivered margins across agreements.")

# Optimization Opportunities
print("\nOptimization Opportunities:")

# Agreements with low freight margins
low_freight_margin_agreements = agreement_analysis[agreement_analysis['freight_margin'] < agreement_analysis['freight_margin'].median()]
print("\nAgreements with Low Freight Margins (Potential for logistics optimization):")
print(low_freight_margin_agreements[['customer_agreement', 'freight_margin', 'total_revenue']])

# Agreements with high variation in pricing
pricing_variation = df.groupby('customer_agreement').agg({
    'avg_price': ['mean', 'std']
}).reset_index()
pricing_variation.columns = ['customer_agreement', 'mean_price', 'price_std']
pricing_variation['coefficient_of_variation'] = pricing_variation['price_std'] / pricing_variation['mean_price']

print("\nAgreements with High Pricing Variation (Potential for price standardization):")
print(pricing_variation.nlargest(10, 'coefficient_of_variation'))

print("\nVisualization images have been saved.")

# Section 6: Freight and Regional Analysis
# ----------------------------------------

# Analyze freight costs in relation to order volume and distance
order_summary = df.groupby('order_number').agg({
    'line_volume_cbm': 'sum',
    'line_freight_cost': 'sum',
    'line_freight_revenue': 'sum',
    'order_region': 'first'
}).reset_index()

order_summary['freight_cost_per_cbm'] = np.where(
    order_summary['line_volume_cbm'] > 0,
    order_summary['line_freight_cost'] / order_summary['line_volume_cbm'],
    0
)

print("Freight Cost Analysis:")
print(order_summary.describe())

# Visualize relationship between order volume and freight cost
plt.figure(figsize=(12, 6))
sns.scatterplot(x='line_volume_cbm', y='line_freight_cost', data=order_summary)
plt.title('Order Volume vs Freight Cost')
plt.xlabel('Order Volume (CBM)')
plt.ylabel('Freight Cost')
plt.savefig('volume_vs_freight_cost.png')
plt.close()

# Analyze freight cost by region
region_freight_analysis = order_summary.groupby('order_region').agg({
    'line_volume_cbm': 'mean',
    'line_freight_cost': 'mean',
    'freight_cost_per_cbm': 'mean'
}).reset_index().sort_values('freight_cost_per_cbm', ascending=False)

print("\nFreight Cost by Region (Top 5):")
print(region_freight_analysis.head())

# Visualize freight cost per CBM by region
plt.figure(figsize=(12, 6))
sns.barplot(x='order_region', y='freight_cost_per_cbm', data=region_freight_analysis.head(10))
plt.title('Freight Cost per CBM by Region (Top 10)')
plt.xticks(rotation=45)
plt.savefig('freight_cost_by_region.png')
plt.close()

# Optimize shipping routes and methods
region_freight_analysis = region_freight_analysis.replace([np.inf, -np.inf], np.nan).dropna()

# Cluster regions based on freight costs and volumes
X = region_freight_analysis[['line_volume_cbm', 'freight_cost_per_cbm']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
region_freight_analysis['cluster'] = kmeans.fit_predict(X_scaled)

print("\nRegion Clusters for Shipping Optimization:")
print(region_freight_analysis.groupby('cluster').agg({
    'order_region': 'count',
    'line_volume_cbm': 'mean',
    'freight_cost_per_cbm': 'mean'
}))

# Visualize clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='line_volume_cbm', y='freight_cost_per_cbm', hue='cluster', data=region_freight_analysis)
plt.title('Region Clusters based on Volume and Freight Cost')
plt.xlabel('Average Order Volume (CBM)')
plt.ylabel('Freight Cost per CBM')
plt.savefig('region_clusters.png')
plt.close()

# Impact of order consolidation on freight costs
df['order_date_rounded'] = df['order_date'].dt.floor('D')
consolidated_orders = df.groupby(['order_date_rounded', 'order_region']).agg({
    'order_number': 'count',
    'line_volume_cbm': 'sum',
    'line_freight_cost': 'sum'
}).reset_index()

consolidated_orders['freight_cost_per_cbm'] = np.where(
    consolidated_orders['line_volume_cbm'] > 0,
    consolidated_orders['line_freight_cost'] / consolidated_orders['line_volume_cbm'],
    0
)

print("\nNon-consolidated vs Consolidated Freight Costs:")
print("Non-consolidated average freight cost per CBM:", order_summary['freight_cost_per_cbm'].mean())
print("Consolidated average freight cost per CBM:", consolidated_orders['freight_cost_per_cbm'].mean())
print("Potential savings (%):", 
      (1 - consolidated_orders['freight_cost_per_cbm'].mean() / order_summary['freight_cost_per_cbm'].mean()) * 100)

# Visualize the impact of consolidation
plt.figure(figsize=(12, 6))
sns.histplot(order_summary['freight_cost_per_cbm'], kde=True, label='Non-consolidated')
sns.histplot(consolidated_orders['freight_cost_per_cbm'], kde=True, label='Consolidated')
plt.title('Distribution of Freight Cost per CBM: Non-consolidated vs Consolidated')
plt.xlabel('Freight Cost per CBM')
plt.legend()
plt.savefig('consolidation_impact.png')
plt.close()

# Analyze consolidation impact by region
consolidation_impact = consolidated_orders.groupby('order_region').agg({
    'freight_cost_per_cbm': 'mean',
    'order_number': 'mean'
}).reset_index()

consolidation_impact = consolidation_impact.merge(
    region_freight_analysis[['order_region', 'freight_cost_per_cbm']], 
    on='order_region', 
    suffixes=('_consolidated', '_original')
)

consolidation_impact['savings_percentage'] = np.where(
    consolidation_impact['freight_cost_per_cbm_original'] > 0,
    (1 - consolidation_impact['freight_cost_per_cbm_consolidated'] / 
     consolidation_impact['freight_cost_per_cbm_original']) * 100,
    0
)

print("\nTop 5 Regions with Highest Potential Savings from Consolidation:")
print(consolidation_impact.sort_values('savings_percentage', ascending=False).head())

# Visualize consolidation impact by region
plt.figure(figsize=(12, 6))
sns.scatterplot(x='order_number', y='savings_percentage', data=consolidation_impact)
plt.title('Consolidation Savings vs Average Daily Orders by Region')
plt.xlabel('Average Daily Orders')
plt.ylabel('Potential Savings (%)')
for i, row in consolidation_impact.iterrows():
    plt.annotate(row['order_region'], (row['order_number'], row['savings_percentage']))
plt.savefig('consolidation_savings_by_region.png')
plt.close()

print("\nVisualization images have been saved.")

# Section 7: Predictive Analysis and Linear Models
# ----------------------------------------

# Correlation Analysis
numeric_columns = ['line_volume_cbm', 'line_quantity', 'line_product_revenue', 'line_product_cost', 
                   'line_freight_revenue', 'line_freight_cost', 'total_revenue', 
                   'product_margin', 'freight_margin', 'delivered_margin']

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Key Metrics')
plt.savefig('correlation_matrix.png')
plt.close()

print("Correlation Matrix:")
print(correlation_matrix)

# Scatter Plot Matrix
sns.pairplot(df[numeric_columns], height=2, aspect=1.5)
plt.tight_layout()
plt.savefig('scatter_plot_matrix.png')
plt.close()

# Linear Regression: Volume vs. Freight Cost
X = df[['line_volume_cbm']]
y = df['line_freight_cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print("\nLinear Regression: Volume vs. Freight Cost")
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Volume vs. Freight Cost: Linear Regression')
plt.xlabel('Volume (CBM)')
plt.ylabel('Freight Cost')
plt.savefig('volume_vs_freight_cost_regression.png')
plt.close()

# Multiple Linear Regression: Predicting Delivered Margin
features = ['line_volume_cbm', 'line_quantity', 'line_product_revenue', 'line_product_cost', 
            'line_freight_revenue', 'line_freight_cost']
target = 'delivered_margin'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

y_pred = mlr_model.predict(X_test)

print("\nMultiple Linear Regression: Predicting Delivered Margin")
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

print("\nFeature Coefficients:")
for feature, coef in zip(features, mlr_model.coef_):
    print(f"{feature}: {coef}")

# Random Forest Regression: Predicting Total Revenue
features = ['line_volume_cbm', 'line_quantity', 'line_product_cost', 'line_freight_cost', 
            'order_channel', 'order_region', 'category_1']
target = 'total_revenue'

# Encode categorical variables
le = LabelEncoder()
for feature in ['order_channel', 'order_region', 'category_1']:
    df[feature] = le.fit_transform(df[feature].astype(str))

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("\nRandom Forest Regression: Predicting Total Revenue")
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Predicting Total Revenue')
plt.savefig('feature_importance_total_revenue.png')
plt.close()

# Time Series Analysis: Monthly Revenue Trend
df['month'] = df['order_date'].dt.to_period('M')
monthly_revenue = df.groupby('month')['total_revenue'].sum().reset_index()
monthly_revenue['month'] = monthly_revenue['month'].astype(str)

plt.figure(figsize=(12, 6))
plt.plot(monthly_revenue['month'], monthly_revenue['total_revenue'])
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_revenue_trend.png')
plt.close()

# Predictive Model: Next Month's Revenue
monthly_revenue['prev_month_revenue'] = monthly_revenue['total_revenue'].shift(1)
monthly_revenue = monthly_revenue.dropna()

X = monthly_revenue[['prev_month_revenue']]
y = monthly_revenue['total_revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print("\nPredictive Model: Next Month's Revenue")
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title("Next Month's Revenue Prediction")
plt.xlabel('Previous Month Revenue')
plt.ylabel('Next Month Revenue')
plt.legend()
plt.savefig('next_month_revenue_prediction.png')
plt.close()

print("\nVisualization images have been saved.")

# Section 8: Inventory and Profitability Analysis
# ----------------------------------------
df['total_profit'] = (df['line_product_revenue'] - df['line_product_cost']) + \
                     (df['line_freight_revenue'] - df['line_freight_cost'])

# Analyze sales patterns
df['month'] = df['order_date'].dt.to_period('M')
monthly_sales = df.groupby(['month', 'item_number']).agg({
    'line_quantity': 'sum',
    'total_profit': 'sum'
}).reset_index()

# Calculate average monthly sales and coefficient of variation
item_sales_stats = monthly_sales.groupby('item_number').agg({
    'line_quantity': ['mean', 'std', 'sum'],
    'total_profit': 'sum'
}).reset_index()
item_sales_stats.columns = ['item_number', 'avg_monthly_sales', 'std_monthly_sales', 'total_sales', 'total_profit']
item_sales_stats['cv_sales'] = item_sales_stats['std_monthly_sales'] / item_sales_stats['avg_monthly_sales']

# Identify slow-moving inventory
slow_moving_threshold = item_sales_stats['avg_monthly_sales'].quantile(0.25)
slow_moving_inventory = item_sales_stats[item_sales_stats['avg_monthly_sales'] < slow_moving_threshold]

print("Slow-moving Inventory:")
print(slow_moving_inventory.sort_values('avg_monthly_sales').head())

# Visualize sales patterns
plt.figure(figsize=(12, 6))
sns.scatterplot(x='avg_monthly_sales', y='cv_sales', size='total_profit', 
                hue='total_profit', data=item_sales_stats)
plt.title('Sales Pattern Analysis')
plt.xlabel('Average Monthly Sales')
plt.ylabel('Coefficient of Variation of Sales')
plt.savefig('sales_pattern_analysis.png')
plt.close()

# Profitability Analysis
# Product-level profitability
product_profitability = df.groupby('item_number').agg({
    'line_quantity': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'total_profit': 'sum'
}).reset_index()
product_profitability['profit_margin'] = product_profitability['total_profit'] / product_profitability['line_product_revenue'] * 100

print("\nTop 10 Most Profitable Products:")
print(product_profitability.nlargest(10, 'total_profit'))

print("\nTop 10 Least Profitable Products:")
print(product_profitability.nsmallest(10, 'total_profit'))

# Category-level profitability
category_profitability = df.groupby('category_1').agg({
    'line_quantity': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'total_profit': 'sum'
}).reset_index()
category_profitability['profit_margin'] = category_profitability['total_profit'] / category_profitability['line_product_revenue'] * 100

print("\nCategory Profitability:")
print(category_profitability.sort_values('profit_margin', ascending=False))

# Customer-level profitability
customer_profitability = df.groupby('customer_name').agg({
    'order_number': 'count',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'total_profit': 'sum'
}).reset_index()
customer_profitability['profit_margin'] = customer_profitability['total_profit'] / customer_profitability['line_product_revenue'] * 100

print("\nTop 10 Most Profitable Customers:")
print(customer_profitability.nlargest(10, 'total_profit'))

# Region-level profitability
region_profitability = df.groupby('order_region').agg({
    'order_number': 'count',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'total_profit': 'sum'
}).reset_index()
region_profitability['profit_margin'] = region_profitability['total_profit'] / region_profitability['line_product_revenue'] * 100

print("\nRegion Profitability:")
print(region_profitability.sort_values('profit_margin', ascending=False))

# Identify factors contributing to profitability
# Prepare data for clustering
profitability_factors = df.groupby('item_number').agg({
    'line_quantity': 'sum',
    'line_product_revenue': 'sum',
    'line_product_cost': 'sum',
    'line_freight_revenue': 'sum',
    'line_freight_cost': 'sum',
    'total_profit': 'sum',
    'profit_margin': 'mean'
}).reset_index()

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(profitability_factors.drop('item_number', axis=1))

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
profitability_factors['cluster'] = kmeans.fit_predict(normalized_data)

# Analyze clusters
cluster_analysis = profitability_factors.groupby('cluster').agg({
    'line_quantity': 'mean',
    'line_product_revenue': 'mean',
    'line_product_cost': 'mean',
    'line_freight_revenue': 'mean',
    'line_freight_cost': 'mean',
    'total_profit': 'mean',
    'profit_margin': 'mean'
}).reset_index()

print("\nProfitability Cluster Analysis:")
print(cluster_analysis)

# Visualize profitability clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='total_profit', y='profit_margin', hue='cluster', 
                size='line_quantity', data=profitability_factors)
plt.title('Profitability Clusters')
plt.xlabel('Total Profit')
plt.ylabel('Profit Margin (%)')
plt.savefig('profitability_clusters.png')
plt.close()

# Inventory Optimization Recommendations
# Identify overstocked items (high inventory, low sales)
overstocked_items = item_sales_stats[
    (item_sales_stats['avg_monthly_sales'] < item_sales_stats['avg_monthly_sales'].median()) &
    (item_sales_stats['total_sales'] > item_sales_stats['total_sales'].quantile(0.75))
]

print("\nPotentially Overstocked Items:")
print(overstocked_items.head())

# Identify understocked items (low inventory, high sales)
understocked_items = item_sales_stats[
    (item_sales_stats['avg_monthly_sales'] > item_sales_stats['avg_monthly_sales'].median()) &
    (item_sales_stats['total_sales'] < item_sales_stats['total_sales'].quantile(0.25))
]

print("\nPotentially Understocked Items:")
print(understocked_items.head())

# Profitability Improvement Recommendations
# Identify low-margin, high-volume products
low_margin_high_volume = product_profitability[
    (product_profitability['profit_margin'] < product_profitability['profit_margin'].median()) &
    (product_profitability['line_quantity'] > product_profitability['line_quantity'].quantile(0.75))
]

print("\nLow-margin, High-volume Products (Consider price optimization):")
print(low_margin_high_volume.head())

# Identify high-margin, low-volume products
high_margin_low_volume = product_profitability[
    (product_profitability['profit_margin'] > product_profitability['profit_margin'].median()) &
    (product_profitability['line_quantity'] < product_profitability['line_quantity'].quantile(0.25))
]

print("\nHigh-margin, Low-volume Products (Consider promotion strategies):")
print(high_margin_low_volume.head())

print("\nVisualization images have been saved.")