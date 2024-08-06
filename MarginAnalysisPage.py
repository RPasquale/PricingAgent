import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from sklearn.feature_selection import f_regression

# Read the CSV file
df = pd.read_csv('G:\\My Drive\\marginwell\\marginwell_sample_lightbulb_output_dataset_1.csv')

# Convert order_date to datetime and calculate margins
df['order_date'] = pd.to_datetime(df['order_date'])
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

# Function to get monthly data for a category
def get_monthly_data(df, category_col, value_col, selected_subcategories):
    filtered_df = df[df[category_col].isin(selected_subcategories)]
    if filtered_df.empty:
        return pd.DataFrame()
    
    monthly_data = filtered_df.groupby([category_col, pd.Grouper(key='order_date', freq='ME')])[value_col].mean().reset_index()
    pivot_df = monthly_data.pivot(index='order_date', columns=category_col, values=value_col)
    return pivot_df

# Function to calculate margin swings
def calculate_margin_swings(df, group_col):
    df['total_revenue'] = df['line_product_revenue'] + df['line_freight_revenue']
    df['total_cost'] = df['line_product_cost'] + df['line_freight_cost']
    
    grouped = df.groupby([group_col, pd.Grouper(key='order_date', freq='ME')]).agg({
        'total_revenue': 'sum',
        'total_cost': 'sum'
    }).reset_index()
    
    grouped['margin'] = (grouped['total_revenue'] - grouped['total_cost']) / grouped['total_revenue'] * 100
    
    pivot = grouped.pivot(index='order_date', columns=group_col, values='margin')
    swings = pivot.max() - pivot.min()
    
    return swings.sort_values(ascending=False)

# Function to identify negative margins
def identify_negative_margins(df, group_col, margin_col, revenue_col, cost_col):
    grouped = df.groupby(group_col).agg({
        margin_col: 'mean',
        revenue_col: 'sum',
        cost_col: 'sum'
    }).reset_index()
    
    grouped['total_loss'] = np.where(
        grouped[margin_col] < 0,
        grouped[cost_col] - grouped[revenue_col],
        0
    )
    
    return grouped.sort_values('total_loss', ascending=False)

# Function to calculate feature importance
def calculate_feature_importance(df, feature_col):
    # One-hot encode categorical variables
    if df[feature_col].dtype == 'object':
        dummies = pd.get_dummies(df[feature_col], prefix=feature_col)
        X = dummies
    else:
        X = df[[feature_col]]
    
    y = df['delivered_margin']
    
    # Use f_regression for feature importance
    f_scores, _ = f_regression(X, y)
    
    # Create a dataframe with feature names and their importance scores
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': f_scores
    })
    
    return feature_importance.sort_values('importance', ascending=False)

# Get all unique categories and subcategories
all_categories = {
    'category_1': df['category_1'].unique(),
    'category_2': df['category_2'].unique(),
    'category_3': df['category_3'].unique(),
    'order_channel': df['order_channel'].unique(),
    'order_region': df['order_region'].unique(),
    'item_number': df['item_number'].unique(),
    'customer_name': df['customer_name'].unique(),
    'customer_agreement': df['customer_agreement'].unique()
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Function to create figures
def create_figure(data, title_prefix):
    fig = make_subplots(rows=1, cols=1, subplot_titles=(f"{title_prefix}",))
    
    for j, column in enumerate(data.columns):
        y_values = data[column].values
        y_values = np.where(np.isinf(y_values), np.nan, y_values)  # Replace inf with nan
        
        fig.add_trace(
            go.Scatter(x=data.index, y=y_values, name=f"{column}", 
                       line=dict(color=colors[j % len(colors)]),
                       connectgaps=False),  # Don't connect gaps (nan values)
            row=1, col=1
        )
    
    # Add average annotation
    avg = np.nanmean(data.values)
    fig.add_annotation(
        x=0.99, y=0.95, xref='paper', yref='paper',
        text=f'Average: {avg:.2f}%',
        showarrow=False, font=dict(size=12), align='right'
    )
    
    fig.update_layout(
        height=600, 
        title_text=f"{title_prefix}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=f"Monthly {title_prefix} (%)")
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

# Initialize the Dash app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Margin Analysis Dashboard"),
    html.Div([
        html.Div([
            html.Label('Select Category:'),
            dcc.Dropdown(
                id='category-dropdown',
                options=[
                    {'label': f'Category {i}', 'value': f'category_{i}'} for i in range(1, 4)
                ] + [
                    {'label': cat.replace('_', ' ').capitalize(), 'value': cat} for cat in ['order_channel', 'order_region', 'item_number', 'customer_name', 'customer_agreement']
                ],
                value='category_1'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Select Subcategories:'),
            dcc.Dropdown(id='subcategory-dropdown', multi=True)
        ], style={'width': '70%', 'display': 'inline-block'})
    ], style={'width': '100%', 'display': 'flex'}),
    dcc.Tabs([
        dcc.Tab(label='Product Margin', children=[
            dcc.Graph(id='product-margin-graph')
        ]),
        dcc.Tab(label='Freight Margin', children=[
            dcc.Graph(id='freight-margin-graph')
        ]),
        dcc.Tab(label='Delivered Margin', children=[
            dcc.Graph(id='delivered-margin-graph')
        ]),
        dcc.Tab(label='Margin Swings', children=[
            dcc.Graph(id='product-margin-swings-graph'),
            dcc.Graph(id='freight-margin-swings-graph'),
            dcc.Graph(id='delivered-margin-swings-graph')
        ]),
        dcc.Tab(label='Negative Margins', children=[
            dcc.Graph(id='product-negative-margins-graph'),
            dcc.Graph(id='freight-negative-margins-graph'),
            dcc.Graph(id='delivered-negative-margins-graph')
        ]),
        dcc.Tab(label='Statistics', children=[
            html.Div([
                html.Label('Select Feature for Analysis:'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[
                        {'label': col, 'value': col} for col in 
                        ['category_1', 'category_2', 'category_3', 'order_channel', 'order_region', 'item_number', 'customer_name', 'customer_agreement']
                    ],
                    value='category_1'
                )
            ]),
            dcc.Graph(id='margin-swings-graph'),
            dcc.Graph(id='feature-importance-graph')
        ]),
    ])
])

# Callback to update subcategory dropdown options
@app.callback(
    Output('subcategory-dropdown', 'options'),
    Input('category-dropdown', 'value')
)
def update_subcategory_options(selected_category):
    return [{'label': sub, 'value': sub} for sub in all_categories[selected_category]]

# Callbacks for Product, Freight, and Delivered Margin graphs
@app.callback(
    Output('product-margin-graph', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('subcategory-dropdown', 'value')]
)
def update_product_margin_graph(selected_category, selected_subcategories):
    return update_margin_graph(selected_category, selected_subcategories, 'product_margin')

@app.callback(
    Output('freight-margin-graph', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('subcategory-dropdown', 'value')]
)
def update_freight_margin_graph(selected_category, selected_subcategories):
    return update_margin_graph(selected_category, selected_subcategories, 'freight_margin')

@app.callback(
    Output('delivered-margin-graph', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('subcategory-dropdown', 'value')]
)
def update_delivered_margin_graph(selected_category, selected_subcategories):
    return update_margin_graph(selected_category, selected_subcategories, 'delivered_margin')

def update_margin_graph(selected_category, selected_subcategories, margin_type):
    if not selected_subcategories:
        return go.Figure().add_annotation(x=0.5, y=0.5, text="Please select subcategories", showarrow=False, font_size=20)
    
    try:
        margin_data = get_monthly_data(df, selected_category, margin_type, selected_subcategories)
        if margin_data.empty:
            return go.Figure()
        
        fig = create_figure(margin_data, f"{margin_type.replace('_', ' ').capitalize()}")
        return fig
    except Exception as e:
        print(f"Error in update_{margin_type}_graph: {str(e)}")
        return go.Figure()

# Callbacks for Margin Swings graphs
@app.callback(
    Output('product-margin-swings-graph', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_product_margin_swings_graph(selected_category):
    return create_margin_swings_graph(df, selected_category, 'product_margin')

@app.callback(
    Output('freight-margin-swings-graph', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_freight_margin_swings_graph(selected_category):
    return create_margin_swings_graph(df, selected_category, 'freight_margin')

@app.callback(
    Output('delivered-margin-swings-graph', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_delivered_margin_swings_graph(selected_category):
    return create_margin_swings_graph(df, selected_category, 'delivered_margin')

def create_margin_swings_graph(df, selected_category, margin_type):
    try:
        swings = calculate_margin_swings(df, selected_category)
        
        fig = go.Figure(go.Bar(
            x=swings.index,
            y=swings.values,
            marker_color=swings.values,
            marker_colorscale='RdYlGn_r'  # Red for largest swings, green for smallest
        ))
        
        fig.update_layout(
            title_text=f"{margin_type.replace('_', ' ').capitalize()} Swings for {selected_category.replace('_', ' ').capitalize()}",
            xaxis_title=selected_category.replace('_', ' ').capitalize(),
            yaxis_title="Margin Swing (%)",
            height=600
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_margin_swings_graph for {margin_type}: {str(e)}")
        return go.Figure()

# Callbacks for Negative Margins graphs
@app.callback(
    Output('product-negative-margins-graph', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_product_negative_margins_graph(selected_category):
    return create_negative_margins_graph(df, selected_category, 'product_margin', 'line_product_revenue', 'line_product_cost')

@app.callback(
    Output('freight-negative-margins-graph', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_freight_negative_margins_graph(selected_category):
    return create_negative_margins_graph(df, selected_category, 'freight_margin', 'line_freight_revenue', 'line_freight_cost')

@app.callback(
    Output('delivered-negative-margins-graph', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_delivered_negative_margins_graph(selected_category):
    df['total_revenue'] = df['line_product_revenue'] + df['line_freight_revenue']
    df['total_cost'] = df['line_product_cost'] + df['line_freight_cost']
    return create_negative_margins_graph(df, selected_category, 'delivered_margin', 'total_revenue', 'total_cost')

def create_negative_margins_graph(df, selected_category, margin_type, revenue_col, cost_col):
    try:
        negative_margins = identify_negative_margins(df, selected_category, margin_type, revenue_col, cost_col)
        
        fig = go.Figure([
            go.Bar(name=f'Average {margin_type.replace("_", " ").capitalize()} (%)', x=negative_margins[selected_category], y=negative_margins[margin_type]),
            go.Bar(name=f'Total {margin_type.replace("_", " ").capitalize()} Loss ($)', x=negative_margins[selected_category], y=negative_margins['total_loss'])
        ])
        
        fig.update_layout(
            title_text=f"Negative {margin_type.replace('_', ' ').capitalize()} for {selected_category.replace('_', ' ').capitalize()}",
            xaxis_title=selected_category.replace('_', ' ').capitalize(),
            yaxis_title=f"{margin_type.replace('_', ' ').capitalize()} (%) / Total Loss ($)",
            barmode='group',
            height=600
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_negative_margins_graph for {margin_type}: {str(e)}")
        return go.Figure()

# New callback for Margin Swings graph in Statistics tab
@app.callback(
    Output('margin-swings-graph', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_margin_swings_graph(selected_feature):
    swings = calculate_margin_swings(df, selected_feature)
    
    fig = go.Figure(go.Bar(
        x=swings.index,
        y=swings.values,
        marker_color=swings.values,
        marker_colorscale='RdYlGn_r'  # Red for largest swings, green for smallest
    ))
    
    fig.update_layout(
        title_text=f"Margin Swings for {selected_feature.replace('_', ' ').capitalize()}",
        xaxis_title=selected_feature.replace('_', ' ').capitalize(),
        yaxis_title="Margin Swing (%)",
        height=600
    )
    
    return fig

# New callback for Feature Importance graph in Statistics tab
@app.callback(
    Output('feature-importance-graph', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_feature_importance_graph(selected_feature):
    importance = calculate_feature_importance(df, selected_feature)
    
    fig = go.Figure(go.Bar(
        x=importance['feature'],
        y=importance['importance'],
        marker_color=importance['importance'],
        marker_colorscale='Viridis'
    ))
    
    fig.update_layout(
        title_text=f"Feature Importance for {selected_feature.replace('_', ' ').capitalize()}",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        height=600
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

print("Margin Analysis Dashboard with Statistics Page is running. Open your web browser and go to http://127.0.0.1:8050/ to view it.")
