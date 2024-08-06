'''import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import os
import plotly.express as px
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

# Directory where the CSV files are stored
output_dir = "category_analysis_results"
logger.info(f"Looking for files in: {os.path.abspath(output_dir)}")

# Get list of all CSV files
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
logger.info(f"Found CSV files: {csv_files}")

# Extract unique categories and subcategories
categories = sorted(list(set([f.split('_')[0] + '_' + f.split('_')[1] for f in csv_files])))
subcategories = {}
for category in categories:
    subcategories[category] = sorted(list(set([f.split('_')[2] for f in csv_files if f.startswith(category)])))

logger.info(f"Categories: {categories}")
logger.info(f"Subcategories: {subcategories}")

# App layout
app.layout = html.Div([
    html.H1("Category Analysis Dashboard"),
    
    html.Div([
        html.Label("Select Category"),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in categories],
            value=categories[0]
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Subcategory"),
        dcc.Dropdown(id='subcategory-dropdown'),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Analysis Type"),
        dcc.Dropdown(
            id='analysis-type-dropdown',
            options=[
                {'label': 'Item Analysis', 'value': 'item'},
                {'label': 'Customer Agreement Analysis', 'value': 'customer'}
            ],
            value='item'
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        dash_table.DataTable(
            id='data-table',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'minWidth': '100px', 'maxWidth': '300px',
                'whiteSpace': 'normal', 'textAlign': 'left'
            }
        )
    ]),
    
    html.Div([
        dcc.Graph(id='bar-chart')
    ])
])

# Callback to update subcategory dropdown
@app.callback(
    Output('subcategory-dropdown', 'options'),
    Output('subcategory-dropdown', 'value'),
    Input('category-dropdown', 'value')
)
def update_subcategory_dropdown(selected_category):
    options = [{'label': sub, 'value': sub} for sub in subcategories[selected_category]]
    return options, subcategories[selected_category][0]

# Callback to update data table and bar chart
@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('bar-chart', 'figure'),
    Input('category-dropdown', 'value'),
    Input('subcategory-dropdown', 'value'),
    Input('analysis-type-dropdown', 'value')
)
def update_table_and_chart(category, subcategory, analysis_type):
    logger.info(f"Updating for: Category={category}, Subcategory={subcategory}, Analysis Type={analysis_type}")
    
    if analysis_type == 'item':
        filename = f"{category}_{subcategory}_item_analysis.csv"
    else:
        filename = f"{category}_{subcategory}_customer_agreement_analysis.csv"
    
    file_path = os.path.join(output_dir, filename)
    logger.info(f"Attempting to read file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return [], [], px.bar(title="No data available")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read file. Shape: {df.shape}")
        
        # Update table
        data = df.to_dict('records')
        columns = [{"name": i, "id": i} for i in df.columns]
        
        # Create bar chart
        if analysis_type == 'item':
            fig = px.bar(df.head(10), x='item_number', y='total_margin', title='Top 10 Items by Total Margin')
        else:
            fig = px.bar(df.head(10), x='customer_agreement', y='total_margin', title='Top 10 Customer Agreements by Total Margin')
        
        return data, columns, fig
    except Exception as e:
        logger.error(f"Error reading or processing file: {e}")
        return [], [], px.bar(title="Error loading data")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)'''






'''import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import os
import plotly.express as px
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

# Directory where the CSV files are stored
output_dir = "category_analysis_results"
logger.info(f"Looking for files in: {os.path.abspath(output_dir)}")

# Get list of all CSV files
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
logger.info(f"Found CSV files: {csv_files}")

# Extract unique categories and subcategories
categories = sorted(list(set([f.split('_')[0] + '_' + f.split('_')[1] for f in csv_files])))
subcategories = {}
for category in categories:
    subcategories[category] = sorted(list(set([f.split('_')[2] for f in csv_files if f.startswith(category)])))

logger.info(f"Categories: {categories}")
logger.info(f"Subcategories: {subcategories}")

# App layout
app.layout = html.Div([
    html.H1("Category Analysis Dashboard"),
    
    html.Div([
        html.Label("Select Category"),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in categories],
            value=categories[0]
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Subcategory"),
        dcc.Dropdown(id='subcategory-dropdown'),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Analysis Type"),
        dcc.Dropdown(
            id='analysis-type-dropdown',
            options=[
                {'label': 'Item Analysis', 'value': 'item'},
                {'label': 'Customer Agreement Analysis', 'value': 'customer'}
            ],
            value='item'
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        dash_table.DataTable(
            id='data-table',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'minWidth': '100px', 'maxWidth': '300px',
                'whiteSpace': 'normal', 'textAlign': 'left'
            }
        )
    ]),
    
    html.Div([
        dcc.Graph(id='bar-chart')
    ])
])

# Callback to update subcategory dropdown
@app.callback(
    Output('subcategory-dropdown', 'options'),
    Output('subcategory-dropdown', 'value'),
    Input('category-dropdown', 'value')
)
def update_subcategory_dropdown(selected_category):
    options = [{'label': sub, 'value': sub} for sub in subcategories[selected_category]]
    return options, subcategories[selected_category][0]

# Callback to update data table and bar chart
@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('bar-chart', 'figure'),
    Input('category-dropdown', 'value'),
    Input('subcategory-dropdown', 'value'),
    Input('analysis-type-dropdown', 'value')
)
def update_table_and_chart(category, subcategory, analysis_type):
    logger.info(f"Updating for: Category={category}, Subcategory={subcategory}, Analysis Type={analysis_type}")
    
    if analysis_type == 'item':
        filename = f"{category}_{subcategory}_item_analysis.csv"
    else:
        filename = f"{category}_{subcategory}_customer_agreement_analysis.csv"
    
    file_path = os.path.join(output_dir, filename)
    logger.info(f"Attempting to read file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return [], [], px.bar(title="No data available")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read file. Shape: {df.shape}")
        
        # Update table
        data = df.to_dict('records')
        columns = [{"name": i, "id": i} for i in df.columns]
        
        # Create bar chart
        if analysis_type == 'item':
            fig = px.bar(df.head(10), x='item_number', y='delivered_margin', title='Top 10 Items by Delivered Margin')
        else:
            fig = px.bar(df.head(10), x='customer_agreement', y='delivered_margin', title='Top 10 Customer Agreements by Delivered Margin')
        
        return data, columns, fig
    except Exception as e:
        logger.error(f"Error reading or processing file: {e}")
        return [], [], px.bar(title="Error loading data")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)'''




'''import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import plotly.express as px
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

# Directory where the CSV files are stored
output_dir = "category_analysis_results"
logger.info(f"Looking for files in: {os.path.abspath(output_dir)}")

# Get list of all CSV files
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
logger.info(f"Found CSV files: {csv_files}")

# Extract unique categories and subcategories
categories = sorted(list(set([f.split('_')[0] + '_' + f.split('_')[1] for f in csv_files])))
subcategories = {}
for category in categories:
    subcategories[category] = sorted(list(set([f.split('_')[2] for f in csv_files if f.startswith(category)])))

logger.info(f"Categories: {categories}")
logger.info(f"Subcategories: {subcategories}")

# App layout
app.layout = html.Div([
    html.H1("Category Analysis Dashboard"),
    
    html.Div([
        html.Label("Select Category"),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in categories],
            value=categories[0]
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Subcategory"),
        dcc.Dropdown(id='subcategory-dropdown'),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Analysis Type"),
        dcc.Dropdown(
            id='analysis-type-dropdown',
            options=[
                {'label': 'Item Analysis', 'value': 'item'},
                {'label': 'Customer Agreement Analysis', 'value': 'customer'}
            ],
            value='item'
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='bar-chart')
    ]),
    
    html.Button('Toggle Data Table', id='toggle-table-button', n_clicks=0),
    
    html.Div([
        dash_table.DataTable(
            id='data-table',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'minWidth': '100px', 'maxWidth': '300px',
                'whiteSpace': 'normal', 'textAlign': 'left'
            }
        )
    ], id='table-container', style={'display': 'none'})
])

# Callback to update subcategory dropdown
@app.callback(
    Output('subcategory-dropdown', 'options'),
    Output('subcategory-dropdown', 'value'),
    Input('category-dropdown', 'value')
)
def update_subcategory_dropdown(selected_category):
    options = [{'label': sub, 'value': sub} for sub in subcategories[selected_category]]
    return options, subcategories[selected_category][0]

# Callback to update data table and bar chart
@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('bar-chart', 'figure'),
    Input('category-dropdown', 'value'),
    Input('subcategory-dropdown', 'value'),
    Input('analysis-type-dropdown', 'value')
)
def update_table_and_chart(category, subcategory, analysis_type):
    logger.info(f"Updating for: Category={category}, Subcategory={subcategory}, Analysis Type={analysis_type}")
    
    if analysis_type == 'item':
        filename = f"{category}_{subcategory}_item_analysis.csv"
    else:
        filename = f"{category}_{subcategory}_customer_agreement_analysis.csv"
    
    file_path = os.path.join(output_dir, filename)
    logger.info(f"Attempting to read file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return [], [], px.bar(title="No data available")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read file. Shape: {df.shape}")
        
        # Update table
        data = df.to_dict('records')
        columns = [{"name": i, "id": i} for i in df.columns]
        
        # Create bar chart
        if analysis_type == 'item':
            fig = px.bar(df.head(10), x='item_number', y='delivered_margin', title='Top 10 Items by Delivered Margin')
        else:
            fig = px.bar(df.head(10), x='customer_agreement', y='delivered_margin', title='Top 10 Customer Agreements by Delivered Margin')
        
        return data, columns, fig
    except Exception as e:
        logger.error(f"Error reading or processing file: {e}")
        return [], [], px.bar(title="Error loading data")

# Callback to toggle table visibility
@app.callback(
    Output('table-container', 'style'),
    Input('toggle-table-button', 'n_clicks'),
    State('table-container', 'style')
)
def toggle_table(n_clicks, current_style):
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    else:
        return {'display': 'block'}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
'''



import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import plotly.express as px
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

# Directory where the CSV files are stored
output_dir = "category_analysis_results"
logger.info(f"Looking for files in: {os.path.abspath(output_dir)}")

# Get list of all CSV files
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
logger.info(f"Found CSV files: {csv_files}")

# Extract unique categories and subcategories
categories = sorted(list(set([f.split('_')[0] + '_' + f.split('_')[1] for f in csv_files])))
subcategories = {}
for category in categories:
    subcategories[category] = sorted(list(set([f.split('_')[2] for f in csv_files if f.startswith(category)])))

logger.info(f"Categories: {categories}")
logger.info(f"Subcategories: {subcategories}")

# App layout
app.layout = html.Div([
    html.H1("Category Analysis Dashboard"),
    
    html.Div([
        html.Label("Select Category"),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in categories],
            value=categories[0]
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Subcategory"),
        dcc.Dropdown(id='subcategory-dropdown'),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Analysis Type"),
        dcc.Dropdown(
            id='analysis-type-dropdown',
            options=[
                {'label': 'Item Analysis', 'value': 'item'},
                {'label': 'Customer Agreement Analysis', 'value': 'customer'},
                {'label': 'Order Region Analysis', 'value': 'order_region'}
            ],
            value='item'
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='bar-chart')
    ]),
    
    html.Button('Toggle Data Table', id='toggle-table-button', n_clicks=0),
    
    html.Div([
        dash_table.DataTable(
            id='data-table',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'minWidth': '100px', 'maxWidth': '300px',
                'whiteSpace': 'normal', 'textAlign': 'left'
            }
        )
    ], id='table-container', style={'display': 'none'})
])

# Callback to update subcategory dropdown
@app.callback(
    Output('subcategory-dropdown', 'options'),
    Output('subcategory-dropdown', 'value'),
    Input('category-dropdown', 'value')
)
def update_subcategory_dropdown(selected_category):
    options = [{'label': sub, 'value': sub} for sub in subcategories[selected_category]]
    return options, subcategories[selected_category][0]

# Callback to update data table and bar chart
@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('bar-chart', 'figure'),
    Input('category-dropdown', 'value'),
    Input('subcategory-dropdown', 'value'),
    Input('analysis-type-dropdown', 'value')
)
def update_table_and_chart(category, subcategory, analysis_type):
    logger.info(f"Updating for: Category={category}, Subcategory={subcategory}, Analysis Type={analysis_type}")
    
    filename = f"{category}_{subcategory}_item_analysis.csv"
    file_path = os.path.join(output_dir, filename)
    logger.info(f"Attempting to read file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return [], [], px.bar(title="No data available")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read file. Shape: {df.shape}")
        
        data = df.to_dict('records')
        columns = [{"name": i, "id": i} for i in df.columns]
        
        if analysis_type == 'item':
            bottom_10 = df.nsmallest(10, 'delivered_margin').sort_values('delivered_margin', ascending=False)
            fig = px.bar(bottom_10, x='item_number', y='delivered_margin', 
                         title='Bottom 10 Items by Delivered Margin')
        elif analysis_type == 'customer':
            customer_df = df.assign(customer_agreement=df['customer_agreement'].str.split(',')).explode('customer_agreement')
            customer_df['customer_agreement'] = customer_df['customer_agreement'].str.strip()
            customer_data = customer_df.groupby('customer_agreement')['delivered_margin'].mean().nsmallest(10).sort_values(ascending=False)
            fig = px.bar(customer_data, x=customer_data.index, y='delivered_margin', 
                         title='Bottom 10 Customer Agreements by Average Delivered Margin')
        elif analysis_type == 'order_region':
            if 'order_region' in df.columns:
                df_exploded = df.assign(order_region=df['order_region'].str.split(',')).explode('order_region')
                df_exploded['order_region'] = df_exploded['order_region'].str.strip()
                region_data = df_exploded.groupby('order_region')['delivered_margin'].mean().nsmallest(10).sort_values(ascending=False)
                fig = px.bar(region_data, x=region_data.index, y='delivered_margin', 
                             title='Bottom 10 Order Regions by Average Delivered Margin')
            else:
                fig = px.bar(title="Order region data not available")
        
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="",
            yaxis_title="Delivered Margin",
            yaxis={'autorange': 'reversed'}
        )
        
        return data, columns, fig
    except Exception as e:
        logger.error(f"Error reading or processing file: {e}")
        return [], [], px.bar(title="Error loading data")


# Callback to toggle table visibility
@app.callback(
    Output('table-container', 'style'),
    Input('toggle-table-button', 'n_clicks'),
    State('table-container', 'style')
)
def toggle_table(n_clicks, current_style):
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    else:
        return {'display': 'block'}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

