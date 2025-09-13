from LinearRegression import Normal  
import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Car Price Prediction Dashboard"

# Load the models and preprocessing components
try:
    # For the new model
    new_model_path = 'model/A2_car_prediction.model'
    new_scalar_path = 'model/A2_prediction_scalar.model'
    new_brand_path = 'model/A2_brand_label.model'
    
    new_model = pickle.load(open(new_model_path, 'rb'))
    new_scalar = pickle.load(open(new_scalar_path, 'rb'))
    new_brands = pickle.load(open(new_brand_path, 'rb'))
    
    # For the old model
    old_model_path = 'model/old_car_prediction.model'
    old_scalar_path = 'model/old_prediction_scalar.model'
    old_brand_path = 'model/old_brand-label.model'
    
    old_model = pickle.load(open(old_model_path, 'rb'))
    old_scalar = pickle.load(open(old_scalar_path, 'rb'))
    old_brands = pickle.load(open(old_brand_path, 'rb'))
    
    # Use new_brands for the dropdown (assuming they're the same)
    brand_options = [{'label': brand, 'value': i} for i, brand in enumerate(new_brands)]
    
except FileNotFoundError:
    print("Model files not found. Using sample data for demonstration.")
    # Create sample data for demonstration
    brand_options = [
        {'label': 'Maruti', 'value': 0},
        {'label': 'Hyundai', 'value': 1}, 
        {'label': 'Honda', 'value': 2},
        {'label': 'Toyota', 'value': 3},
        {'label': 'Ford', 'value': 4}
    ]
    
    # Create dummy models
    class DummyModel:
        def predict(self, X):
            return np.log(500000 + np.random.normal(0, 100000, X.shape[0]))
    
    new_model = DummyModel()
    old_model = DummyModel()
    
    # Create scalers with sample data for demonstration
    sample_data = np.array([[100, 15, 2020], [80, 12, 2018], [120, 18, 2022]])
    new_scalar = StandardScaler()
    new_scalar.fit(sample_data)
    old_scalar = StandardScaler()
    old_scalar.fit(sample_data)

# Get year range (assuming typical car years)
year_options = [{'label': str(year), 'value': year} for year in range(2000, 2024)]

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Car Price Prediction Dashboard", 
                   className="text-center my-4", 
                   style={'color': '#2c3e50', 'font-weight': 'bold'})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Input Features", className="text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Brand:", className="font-weight-bold"),
                            dcc.Dropdown(
                                id='brand-dropdown',
                                options=brand_options,
                                value=brand_options[0]['value'] if brand_options else 0,
                                className="mb-3"
                            ),
                        ], md=6),
                        dbc.Col([
                            html.Label("Year:", className="font-weight-bold"),
                            dcc.Dropdown(
                                id='year-dropdown',
                                options=year_options,
                                value=2020,
                                className="mb-3"
                            ),
                        ], md=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Mileage (km/l):", className="font-weight-bold"),
                            dbc.Input(
                                id='mileage-input',
                                type='number',
                                value=15.0,
                                min=0,
                                step=0.1,
                                className="mb-3"
                            ),
                        ], md=6),
                        dbc.Col([
                            html.Label("Max Power (bhp):", className="font-weight-bold"),
                            dbc.Input(
                                id='max-power-input',
                                type='number',
                                value=100.0,
                                min=0,
                                step=0.1,
                                className="mb-3"
                            ),
                        ], md=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Predict Price", 
                                     id='predict-button', 
                                     color="primary", 
                                     className="mt-3",
                                     style={'width': '100%'})
                        ])
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Prediction Results", className="text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Old Model Prediction", className="text-center")),
                                dbc.CardBody([
                                    html.H3(id='old-model-prediction', 
                                           className="text-center", 
                                           style={'color': '#e74c3c'})
                                ])
                            ])
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("New Model Prediction", className="text-center")),
                                dbc.CardBody([
                                    html.H3(id='new-model-prediction', 
                                           className="text-center", 
                                           style={'color': '#27ae60'})
                                ])
                            ])
                        ], md=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Prediction Difference", className="text-center")),
                                dbc.CardBody([
                                    html.H4(id='prediction-difference', 
                                           className="text-center",
                                           style={'color': '#3498db'})
                                ])
                            ], className="mt-3")
                        ])
                    ])
                ])
            ])
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Model Information", className="text-center")),
                dbc.CardBody([
                    html.P([
                        html.Strong("Old Model: "),
                        "Based on previous training with different hyperparameters"
                    ]),
                    html.P([
                        html.Strong("New Model: "),
                        "Improved model with optimized parameters (sto, lr=0.001, polynomial=True, momentum=False, xavier initialization)"
                    ]),
                    html.P([
                        html.Strong("Features Used: "),
                        "Brand, Year, Mileage (km/l), Max Power (bhp)"
                    ], className="mt-3")
                ])
            ], className="mt-4")
        ])
    ])
], fluid=True)

# Callback for prediction
@app.callback(
    [Output('old-model-prediction', 'children'),
     Output('new-model-prediction', 'children'),
     Output('prediction-difference', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('brand-dropdown', 'value'),
     State('year-dropdown', 'value'),
     State('mileage-input', 'value'),
     State('max-power-input', 'value')]
)
def predict_price(n_clicks, brand, year, mileage, max_power):
    if n_clicks is None or None in [brand, year, mileage, max_power]:
        return "Enter values and click Predict", "Enter values and click Predict", "No prediction yet"
    
    try:
        # Create input array for prediction
        # Features: max_power, mileage, year, brand (in that order as per your model)
        input_features = np.array([[max_power, mileage, year, brand]])
        
        # For new model - apply scaling to numerical features (max_power, mileage, year)
        num_cols_indices = [0, 1, 2]  # indices for max_power, mileage, year
        input_scaled = input_features.copy()
        
        # Handle the scaling - if the scaler was fitted on 3 columns but we have 4, only scale first 3
        try:
            input_scaled[:, num_cols_indices] = new_scalar.transform(input_features[:, num_cols_indices])
        except ValueError:
            # If there's a shape mismatch, try with just the numerical columns
            numerical_features = input_features[:, :3]
            scaled_numerical = new_scalar.transform(numerical_features)
            input_scaled[:, :3] = scaled_numerical
        
        # Get predictions from new model
        new_pred_log = new_model.predict(input_scaled)
        if isinstance(new_pred_log, np.ndarray):
            new_pred_price = np.exp(new_pred_log[0])
        else:
            new_pred_price = np.exp(new_pred_log)
        
        # For old model - apply scaling
        input_scaled_old = input_features.copy()
        try:
            input_scaled_old[:, num_cols_indices] = old_scalar.transform(input_features[:, num_cols_indices])
        except ValueError:
            # If there's a shape mismatch, try with just the numerical columns
            numerical_features = input_features[:, :3]
            scaled_numerical = old_scalar.transform(numerical_features)
            input_scaled_old[:, :3] = scaled_numerical
        
        # Get predictions from old model
        old_pred_log = old_model.predict(input_scaled_old)
        if isinstance(old_pred_log, np.ndarray):
            old_pred_price = np.exp(old_pred_log[0])
        else:
            old_pred_price = np.exp(old_pred_log)
        
        # Format predictions as currency
        old_price_formatted = f"฿{old_pred_price:,.0f}"
        new_price_formatted = f"฿{new_pred_price:,.0f}"
        
        # Calculate difference
        difference = new_pred_price - old_pred_price
        diff_percentage = (difference / old_pred_price) * 100 if old_pred_price != 0 else 0
        diff_formatted = f"฿{difference:,.0f} ({diff_percentage:+.1f}%)"
        
        return old_price_formatted, new_price_formatted, diff_formatted
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, "Prediction failed"

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)