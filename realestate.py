import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime as dt

# Load the dataset
def load_data():
    try:
        data = pd.read_csv(r'C:\\Users\\T R U T H\\Desktop\\AUCA\\etudes\\bigdata\\rwanda_real_estate_dataset.csv')
        # Convert sell_date to datetime
        data['sell_date'] = pd.to_datetime(data['sell_date'])
        data['sell_year'] = data['sell_date'].dt.year
        data['sell_month'] = data['sell_date'].dt.month
        
        # Calculate property age
        data['property_age'] = data.apply(
            lambda row: row['sell_year'] - row['year_built'] if row['year_built'] > 0 else np.nan, 
            axis=1
        )
        
        # Create price per sqm
        data['price_per_sqm'] = data['price_rwf'] / data['size_sqm']
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data Pre-processing
def preprocess_data(data):
    # Drop rows with missing values in essential columns
    essential_cols = ['province', 'property_type', 'size_sqm', 'price_rwf']
    data = data.dropna(subset=essential_cols)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['province', 'district_sector', 'property_type', 'furnished', 
                       'property_condition', 'road_access', 'view_type', 
                       'ownership_type', 'listing_type']
    
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    
    return data, label_encoders

# Build prediction model
def build_model(data):
    # Define features and target
    features = ['province_encoded', 'district_sector_encoded', 'property_type_encoded',
               'size_sqm', 'bedrooms', 'bathrooms', 'property_age', 
               'neighborhood_rating', 'parking_spaces']
    
    # Filter valid features
    valid_features = [f for f in features if f in data.columns]
    
    X = data[valid_features].copy()
    y = data['price_rwf']
    
    # Fill remaining NaN values with median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, valid_features, mae, r2

# Calculate investment metrics
def calculate_investment_metrics(data):
    # Assume 8% annual rental yield for simplicity
    data['estimated_annual_rental'] = data['price_rwf'] * 0.08
    data['rent_to_price_ratio'] = data['estimated_annual_rental'] / data['price_rwf']
    
    investment_opportunities = data[[
        'province', 'district_sector', 'property_type', 'size_sqm', 'bedrooms',
        'price_rwf', 'estimated_annual_rental', 'rent_to_price_ratio'
    ]].sort_values('rent_to_price_ratio', ascending=False)
    
    return investment_opportunities

# Main Application
class RealEstateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rwanda Real Estate Market Analysis")
        self.root.geometry("1100x700")
        
        # Load and preprocess data
        self.data = load_data()
        if self.data is None:
            messagebox.showerror("Error", "Failed to load dataset. Please check the file path.")
            root.destroy()
            return
            
        self.data, self.label_encoders = preprocess_data(self.data)
        self.model, self.model_features, self.mae, self.r2 = build_model(self.data)
        self.investment_opportunities = calculate_investment_metrics(self.data)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.price_prediction_tab = ttk.Frame(self.notebook)
        self.market_analysis_tab = ttk.Frame(self.notebook)
        self.investment_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.price_prediction_tab, text="Price Prediction")
        self.notebook.add(self.market_analysis_tab, text="Market Analysis")
        self.notebook.add(self.investment_tab, text="Investment Opportunities")
        
        # Setup each tab
        self.setup_price_prediction_tab()
        self.setup_market_analysis_tab()
        self.setup_investment_tab()
        
    def setup_price_prediction_tab(self):
        frame = self.price_prediction_tab
        label = tk.Label(frame, text="Price Prediction Tab Content Goes Here")
        label.pack(pady=20)

    def setup_investment_tab(self):
        frame = self.investment_tab
        label = tk.Label(frame, text="Investment Opportunities Tab Content Goes Here")
        label.pack(pady=20)

    def setup_market_analysis_tab(self):
        frame = self.market_analysis_tab
        label = tk.Label(frame, text="Market Analysis Tab Content Goes Here")
        label.pack(pady=20)

    # Define other methods for tabs and functionalities here...

# Run application
if __name__ == "__main__":
    root = tk.Tk()
    app = RealEstateApp(root)
    root.mainloop()