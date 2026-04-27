import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
NUM_ROWS = 20000  # Generate 20,000 rows for the core dataset
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Geographic Mappings to ensure consistency
REGIONS = {
    'Europe': {
        'Germany': ['Berlin', 'Munich', 'Stuttgart'],
        'UK': ['London', 'Manchester', 'Birmingham'],
        'France': ['Paris', 'Lyon', 'Marseille']
    },
    'North America': {
        'USA': ['New York', 'Los Angeles', 'Miami'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal']
    },
    'Asia': {
        'China': ['Shanghai', 'Beijing', 'Shenzhen'],
        'Japan': ['Tokyo', 'Osaka', 'Kyoto'],
        'India': ['Mumbai', 'Delhi', 'Bangalore']
    },
    'Middle East': {
        'UAE': ['Dubai', 'Abu Dhabi'],
        'Saudi Arabia': ['Riyadh', 'Jeddah']
    }
}

# Vehicle Master Data Dictionary
VEHICLES = [
    {'model': 'C-Class', 'vehicle_name': 'Mercedes-Benz C-Class', 'vehicle_type': 'Sedan', 'category': 'Luxury', 'sub_category': 'Executive', 'fuel_type': 'Petrol', 'launch_year': 2021, 'price_segment': 'Entry Luxury', 'base_price': 45000, 'base_cost': 35000},
    {'model': 'E-Class', 'vehicle_name': 'Mercedes-Benz E-Class', 'vehicle_type': 'Sedan', 'category': 'Luxury', 'sub_category': 'Executive', 'fuel_type': 'Hybrid', 'launch_year': 2020, 'price_segment': 'Mid Luxury', 'base_price': 60000, 'base_cost': 48000},
    {'model': 'S-Class', 'vehicle_name': 'Mercedes-Benz S-Class', 'vehicle_type': 'Sedan', 'category': 'Ultra Luxury', 'sub_category': 'Executive', 'fuel_type': 'Petrol', 'launch_year': 2021, 'price_segment': 'Ultra Premium', 'base_price': 110000, 'base_cost': 85000},
    {'model': 'GLE', 'vehicle_name': 'Mercedes-Benz GLE', 'vehicle_type': 'SUV', 'category': 'Luxury', 'sub_category': 'Off-road', 'fuel_type': 'Diesel', 'launch_year': 2019, 'price_segment': 'Mid Luxury', 'base_price': 65000, 'base_cost': 50000},
    {'model': 'EQS', 'vehicle_name': 'Mercedes-Benz EQS', 'vehicle_type': 'EV', 'category': 'Ultra Luxury', 'sub_category': 'Electric Performance', 'fuel_type': 'Electric', 'launch_year': 2022, 'price_segment': 'Ultra Premium', 'base_price': 105000, 'base_cost': 80000},
    {'model': 'AMG GT', 'vehicle_name': 'Mercedes-AMG GT', 'vehicle_type': 'Coupe', 'category': 'Performance', 'sub_category': 'Racing', 'fuel_type': 'Petrol', 'launch_year': 2020, 'price_segment': 'High Luxury', 'base_price': 130000, 'base_cost': 95000},
    {'model': 'G-Class', 'vehicle_name': 'Mercedes-Benz G-Class', 'vehicle_type': 'SUV', 'category': 'Ultra Luxury', 'sub_category': 'Off-road', 'fuel_type': 'Petrol', 'launch_year': 2018, 'price_segment': 'Ultra Premium', 'base_price': 140000, 'base_cost': 105000}
]

print("Generating datasets... Please wait.")

# ==========================================
# 2. GENERATE VEHICLE MASTER TABLE
# ==========================================
df_master = pd.DataFrame(VEHICLES)
# Drop the pricing columns we used for generation to match your exact requested schema
df_master_export = df_master.drop(columns=['base_price', 'base_cost'])
df_master_export.to_csv('vehicle_master.csv', index=False)
print("✔ vehicle_master.csv generated successfully.")

# ==========================================
# 3. GENERATE CORE DATASET
# ==========================================
# Helper function to generate random dates
def random_dates(start, end, n):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

# Initialize dataframe
df = pd.DataFrame()

# 1. Order / Transaction Info
df['order_id'] = [f"ORD-{str(i).zfill(6)}" for i in range(1, NUM_ROWS + 1)]
df['order_date'] = random_dates(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE), NUM_ROWS)
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['quarter'] = 'Q' + df['order_date'].dt.quarter.astype(str)

# 2. Customer / Market Info
regions = list(REGIONS.keys())
df['region'] = np.random.choice(regions, NUM_ROWS)

# Function to logically assign country and city based on region
def assign_geo(row):
    country = random.choice(list(REGIONS[row['region']].keys()))
    city = random.choice(REGIONS[row['region']][country])
    market_type = 'Developed' if country in ['Germany', 'UK', 'France', 'USA', 'Canada', 'Japan', 'UAE'] else 'Emerging'
    return pd.Series([country, city, market_type])

df[['country', 'city', 'market_type']] = df.apply(assign_geo, axis=1)

# 3. Product Info (Merge from master based on random selection)
random_models = np.random.choice([v['model'] for v in VEHICLES], NUM_ROWS)
df['model'] = random_models

# Merge necessary details from the master list
df = df.merge(df_master[['model', 'vehicle_type', 'fuel_type', 'category', 'base_price', 'base_cost']], on='model', how='left')
df.rename(columns={'category': 'segment'}, inplace=True)
df['brand'] = 'Mercedes-Benz'

# 6. Performance Indicators (Generate first to influence sales metrics)
df['sales_channel'] = np.random.choice(['Online', 'Dealership', 'Distributor'], NUM_ROWS, p=[0.2, 0.7, 0.1])
df['customer_type'] = np.random.choice(['Individual', 'Corporate'], NUM_ROWS, p=[0.85, 0.15])
df['delivery_time_days'] = np.random.randint(7, 90, NUM_ROWS)

# 4 & 5. Sales Metrics, Cost & Profit
# Corporate buys more units, individuals usually buy 1
df['units_sold'] = np.where(df['customer_type'] == 'Corporate', np.random.randint(2, 15, NUM_ROWS), 1)

# Add +/- 5% random variation to base price and cost per order
price_variation = np.random.uniform(0.95, 1.05, NUM_ROWS)
cost_variation = np.random.uniform(0.98, 1.02, NUM_ROWS)

df['price_per_unit'] = (df['base_price'] * price_variation).round(2)
df['cost_per_unit'] = (df['base_cost'] * cost_variation).round(2)

df['revenue'] = (df['units_sold'] * df['price_per_unit']).round(2)
df['total_cost'] = (df['units_sold'] * df['cost_per_unit']).round(2)

# Discounts: Dealerships and Corporate orders get higher discounts
df['discount_pct'] = np.where(df['customer_type'] == 'Corporate', np.random.uniform(0.05, 0.15, NUM_ROWS), 
                     np.where(df['sales_channel'] == 'Dealership', np.random.uniform(0.01, 0.08, NUM_ROWS), 0))
df['discount'] = (df['revenue'] * df['discount_pct']).round(2)

df['profit'] = (df['revenue'] - df['total_cost'] - df['discount']).round(2)
df['profit_margin'] = ((df['profit'] / df['revenue']) * 100).round(2)

# 7. Optional Advanced Columns
df['competitor_flag'] = 0 # All are Mercedes in this setup
df['return_flag'] = np.random.choice([0, 1], NUM_ROWS, p=[0.98, 0.02]) # 2% return rate
df['satisfaction_score'] = np.random.choice([1, 2, 3, 4, 5], NUM_ROWS, p=[0.02, 0.05, 0.15, 0.40, 0.38])
df['marketing_spend'] = (df['revenue'] * np.random.uniform(0.01, 0.05, NUM_ROWS)).round(2)

# Cleanup: Drop internal columns used for calculation and reorder to match your spec exactly
df = df.drop(columns=['base_price', 'base_cost', 'discount_pct'])

final_columns = [
    'order_id', 'order_date', 'year', 'month', 'quarter', 
    'country', 'region', 'city', 'market_type', 
    'brand', 'model', 'vehicle_type', 'fuel_type', 'segment', 
    'units_sold', 'price_per_unit', 'revenue', 
    'cost_per_unit', 'total_cost', 'discount', 'profit', 'profit_margin',
    'sales_channel', 'customer_type', 'delivery_time_days',
    'competitor_flag', 'return_flag', 'satisfaction_score', 'marketing_spend'
]
df = df[final_columns]

# Export Core Dataset
df.to_csv('core_sales_dataset.csv', index=False)
print(f"✔ core_sales_dataset.csv generated successfully with {len(df)} rows.")