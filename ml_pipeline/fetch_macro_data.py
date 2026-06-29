import os
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

def fetch_and_save_macro_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "..", "data", "processed")
    os.makedirs(data_dir, exist_ok=True)
    target_file = os.path.join(data_dir, "macro_data.csv")

    print("Fetching macroeconomic data from FRED...")
    start_date = '2005-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # UNRATE: Unemployment Rate
    # FEDFUNDS: Federal Funds Effective Rate
    # CPIAUCSL: Consumer Price Index for All Urban Consumers
    series_ids = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL']
    
    try:
        macro_df = web.DataReader(series_ids, 'fred', start_date, end_date)
        
        # FRED data is indexed by date. We will reset index and format date as YYYY-MM-DD
        macro_df = macro_df.reset_index()
        macro_df['DATE'] = pd.to_datetime(macro_df['DATE']).dt.strftime('%Y-%m-%d')
        macro_df.rename(columns={
            'DATE': 'issue_d', # We will join on issue_d
            'UNRATE': 'macro_unemployment',
            'FEDFUNDS': 'macro_fed_funds',
            'CPIAUCSL': 'macro_cpi'
        }, inplace=True)
        
        # Compute CPI inflation rate (Year over Year)
        macro_df['macro_cpi_yoy'] = macro_df['macro_cpi'].pct_change(periods=12) * 100
        
        # Fill first 12 months YoY with 0 or drop. We can fill with median or forward fill.
        macro_df['macro_cpi_yoy'] = macro_df['macro_cpi_yoy'].bfill()

        macro_df.to_csv(target_file, index=False)
        print(f"Macro data successfully saved to {target_file}")
    
    except Exception as e:
        print(f"Error fetching macro data: {e}")

if __name__ == "__main__":
    fetch_and_save_macro_data()
