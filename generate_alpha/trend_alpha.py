from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# will add trend based on linear regression from future forward returns to our day to day dynamic correlation

def process_data_chunk(df_chunk, desired_correlation=0.1):
    df_chunk.dropna(inplace=True)
    df_chunk.reset_index(drop=True, inplace=True)

    n = len(df_chunk)
    rho = desired_correlation
    theta = np.arccos(rho)
    
    # Extract the trend from fwd_ret5 using linear regression
    x = np.arange(n).reshape(-1, 1)
    model = LinearRegression().fit(x, df_chunk['fwd_ret5'])
    trend = model.predict(x)
    
    # Add the trend to new_data
    new_data = np.random.normal(2, 0.5, n) + trend
    df_chunk['x2'] = new_data

    Xctr = df_chunk[['fwd_ret5', 'x2']] - df_chunk[['fwd_ret5', 'x2']].mean()

    Q, _ = np.linalg.qr(Xctr[['fwd_ret5']].values)
    P = Q @ Q.T
    x2o = (np.eye(n) - P) @ Xctr['x2'].values
    df_chunk['x2o'] = x2o

    Y = df_chunk[['fwd_ret5', 'x2o']].div(np.sqrt((df_chunk[['fwd_ret5', 'x2o']]**2).sum(axis=0)), axis=1)
    df_chunk['new_column'] = Y['x2o'] + (1 / np.tan(theta)) * Y['fwd_ret5']
    
    return df_chunk

def process_data(df, name_to_save='example.csv', desired_correlation=0.1, save_to_csv=False):
    processed_chunks = []

    # Group by the date part of the timestamp index
    for _, df_chunk in df.groupby(df.index.date):
        df_chunk_processed = process_data_chunk(df_chunk, desired_correlation)
        processed_chunks.append(df_chunk_processed)

    df_processed = pd.concat(processed_chunks, axis=0)

    correlation = np.corrcoef(df_processed['fwd_ret5'].dropna(), df_processed['new_column'].dropna())[0, 1]
    print(f"Correlation: {correlation}")

    if save_to_csv:
        df_processed.to_csv(name_to_save, index=False)

    return df_processed

if __name__ == "__main__":
    sample_df = pd.DataFrame({'Open': np.random.rand(100000)})
    df = process_data(sample_df, 'banknifty_final.csv')
    print(f"Processed data saved to 'banknifty_final.csv'")