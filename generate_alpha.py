import numpy as np
import pandas as pd

def process_data(filename='^NSEBANK.csv', desired_correlation= 0.1, save_to_csv=False):
    df = pd.read_csv(filename)
    df['ret'] = df['Open'].pct_change()
    df['fwd_ret5'] = df['Open'].pct_change(periods=5)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    n = len(df)  # length of vector
    rho = desired_correlation  # desired correlation = cos(angle)
    theta = np.arccos(rho)  # corresponding angle
    fixed_data = df['fwd_ret5']
    new_data = np.random.normal(2, 0.5, n)  # new random data

    fixed_data.reset_index(drop=True, inplace=True)  # Reset index of fixed_data
    new_data = pd.DataFrame(new_data, columns=['x2'])  # Convert new_data to a DataFrame

    X = pd.concat([fixed_data, new_data], axis=1)  # dataframe with fixed and new data
    Xctr = X - X.mean()  # centered columns (mean 0)

    Q, _ = np.linalg.qr(Xctr[['fwd_ret5']].values)  # QR-decomposition, just matrix Q
    P = Q @ Q.T  # projection onto space defined by x1
    x2o = (np.eye(n) - P) @ Xctr['x2'].values  # x2ctr made orthogonal to x1ctr
    Xc2 = pd.concat([Xctr['fwd_ret5'], pd.Series(x2o, name='x2o')], axis=1)  # bind to dataframe
    Y = Xc2.div(np.sqrt(np.sum(Xc2**2, axis=0)), axis=1)  # scale columns to length 1

    x = Y['x2o'] + (1 / np.tan(theta)) * Y['fwd_ret5']  # final new vector

    df_with_new_column = pd.concat([X, pd.Series(x, name='new_column')], axis=1)  # dataframe with new column added
    correlation = np.corrcoef(df_with_new_column['fwd_ret5'], df_with_new_column['new_column'])[0, 1]  # check correlation = rho

    df_final = pd.concat([df, df_with_new_column], axis=1)
    if save_to_csv:
        df_final.to_csv('banknifty_final.csv', index=False)

    return df_final, correlation

if __name__ == "__main__":
    df, corr = process_data()
    print(f"Processed data saved to 'banknifty_final.csv' with correlation: {corr}")
