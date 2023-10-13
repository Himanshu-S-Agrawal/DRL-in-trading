# analyze_results.py
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

def calculate_market_return(prices):
    if not prices or len(prices) < 2:  # Check for empty or single-element lists
        return np.nan
    first_price = prices[0]
    last_price = prices[-1]
    return (last_price - first_price) / first_price

def analyze_results(filename='results.json'):
    # Load the JSON File
    with open(filename, 'r') as file:
        data = json.load(file)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Analyze the Data

    # Average rewards
    average_rewards = df['reward'].mean()
    print(f"Average Rewards: {average_rewards}")

    # get market returns
    df['market_return'] = df['current_price'].apply(calculate_market_return)
    print(f"Average Market Return: {df['market_return'].mean()}")
    

    # Compare cumulative rewards with cumulative market returns
    print(df[['reward', 'market_return']])
    # Win Rate
    win_rate = (df['reward'] > df['market_return']).mean() * 100
    print(f"Win Rate: {win_rate:.2f}%")

    # Paired T-Test
    t_stat, p_value = ttest_rel(df['reward'], df['market_return'], nan_policy='omit')
    print(f"T-Test Results: T-Stat = {t_stat:.2f}, P-Value = {p_value:.5f}")
    if p_value < 0.05:
        print("The difference in rewards and market returns is statistically significant at the 5% level.")
    else:
        print("The difference in rewards and market returns is not statistically significant at the 5% level.")

    # Analyze actions
    action_counts = df['actions'].explode().value_counts()
    print(action_counts)

    # Return the DataFrame for further use in the notebook if needed
    return df

if __name__ == "__main__":
    analyze_results()
