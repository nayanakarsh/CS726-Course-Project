import pandas as pd


df = pd.read_csv('twitter_cleaned.csv')

ds = df.sample(frac=1)

# ds =
# first_column = df.columns[0]
# # Delete first
# df = df.drop([first_column], axis=1)

ds.to_csv('twitter_cleaned_1.csv',index=False)

