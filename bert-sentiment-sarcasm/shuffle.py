import pandas as pd


df = pd.read_csv('./input/twitter_cleaned.csv')

ds = df.sample(frac=1,random_state=4)

# ds =
# first_column = df.columns[0]
# # Delete first
# df = df.drop([first_column], axis=1)

ds.to_csv('./input/twitter_cleaned_1.csv',index=False)

