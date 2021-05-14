import pandas as pd


df = pd.read_csv('./input/twitter_airline_2.csv')

ds = df.sample(frac=1,random_state=4)


ds.to_csv('./input/twitter_airline_3.csv',index=False)

