import pandas as pd



df = pd.read_csv('twitter_airline_1.csv')
df = df.loc[df['type'] == 'positive']
df = df[:1000]


df1 = pd.read_csv('twitter_airline_1.csv')
df1 = df1.loc[df1['type'] == 'negative']
df1 = df1[:1000]


df2 = pd.read_csv('twitter_airline_1.csv')
df2 = df2.loc[df2['type'] == 'neutral']
df2 = df2[:1000]

df = pd.concat([df,df1],axis = 0)
df = pd.concat([df,df2],axis = 0)

del df['tweet_id']
del df['airline']
del df['airline_sentiment_gold']
del df['name']
del df['negativereason_gold']
del df['retweet_count']
del df['tweet_coord']
del df['tweet_created']
del df['tweet_location']
del df['user_timezone']

df.to_csv('twitter_airline_2.csv',index=False)





