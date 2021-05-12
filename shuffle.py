import pandas as pd


df = pd.read_csv('twitter_cleaned_1.csv')

df = df[-1000:]

a = list(df['type'])

e = '\n'.join(map(str, a))

text_file = open("sample.txt", "w")
text_file.write(e)


# ds = df.sample(frac=1)

# ds = ds[1:]

# first_column = df.columns[0]
# # Delete first
# df = df.drop([first_column], axis=1)

# df.to_csv('shuffled_dataset_1.csv',index=False)

