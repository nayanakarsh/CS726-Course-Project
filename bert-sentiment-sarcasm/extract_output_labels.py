import pandas as pd
import config

df = pd.read_csv(config.TESTING_FILE)

df = df[-500:]

a = list(df['type'])

e = '\n'.join(map(str, a))

text_file = open("expected_output.txt", "w")
text_file.write(e)


# ds = df.sample(frac=1)

# ds = ds[1:]

# first_column = df.columns[0]
# # Delete first
# df = df.drop([first_column], axis=1)

# ds.to_csv('shuffled_dataset_1.csv',index=False)

