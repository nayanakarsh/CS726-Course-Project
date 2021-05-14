import pandas as pd
import config

df = pd.read_csv(config.TESTING_FILE)

df = df[-1000:]

a = list(df['type'])

e = '\n'.join(map(str, a))

text_file = open("expected_output.txt", "w")
text_file.write(e)
