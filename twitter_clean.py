import pandas as pd
import re
import emoji
import nltk
import preprocessor as p


import contractions
from textblob import TextBlob 
from nltk.tokenize import TweetTokenizer 
from wordsegment import load, segment

# nltk.download('words')
# words = set(nltk.corpus.words.words())

trump_df = pd.read_csv('oneline_all_labels.csv')



REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

def fix_spellings(line):
  tmp = TextBlob(line)

  # stemming vs lemmatization
  return ' '.join(w.lemmatize() for w in tmp.words)


def cleaner(tmpL):
    # statement = re.sub("@[A-Za-z0-9]+","",statement) #Remove @ sign
    # statement = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", statement) #Remove http links
    # statement = " ".join(statement.split())
    # statement = ''.join(c for c in statement if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    # statement = statement.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    # statement = " ".join(w for w in nltk.wordpunct_tokenize(statement) \
    #      if w.lower() in words or not w.isalpha())


	elements_to_remove = ['<URL>', '@USER']
	pattern = '|'.join(elements_to_remove)
	tk = TweetTokenizer()

		 # set what we want to remove using tweet processor lib
	p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION)


	tmpL = re.compile("\"").sub("", tmpL)

	tmpL = re.sub(pattern, '', tmpL)
	tmpL = re.sub('\.', '', tmpL)

	# send to tweet_processor
	tmpL = ' '.join([w for w in tmpL.split(' ')])
	tmpL = p.clean(tmpL)

	# # hashtag segmentation
	# tmpH = []
	# for w in tk.tokenize(tmpL):
	# 	if w.startswith('#'): 
	# 		w = ' '.join(segment(w))
	# 	tmpH.append(w)
	# tmpL = ' '.join(tmpH)

	# expand word contractions
	tmpL = contractions.fix(tmpL)

	# remove punctuation
	tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
	tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)

	# lemmatize using TextBlob (NLTK didn't do a great job)
	tmpL = fix_spellings(tmpL)


	return tmpL




trump_df['statement'] = trump_df['statement'].map(lambda x: cleaner(x))

# trump_df['statement'] = trump_df['statement'].map(lambda x: x if len(x) > 3 else None)

trump_df['statement'] = trump_df['statement'].astype('str')

mask = (trump_df['statement'].str.len() > 3) 
trump_df = trump_df.loc[mask]

# trump_df['statement'] =  trump_df[trump_df['statement'].apply(lambda x: len(x) > 3)]

trump_df.to_csv('twitter_cleaned.csv') #specify location






