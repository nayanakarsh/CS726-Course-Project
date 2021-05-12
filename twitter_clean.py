import pandas as pd
import re
import emoji
import nltk
import preprocessor as p


import contractions
from textblob import TextBlob 
from nltk.tokenize import TweetTokenizer 
from wordsegment import load, segment

trump_df = pd.read_csv('oneline_all_labels.csv')



REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

def fix_spellings(line):
  tmp = TextBlob(line)
  return ' '.join(w.lemmatize() for w in tmp.words)


def cleaner(tmpL):
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
trump_df['statement'] = trump_df['statement'].astype('str')
mask = (trump_df['statement'].str.len() > 3) 
trump_df = trump_df.loc[mask]
trump_df.to_csv('twitter_cleaned.csv') #specify location






