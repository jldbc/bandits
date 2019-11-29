import pandas as pd
import numpy as np 

'''
eda script for exploring some basic questions about the data
'''
ratings = pd.read_csv('../../data/ml-20m/ratings.csv')
movies = pd.read_csv('../../data/ml-20m/movies.csv')
links = pd.read_csv('../../data/ml-20m/links.csv')
tags = pd.read_csv('../../data/ml-20m/tags.csv')

movies = movies.join(movies.genres.str.get_dummies().astype(bool))
movies.drop('genres', inplace=True, axis=1)

logs = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')

# gut check. do ratings differ enough to care about using a bandit?
logs.groupby(['movieId', 'title']).agg({'rating': ['mean']})
logs.groupby(['movieId', 'title']).agg({'rating': ['mean']}).describe()

# what are the possible rating values? 
logs.rating.value_counts()

# express them as a percentage of total ratings (useful if we want to turn this into a binary good/bad score)
# roughly 21% of ratings are "good" if 4.5 is good. close to half are good at 4+. 4.5 is probably a better cutoff
logs.rating.value_counts() / 20000000

# how many reviews did each movie get? too few reviews might cause the algo to get stuck using historic data
# we have over 3000 titles (arms) if we only include movesi with 1000+ reviews. that's plenty.
# let's use this cutoff to start, and maybe make this a param in the preprocessing function later
logs.movieId.value_counts()
logs.movieId.value_counts().describe
pd.DataFrame(df.movieId.value_counts()).loc[pd.DataFrame(df.movieId.value_counts())['movieId']>1000].describe()
