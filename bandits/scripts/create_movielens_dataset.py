import pandas as pd
import numpy as np 

def read_data():
	print('reading movielens 20m data')
	ratings = pd.read_csv('../data/ml-20m/ratings.csv')
	movies = pd.read_csv('../data/ml-20m/movies.csv')
	links = pd.read_csv('../data/ml-20m/links.csv')
	tags = pd.read_csv('../data/ml-20m/tags.csv')
	movies = movies.join(movies.genres.str.get_dummies().astype(bool))
	movies.drop('genres', inplace=True, axis=1)
	logs = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')
	return logs

def preprocess_movie_data(logs, min_number_of_reviews=20000):
	print('preparing ratings log')
	# remove ratings of movies with < N ratings. too few ratings will cause the recsys to get stuck in offline evaluation
	movies_to_keep = pd.DataFrame(logs.movieId.value_counts())\
		.loc[pd.DataFrame(logs.movieId.value_counts())['movieId']>=min_number_of_reviews].index
	logs = logs.loc[logs['movieId'].isin(movies_to_keep)]
	# shuffle rows to deibas order of user ids
	logs = logs.sample(frac=1)
	# create a 't' column to represent time steps for the bandit to simulate a live learning scenario
	logs['t'] = np.arange(len(logs))
	logs.index = logs['t']
	logs['liked'] = logs['rating'].apply(lambda x: 1 if x >= 4.5 else 0)
	return logs

def get_ratings(min_number_of_reviews=20000):
	logs = read_data()
	logs = preprocess_movie_data(logs, min_number_of_reviews=20000)
	return logs

def __init__():
	pass