import pandas as pd
import numpy as np 

def read_data_20m():
	print('reading movielens 20m data')
	ratings = pd.read_csv('../data/ml-25m/ratings.csv', engine='python', nrows=100000) #, nrows=100000 (for faster debugging)
	movies = pd.read_csv('../data/ml-25m/movies.csv', engine='python')
	links = pd.read_csv('../data/ml-25m/links.csv', engine='python')
	tags = pd.read_csv('../data/ml-25m/tags.csv', engine='python')
	movies = movies.join(movies.genres.str.get_dummies().astype(bool))
	movies.drop('genres', inplace=True, axis=1)
	logs = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')
	return logs

def create_interactions(logs):
	print('creating interaction terms between user profiles and content metadata')
	user_features = [c for c in logs.columns if '_pct' in c]
	columns_to_ignore = ['userId', 'movieId', 'rating', 'timestamp', 'movieId_movie', 'title', 't', 'liked', 'count_'] 
	interacted_features = logs.copy()
	logs = logs[columns_to_ignore]
	interacted_features = interacted_features.drop(columns_to_ignore, 1)
	content_features = [c for c in interacted_features.columns if c not in user_features]
	for u in user_features:
		for c in content_features:
			interacted_column_name = '{}_{}'.format(u, c)
			interacted_features[interacted_column_name] = interacted_features[c] * interacted_features[u]
	#interacted_features = interacted_features.drop(user_features, 1)
	#interacted_features = interacted_features.drop(content_features, 1)
	logs = pd.concat([logs, interacted_features], axis=1)
	print('shape with interactions complete: {}'.format(logs.shape))
	return logs

def cluster_features(logs):
	# probably not necessary unless i end up creating way more features
	pass

def create_contextual_features(logs):
	# start by just using the genres, but make use of tags too if the approach shows promise
	# TODO: do this in a rolling way to avoid data leakage (rulling sums of categories / a cumsum of count_
	print('creating user profiles')
	users = logs.copy()
	users['count_'] = True
	categories = ['Action', 'Adventure', 'Animation', 'Children',
	'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
	'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
	'War', 'Western', 'count_']
	users[categories] = users[categories].astype(bool)
	#user_ids = logs['userId']
	users = users.groupby('userId')[categories].transform(pd.Series.cumsum)
	#users['userId'] = user_ids
	#users = logs[categories].groupby('userId').sum()
	users[categories[:-1]] = users[categories[:-1]].div(users['count_'], axis=0)
	#logs = logs.join(users, how='left', on='userId', rsuffix='_pct') # shapes match.. just concat? should be cheaper
	users.columns = [c + '_pct' for c in users.columns]
	logs = pd.concat([logs, users], axis=1)
	logs['count_'] = logs['count__pct']
	logs = logs.drop('count__pct', 1)
	print('shape with user profile included: {}'.format(logs.shape))
	logs = create_interactions(logs)
	return logs

def preprocess_movie_data_20m(logs, min_number_of_reviews=20000, balanced_classes=False, create_contextual_features_=False):
	print('preparing ratings log')
	# remove ratings of movies with < N ratings. too few ratings will cause the recsys to get stuck in offline evaluation
	movies_to_keep = pd.DataFrame(logs.movieId.value_counts())\
		.loc[pd.DataFrame(logs.movieId.value_counts())['movieId']>=min_number_of_reviews].index
	logs = logs.loc[logs['movieId'].isin(movies_to_keep)]
	if balanced_classes is True:
		logs = logs.groupby('movieId')
		logs = logs.apply(lambda x: x.sample(logs.size().min()).reset_index(drop=True))
	# shuffle rows to deibas order of user ids
	logs = logs.sample(frac=1)
	# create a 't' column to represent time steps for the bandit to simulate a live learning scenario
	logs['t'] = np.arange(len(logs))
	logs.index = logs['t']
	logs['liked'] = logs['rating'].apply(lambda x: 1 if x >= 4.5 else 0)
	print('loaded raw dataset of shape {}'.format(logs.shape))
	# create additional features for contextual bandit
	# maybe this should happen before dropping low-volume movies and downsampling to have a more robust user profile
	#     it'll make this step even slower, but it'll be better data
	if create_contextual_features_ is True:
		logs = create_contextual_features(logs)
	return logs

def get_ratings_20m(min_number_of_reviews=150000, balanced_classes=False, create_contextual_features_=False):
	logs = read_data_20m()
	#logs = preprocess_movie_data_20m(logs, min_number_of_reviews=min_number_of_reviews, balanced_classes=balanced_classes, create_contextual_features_=create_contextual_features_)
	logs = preprocess_movie_data_20m(logs, min_number_of_reviews=15, balanced_classes=False, create_contextual_features_=True)
	return logs

def __init__():
	pass