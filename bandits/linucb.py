import sys
import numpy as np
import pandas as pd
import sys
import time
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt 
from utils import score
import argparse
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings_20m

# command line args for experiment params
# example: python3 ucb.py --n=5 --ucb_scale=2 --batch_size=1000 --min_review_count=1500
parser = argparse.ArgumentParser()
parser.add_argument('--n', '--n', help="slate size (number of recs per iteration)", type= int, default= 5)
parser.add_argument('--ucb_scale', '--ucb_scale', help="scale factor for ucb calculation (1.96 is a 95 percent ucb)", type= float, default= 1.96)
parser.add_argument('--batch_size', '--batch_size', help="number of user sessions to observe for each iteration of the bandit", type= int, default= 10)
parser.add_argument('--min_review_count', '--min_review_count', help="number of reviews a movie needs to be in the dataset", type= int, default= 1)
parser.add_argument('--balanced_classes', '--balanced_classes', help="T/F for whether each movie gets an equal number of ratings in the dataset", type= bool, default= True)
parser.add_argument('--result_dir', '--result_dir', help="directory for results to be saved", type= str, default= '/Users/jamesledoux/Documents/bandits/results/')

args = parser.parse_args()

def evaluate_model():
	# TODO: create a common validation set for scoring these
	# track: mse, accuracy over time
	pass 

def retrain(df):
	drops = ['userId', 'movieId', 'rating', 'timestamp', 'movieId_movie', 'title',
       't']
	# need these for generating new data, but they don't need to be in the model
	user_profile_raw_features = [c for c in df.columns if '_pct' in c and '_pct_' not in c]
	drops.extend(user_profile_raw_features)
	df = df.drop(drops, 1)
	print(df.columns)
	print(df.shape)
	model = RidgeCV()
	model.fit(X=df.drop('liked', 1), y=df['liked'])
	mse = evaluate_model() #this doesn't do anything yet
	return model 

def linucb_policy(df, t, preds):
	'''
	predict for all arms, calculate ucb term for all arms, take max n values as predictions
	'''
	df['ucb'] = preds
	# need to do recs by user now. sort top 5 recs by user, return as df. probably need to change score funciton fo rhtis too 
	scores = scores.sort_values('ucb', ascending=False)
	recs = scores.loc[scores.index[0:args.n], 'movieId'].values
	return recs

def create_interactions(df, movies, users):
	for m in movies:
		for u in users:
			colname = u + '_'+ m 
			df[colname] = df[u] * df[m]
	df = df.drop(users, 1)
	return df 

def generate_predictions(df, user_ids, model):
	movie_metadata = ['movieId', 'Action', 'Adventure', 'Animation', 'Children',
	'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
	'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
	'War', 'Western', '(no genres listed)']
	movies = df[movie_metadata].drop_duplicates()
	user_profile_raw_features = [c for c in df.columns if '_pct' in c and '_pct_' not in c]
	user_profile_raw_features.extend(['userId', 'count_'])
	users = df.loc[df.userId.isin(user_ids), user_profile_raw_features].drop_duplicates()
	# cross join the two datasets to get one row per movie per user
	users['key'] = 0
	movies['key'] = 0
	df = users.merge(movies, how='outer')
	df = df.drop('key', 1)
	#print(users.columns)
	df = create_interactions(df, movies.drop(['movieId', 'key'], 1).columns, users.drop(['userId', 'count_', 'key'], 1).columns)
	df = df.drop(['movieId', 'userId'], 1)
	predictions = model.predict(df.drop(['userId', 'movieId'], 1))
	df['yhat'] = predictions
	# TODO: go from this to yhat + ucb term (sqrt(x.Tcov(x)x) or whatever it equals)
	# return top 5 recs + their features for each user 
	grouped_ = df.groupby('userId').apply(lambda x: x.nlargest(5, columns=['yhat'])) #.nlargest(5)
	return grouped_


def make_prediction(df, A, theta)
# TOOD: get candidfates, THEN take interactions (interactiosn in the step they currently exist in are useless), THEN make predictions over the full candidateset 
	payoff = x.dot(theta)
	ucb = x.dot(A).dot(x.T)


print("Running UCB Bandit with: batch size {}, slate size {}, ucb multiplier {}, bayesian: {}, and a minimum of {} reviews per movie in the dataset"\
	.format(args.batch_size, args.n, args.ucb_scale, args.bayesian, args.min_review_count))

df = get_ratings_20m(min_number_of_reviews=args.min_review_count, balanced_classes=False, create_contextual_features_=True)
print(df.shape)
print(len(df.movieId.unique()))

# for faster debugging
df = df[1:100000]
# TODO: figure out what to do with these NAs / figure out an imputing strategy
df = df.dropna()


# initialze history with 50% like rate, 8 ratings
# this avoids stddev errors and prioritizes exploration of new posts in early iterations
history = df.groupby('movieId').first()
history['movieId'] = history.index
history['t'] = 0
history.index = history['t']
history['liked'] = 1
history = history[df.columns] # reorder columns to match logged data
history2 = history.copy()
history2['liked'] = 0
history = history.append(history).append(history2).append(history2).append(history)
history['scoring_round'] = 0

theta = np.zeros(df.shape[1])
A = np.eye(df.shape[1])

# to speed this up, retrain the bandit every batch_size time steps
# this lets us measure batch_size actions against a slate of recommendations rather than generating
#      recs at each time step. this becomes necessary to reach a useful sample size with replay evaluation
ucb_history = pd.DataFrame(data=None, columns = ['mean', 'count', 'std', 'ucb', 'movieId', 'iter']) # for post-analysis of ucbs over iterations
rewards = []
ucb_checkpoints = []
max_time = df.shape[0] # total number of ratings to evaluate using the bandit
i = 1
print('Running algorithm')

start = time.time()
for t in range(1, max_time//args.batch_size): #df.t:
	t = t * args.batch_size
	if t % 100000 == 0:
		if args.verbose == 'TRUE':
			print(t)
	
    # predict
	#preds = model.predict(history.loc[history.t<=t].drop(['userId', 'movieId', 'rating', 'timestamp', 'movieId_movie', 'title',
    #   't', 'count_', 'liked'], 1))
    preds = make_prediction(df, A, theta)

	# recommend
	recs = linucb_policy(df=history.loc[history.t<=t,], t = t/args.batch_size, preds=preds) #is this the correct t? or raw t.. 
	history, action_score = score(history, df, t, args.batch_size, recs)
	
    # evaluate
	if action_score is not None:
		action_score = action_score.liked.tolist()
		rewards.extend(action_score)

	    # update the policy
	    model = retrain(model, history)

end = time.time()
print('finished in {} seconds'.format(end - start))

# save experiment results 
if args.bayesian=='TRUE':
	algorithm_version = 'bayesian_'
else:
	algorithm_version = 'ucb1_'

filename = algorithm_version + str(args.batch_size) + '_' + str(args.n) + '_' + str(args.ucb_scale) + '_' + str(args.min_review_count)
full_filename = args.result_dir + filename

print("saving results to {}".format(full_filename))

text = ['batch_size, slate_size, ucb_multiplier, min_reviews_per_movie, mean_reward, sum_reward, num_trials',
         '{}, {}, {}, {}, {}, {}, {}'.format(args.batch_size, args.n, args.ucb_scale, args.min_review_count, np.mean(rewards), np.sum(rewards), len(rewards))]

with open(full_filename + '.csv','w') as file:
    for line in text:
        file.write(line)
        file.write('\n')

with open(full_filename + '_raw.csv','w') as file:
	file.write(str(rewards))

cumulative_avg = np.cumsum(rewards) / np.linspace(1, len(rewards), len(rewards))
plt.plot(pd.Series(rewards).rolling(200).mean(), label='epsilon')
plt.plot(cumulative_avg, label='epsilon')
plt.savefig(full_filename + '_training_avg_reward.png', dpi = 300)

print(pd.Series(rewards).rolling(200).mean())
print(cumulative_avg)