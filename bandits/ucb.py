import sys
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt 
from utils import score
import argparse
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings_1m, get_ratings_20m

# command line args for experiment params
# example: python3 ucb.py --n=5 --ucb_scale=2 --batch_size=1000 --min_review_count=1500
parser = argparse.ArgumentParser()
parser.add_argument('--n', '--n', help="slate size (number of recs per iteration)", type= int, default= 5)
parser.add_argument('--ucb_scale', '--ucb_scale', help="scale factor for ucb calculation (1.96 is a 95 percent ucb)", type= float, default= 1.96)
parser.add_argument('--batch_size', '--batch_size', help="number of user sessions to observe for each iteration of the bandit", type= int, default= 10)
parser.add_argument('--min_review_count', '--min_review_count', help="number of reviews a movie needs to be in the dataset", type= int, default= 1500)
parser.add_argument('--balanced_classes', '--balanced_classes', help="T/F for whether each movie gets an equal number of ratings in the dataset", type= bool, default= True)
parser.add_argument('--result_dir', '--result_dir', help="directory for results to be saved", type= str, default= '/Users/jamesledoux/Documents/bandits/results/')
parser.add_argument('--verbose', '--verbose', help="TRUE if you want updates on training progress", type= str, default= 'TRUE')
parser.add_argument('--bayesian', '--bayesian', help="TRUE if you want to run bayesian ucb, false if ucb1", type= str, default= 'FALSE')

args = parser.parse_args()

def ucb1_policy(df, t, ucb_scale=2.0):
	'''
	Applies UCB1 policy to generate movie recommendations
	args:
		df: dataframe. Dataset to apply UCB policy to.
		ucb_scale: float. Most implementations use 2.0.
		t: int. represents the current time step.
	'''
	scores = df[['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count', 'std']})
	scores.columns = ['mean', 'count', 'std']
	if args.bayesian == 'TRUE':
		scores['ucb'] = scores['mean'] + (ucb_scale * scores['std'] / np.sqrt(scores['count']))
	else:
		scores['ucb'] = scores['mean'] + np.sqrt(
				(
					(2 * np.log10(t)) /
					scores['count']
				)
			)
	scores['movieId'] = scores.index
	scores = scores.sort_values('ucb', ascending=False)
	recs = scores.loc[scores.index[0:args.n], 'movieId'].values
	return recs

print("Running UCB Bandit with: batch size {}, slate size {}, ucb multiplier {}, bayesian: {}, and a minimum of {} reviews per movie in the dataset"\
	.format(args.batch_size, args.n, args.ucb_scale, args.bayesian, args.min_review_count))

df = get_ratings_20m(min_number_of_reviews=args.min_review_count, balanced_classes=args.balanced_classes)
print(df.shape)
print(len(df.movieId.unique()))

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
	recs = ucb1_policy(df=history.loc[history.t<=t,], t = t/args.batch_size, ucb_scale=args.ucb_scale) #is this the correct t? or raw t.. 
	history, action_score = score(history, df, t, args.batch_size, recs)
	if action_score is not None:
		action_score = action_score.liked.tolist()
		rewards.extend(action_score)

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