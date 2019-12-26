import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from utils import score
import argparse
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings_1m


# command line args for experiment params
# example: python3 epsilon_greedy.py --n=5 --epsilon=0.15 --batch_size=100 --min_review_count=1500
parser = argparse.ArgumentParser()
parser.add_argument('--n', '--n', help="slate size (number of recs per iteration)", type= int, default= 5)
parser.add_argument('--epsilon', '--epsilon', help="scale factor for ucb calculation (1.96 is a 95 percent ucb)", type= float, default= 0.15)
parser.add_argument('--batch_size', '--batch_size', help="number of user sessions to observe for each iteration of the bandit", type= int, default= 10)
parser.add_argument('--min_review_count', '--min_review_count', help="number of reviews a movie needs to be in the dataset", type= int, default= 1500)
parser.add_argument('--result_dir', '--result_dir', help="number of reviews a movie needs to be in the dataset", type= str, default= '/Users/jamesledoux/Documents/bandits/results/')

args = parser.parse_args()

print("Running UCB1 Bandit with: batch size {}, slate size {}, epsilon {}, and a minimum of {} reviews per movie in the dataset"\
	.format(args.batch_size, args.n, args.epsilon, args.min_review_count))

df = get_ratings_1m(min_number_of_reviews=args.min_review_count)

# initialize empty history 
# (offline eval means you can only add to history when rec matches historic data)
history = pd.DataFrame(data=None, columns=df.columns)
history = history.astype({'movieId': 'int32', 'liked': 'float'})

# to speed this up, retrain the bandit every batch_size time steps
# this lets us measure batch_size actions against a slate of recommendations rather than generating
#      recs at each time step. this seems like the only way to make it through a large dataset like
#      this and get a meaningful sample size with offline/replay evaluation
rewards = []
max_time = df.shape[0] # total number of ratings to evaluate using the bandit
for t in range(max_time//args.batch_size): #df.t:
	t = t * args.batch_size
	# choose to explore epsilon % of the time 
	explore = np.random.binomial(1, args.epsilon)
	if explore == 1 or history.shape[0]==0:
		# shuffle movies to choose a random slate
		recs = np.random.choice(df.movieId.unique(), size=(args.n), replace=False)
	else:
		scores = history.loc[history.t<=t, ['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count']})
		scores.columns = ['mean', 'count']
		scores['movieId'] = scores.index
		scores = scores.sort_values('mean', ascending=False)
		recs = scores.loc[scores.index[0:args.n], 'movieId'].values
	history, action_score = score(history, df, t, args.batch_size, recs)
	if action_score is not None:
		action_score = action_score.liked.tolist()
		rewards.extend(action_score)


# save experiment results 
filename = 'epsilon_greedy_' + str(args.batch_size) + '_' + str(args.n) + '_' + str(args.epsilon) + '_' + str(args.min_review_count)
full_filename = args.result_dir + filename

print("saving results to {}".format(full_filename))

text = ['batch_size, slate_size, epsilon, min_reviews_per_movie, mean_reward, sum_reward, num_trials',
         '{}, {}, {}, {}, {}, {}, {}'.format(args.batch_size, args.n, args.epsilon, args.min_review_count, np.mean(rewards), np.sum(rewards), len(rewards))]

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