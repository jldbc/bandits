import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from utils import score
import argparse
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings_1m, get_ratings_20m


# command line args for experiment params
# example: python3 epsilon_greedy.py --n=5 --epsilon=0.15 --batch_size=1000 --min_review_count=1500
parser = argparse.ArgumentParser()
parser.add_argument('--n', '--n', help="slate size (number of recs per iteration)", type= int, default= 5)
parser.add_argument('--epsilon', '--epsilon', help="scale factor for ucb calculation (1.96 is a 95 percent ucb)", type= float, default= 0.15)
parser.add_argument('--batch_size', '--batch_size', help="number of user sessions to observe for each iteration of the bandit", type= int, default= 10)
parser.add_argument('--min_review_count', '--min_review_count', help="number of reviews a movie needs to be in the dataset", type= int, default= 1500)
parser.add_argument('--balanced_classes', '--balanced_classes', help="T/F for whether each movie gets an equal number of ratings in the dataset", type= bool, default= True)
parser.add_argument('--result_dir', '--result_dir', help="number of reviews a movie needs to be in the dataset", type= str, default= '/Users/jamesledoux/Documents/bandits/results/')
parser.add_argument('--verbose', '--verbose', help="TRUE if you want updates on training progress", type= str, default= 'TRUE')

args = parser.parse_args()

def epsilon_greedy_policy(df, arms, epsilon=0.15, slate_size=5, batch_size=50):
    '''
    df: dataset to apply the policy to
    epsilon: float. represents the % of timesteps where we explore random arms
    slate_size: int. the number of recommendations to make at each step.
    batch_size: int. the number of users to serve these recommendations to before updating the bandit's policy.
    '''
    # draw a 0 or 1 from a binomial distribution, with epsilon% likelihood of drawing a 1
    explore = np.random.binomial(1, epsilon)
    # if explore: shuffle movies to choose a random set of recommendations
    if explore == 1 or df.shape[0]==0:
        recs = np.random.choice(arms, size=(slate_size), replace=False)
    # if exploit: sort movies by "like rate", recommend movies with the best performance so far
    else:
        scores = df[['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count']})
        scores.columns = ['mean', 'count']
        scores['movieId'] = scores.index
        scores = scores.sort_values('mean', ascending=False)
        recs = scores.loc[scores.index[0:slate_size], 'movieId'].values
    return recs

print("Running Epsilon Greedy Bandit with: batch size {}, slate size {}, epsilon {}, and a minimum of {} reviews per movie in the dataset"\
	.format(args.batch_size, args.n, args.epsilon, args.min_review_count))

df = get_ratings_20m(min_number_of_reviews=args.min_review_count, balanced_classes=args.balanced_classes)

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
	if t % 100000 == 0:
		if args.verbose == 'TRUE':
			print(t)
	# choose which arm to pull
	recs = epsilon_greedy_policy(df=history.loc[history.t<=t,], arms=df.movieId.unique(), epsilon=args.epsilon, slate_size=args.n, batch_size=args.batch_size)
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