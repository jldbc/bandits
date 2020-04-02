import numpy as np
from numpy.random import choice
import pandas as pd
import sys
import math
from utils import score
import matplotlib.pyplot as plt
import argparse
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings_1m, get_ratings_20m

# taking guidance from this https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/

# command line args for experiment params
# example: python3 exp3.py --n=5 --gamma=0.07 --batch_size=100 --min_review_count=1500
parser = argparse.ArgumentParser()
parser.add_argument('--n', '--n', help="slate size (number of recs per iteration)", type= int, default= 5)
parser.add_argument('--gamma', '--gamma', help="scale factor for ucb calculation (1.96 is a 95 percent ucb)", type= float, default= 0.7)
parser.add_argument('--batch_size', '--batch_size', help="number of user sessions to observe for each iteration of the bandit", type= int, default= 10)
parser.add_argument('--min_review_count', '--min_review_count', help="number of reviews a movie needs to be in the dataset", type= int, default= 1500)
parser.add_argument('--balanced_classes', '--balanced_classes', help="T/F for whether each movie gets an equal number of ratings in the dataset", type= bool, default= True)
parser.add_argument('--result_dir', '--result_dir', help="number of reviews a movie needs to be in the dataset", type= str, default= '/Users/jamesledoux/Documents/bandits/results/')
parser.add_argument('--verbose', '--verbose', help="TRUE if you want updates on training progress", type= str, default= 'TRUE')

args = parser.parse_args()

print("Running EXP3 Bandit with: batch size {}, slate size {}, gamma {}, and a minimum of {} reviews per movie in the dataset"\
	.format(args.batch_size, args.n, args.gamma, args.min_review_count))

df = get_ratings_20m(min_number_of_reviews=args.min_review_count, balanced_classes=args.balanced_classes)


def distr(weights, gamma=0.0):
    weight_sum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / weight_sum) + (gamma / len(weights)) for w in weights)

def draw(probability_distribution, n_recs=1):
    arm = choice(df.movieId.unique(), size=n_recs,
        p=probability_distribution, replace=False)
    return arm

def update_weights(weights, gamma, movieId_weight_mapping, probability_distribution, actions):
    # iter through actions. up to n updates / rec
    if actions.shape[0] == 0:
        return weights
    for a in range(actions.shape[0]):
        action = actions[a:a+1]
        weight_idx = movieId_weight_mapping[action.movieId.values[0]]
        estimated_reward = 1.0 * action.liked.values[0] / probability_distribution[weight_idx]
        weights[weight_idx] *= math.exp(estimated_reward * gamma / num_arms)
    return weights

def exp3_policy(df, history, t, weights, movieId_weight_mapping, gamma, n_recs, batch_size):
    '''
    Applies EXP3 policy to generate movie recommendations
    Args:
        df: dataframe. Dataset to apply EXP3 policy to
        history: dataframe. events that the offline bandit has access to (not discarded by replay evaluation method)
        t: int. represents the current time step.
        weights: array or list. Weights used by EXP3 algorithm.
        movieId_weight_mapping: dict. Maping between movie IDs and their index in the array of EXP3 weights.
        gamma: float. hyperparameter for algorithm tuning.
        n_recs: int. Number of recommendations to generate in each iteration. 
        batch_size: int. Number of observations to show recommendations to in each iteration.
    '''
    probability_distribution = distr(weights, gamma)
    recs = draw(probability_distribution, n_recs=n_recs)
    history, action_score = score(history, df, t, batch_size, recs)
    weights = update_weights(weights, gamma, movieId_weight_mapping, probability_distribution, action_score)
    action_score = action_score.liked.tolist()
    return history, action_score, weights

# vv don't need this, should swap it out for the epsilon greedy initialization approach 
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
history = history.append(history)
history = history.append(history2)
history = history.append(history2)
history = history.append(history)
#history = history.append(history)
#history = history.append(history)


rewards = []
num_arms = df.movieId.unique().shape[0]
max_time = df.shape[0] # total number of ratings to evaluate using the bandit
weights = [1.0] * df.movieId.unique().shape[0] # initialize one weight per arm
movieId_weight_mapping = dict(map(lambda t: (t[1], t[0]), enumerate(df.movieId.unique())))
i = 1
for t in range(max_time//args.batch_size): #df.t:
	t = t * args.batch_size
	if t % 100000 == 0:
		if args.verbose == 'TRUE':
			print(t)
	history, action_score, weights = exp3_policy(df, history, t, weights, movieId_weight_mapping, args.gamma, args.n, args.batch_size)	
	rewards.extend(action_score)


# save experiment results 
filename = 'exp3_' + str(args.batch_size) + '_' + str(args.n) + '_' + str(args.gamma) + '_' + str(args.min_review_count)
full_filename = args.result_dir + filename

print("saving results to {}".format(full_filename))

text = ['batch_size, slate_size, gamma, min_reviews_per_movie, mean_reward, sum_reward, num_trials',
         '{}, {}, {}, {}, {}, {}, {}'.format(args.batch_size, args.n, args.gamma, args.min_review_count, np.mean(rewards), np.sum(rewards), len(rewards))]

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