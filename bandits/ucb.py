import sys
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt 
from utils import score
import argparse
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings_1m

# command line args for experiment params
# example: python3 ucb.py --n=5 --ucb_scale=1.96 --batch_size=100 --min_review_count=1500
parser = argparse.ArgumentParser()
parser.add_argument('--n', '--n', help="slate size (number of recs per iteration)", type= int, default= 5)
parser.add_argument('--ucb_scale', '--ucb_scale', help="scale factor for ucb calculation (1.96 is a 95 percent ucb)", type= float, default= 1.96)
parser.add_argument('--batch_size', '--batch_size', help="number of user sessions to observe for each iteration of the bandit", type= int, default= 10)
parser.add_argument('--min_review_count', '--min_review_count', help="number of reviews a movie needs to be in the dataset", type= int, default= 1500)
parser.add_argument('--result_dir', '--result_dir', help="number of reviews a movie needs to be in the dataset", type= str, default= '/Users/jamesledoux/Documents/bandits/results/')

args = parser.parse_args()

print("Running UCB1 Bandit with: batch size {}, slate size {}, ucb multiplier {}, and a minimum of {} reviews per movie in the dataset"\
	.format(args.batch_size, args.n, args.ucb_scale, args.min_review_count))

df = get_ratings_1m(min_number_of_reviews=args.min_review_count)

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
for t in range(max_time//args.batch_size): #df.t:
	t = t * args.batch_size
	scores = history.loc[history.t<=t, ['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count', 'std']})
	scores.columns = ['mean', 'count', 'std']
	# todo: test this, this is the version i see more people using  (ucb just a function of how many rounds deep vs. number of pulls for each arm )
	# scores['mean'] + math.sqrt((2 * math.log(t/args.batch_size)) / float(scores['count]))
	scores['ucb'] = scores['mean'] + (args.ucb_scale * scores['std'] / np.sqrt(scores['count']))
	scores['movieId'] = scores.index
	scores = scores.sort_values('ucb', ascending=False)
	recs = scores.loc[scores.index[0:args.n], 'movieId'].values
	history, action_score = score(history, df, t, args.batch_size, recs)
	if action_score is not None:
		action_score = action_score.liked.tolist()
		rewards.extend(action_score)


# save experiment results 
filename = 'ucb1_' + str(args.batch_size) + '_' + str(args.n) + '_' + str(args.ucb_scale) + '_' + str(args.min_review_count)
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