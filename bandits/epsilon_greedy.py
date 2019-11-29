import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings

def score(history, df, t, batch_size, recs):
	# https://arxiv.org/pdf/1003.5956.pdf
	# replay score. reward if rec matches logged data, ignore otherwise
	actions = df[t:t+batch_size]
	actions = actions.loc[actions['movieId'].isin(recs)]
	# add row to history if recs match logging policy
	history = history.append(actions)
	action_liked = actions.liked.tolist()
	print('{} : {}'.format(t, t+batch_size))
	return history, action_liked

df = get_ratings(min_number_of_reviews=20000)

# initialize empty history 
# (offline eval means you can only add to history when rec matches historic data)
history = pd.DataFrame(data=None, columns=df.columns)
history = history.astype({'movieId': 'int32', 'liked': 'float'})

n = 5 # slate size
epsilon = .05 # explore rate

# to speed this up, retrain the bandit every batch_size time steps
# this lets us measure batch_size actions against a slate of recommendations rather than generating
#      recs at each time step. this seems like the only way to make it through a large dataset like
#      this and get a meaningful sample size with offline/replay evaluation
rewards = []
max_time = 100000 # total number of ratings to evaluate using the bandit
batch_size = 100 # number of ratings to observe for each iteration of the bandit before generating new recs
for t in range(max_time//batch_size): #df.t:
	t = t * batch_size
	# choose to explore epsilon % of the time 
	explore = np.random.binomial(1, epsilon)
	if explore == 1 or history.shape[0]==0:
		# shuffle movies to choose a random slate
		recs = np.random.choice(df.movieId.unique(), size=(n), replace=False)
	else:
		scores = history.loc[history.t<=t, ['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count']})
		scores.columns = ['mean', 'count']
		scores['movieId'] = scores.index
		scores = scores.sort_values('mean', ascending=False)
		recs = scores.loc[scores.index[0:n], 'movieId'].values
	history, action_score = score(history, df, t, batch_size, recs)
	if action_score is not None:
		rewards.extend(action_score)

# todo: need better ways to evaluate than this. somethign about smoothed take rates
# over time or batched take rate for every N trials.
# make utils for this so they can be consistent across algos
import matplotlib.pyplot as plt 
plt.plot(np.cumsum(rewards))
plt.show()