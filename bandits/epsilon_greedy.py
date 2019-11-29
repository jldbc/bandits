import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings

def score(history, df, t, recs):
	# https://arxiv.org/pdf/1003.5956.pdf
	# replay score. reward if rec matches logged data, ignore otherwise
	action_movieId = df.at[t, 'movieId']
	action_liked = df.at[t, 'liked']
	if action_movieId not in recs:
		return history, None
	# add row to history if recs match logging policy
	history = history.append(df.iloc[t])
	return history, action_liked

df = get_ratings(min_number_of_reviews=20000)

# initialize empty history 
# (offline eval means you can only add to history when rec matches historic data)
history = pd.DataFrame(data=None, columns=df.columns)
history = history.astype({'movieId': 'int32', 'liked': 'float'})

n = 5 # slate size
epsilon = .05 # explore rate

rewards = []
# TODO: only add to history when it's a matched action. otherwise doesn't reflect the policy you're learning
for t in range(10000): #df.t:
	# choose to explore epsilon % of the time 
	explore = np.random.binomial(1, epsilon)
	if explore == 1 or history.shape[0]==0:
		# shuffle movies to choose a random slate
		recs = np.random.choice(df.movieId.unique(), size=(n,1), replace=False)
	else:
		scores = history.loc[history.t<=t, ['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count']})
		scores.columns = ['mean', 'count']
		scores['movieId'] = scores.index
		scores = scores.sort_values('mean', ascending=False)
		recs = scores.loc[scores.index[0:n], 'movieId'].values
	action = df[t:t+1][['movieId', 'liked']]
	history, action_score = score(history, df, t, recs)
	if action_score is not None:
		rewards.append(action_score)
	print(t)

import matplotlib.pyplot as plt 
plt.plot(np.cumsum(rewards))
plt.plot(np.cumsum(rewards1))
plt.show()