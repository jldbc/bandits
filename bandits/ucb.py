import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings

pd.set_option('display.float_format', lambda x: '%.3f' % x)

def score(history, df, t, batch_size, recs):
	# https://arxiv.org/pdf/1003.5956.pdf
	# replay score. reward if rec matches logged data, ignore otherwise
	actions = df[t:t+batch_size]
	actions = actions.loc[actions['movieId'].isin(recs)]
	# add row to history if recs match logging policy
	history = history.append(actions)
	action_liked = actions.liked.tolist()
	#print('{} : {}'.format(t, t+batch_size))
	return history, action_liked

df = get_ratings(min_number_of_reviews=20000)


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



n = 5 # slate size (it's more correct to server one at a time, but trains faster to recommend slates of top n)

# to speed this up, retrain the bandit every batch_size time steps
# this lets us measure batch_size actions against a slate of recommendations rather than generating
#      recs at each time step. this seems like the only way to make it through a large dataset like
#      this and get a meaningful sample size with offline/replay evaluation

ucb_history = pd.DataFrame(data=None, columns = ['mean', 'count', 'std', 'ucb', 'movieId', 'iter']) # for post-analysis of ucbs over iterations
rewards = []
ucb_checkpoints = []
max_time = 500000 # total number of ratings to evaluate using the bandit
batch_size = 100 # number of ratings to observe for each iteration of the bandit before generating new recs
i = 1
for t in range(max_time//batch_size): #df.t:
	t = t * batch_size
	scores = history.loc[history.t<=t, ['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count', 'std']})
	#scores['std'] = scores['std'] + .0000001 # to avoid dividing by zero in ucb calculation
	# sqrt(2 * np.log(1/var)/count)
	scores.columns = ['mean', 'count', 'std']
	scores['ucb'] = scores['mean'] + (1.96 * scores['std'] / np.sqrt(scores['count'])) #np.sqrt((2 * np.log(1/scores['std']))/scores['count'])
	scores['movieId'] = scores.index
	scores = scores.sort_values('ucb', ascending=False)
	recs = scores.loc[scores.index[0:n], 'movieId'].values
	history, action_score = score(history, df, t, batch_size, recs)
	if action_score is not None:
		rewards.extend(action_score)
	if(t%1000==0):
		print(t)
	#if(t%10000==0):
	#	scores['iter'] = i
	#	ucb_history = ucb_history.append(scores)
	#i += 1


# todo: need better ways to evaluate than this. somethign about smoothed take rates
# over time or batched take rate for every N trials.
# make utils for this so they can be consistent across algos
import matplotlib.pyplot as plt 
# plot cumulative gain
#plt.plot(np.cumsum(rewards), label='.05')
#plt.legend()
#plt.show()

# visualize mean/ucb movemebt for a movie across training iterations
'''
plt.plot(ucb_history.loc[ucb_history.movieId==2959, 'iter'], ucb_history.loc[ucb_history.movieId==2959, 'mean'])
plt.plot(ucb_history.loc[ucb_history.movieId==2959, 'iter'], ucb_history.loc[ucb_history.movieId==2959, 'ucb'])
plt.show()

plt.plot(ucb_history.loc[ucb_history.movieId==2997, 'iter'], ucb_history.loc[ucb_history.movieId==2997, 'mean'])
plt.plot(ucb_history.loc[ucb_history.movieId==2997, 'iter'], ucb_history.loc[ucb_history.movieId==2997, 'ucb'])
plt.plot(ucb_history.loc[ucb_history.movieId==750, 'iter'], ucb_history.loc[ucb_history.movieId==750, 'mean'])
plt.plot(ucb_history.loc[ucb_history.movieId==750, 'iter'], ucb_history.loc[ucb_history.movieId==750, 'ucb'])
plt.show()
'''

plt.plot(pd.Series(rewards).rolling(200).mean(), label='ucb')
plt.legend()
plt.show()



print(np.mean(rewards))
print(len(rewards))