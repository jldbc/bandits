import numpy as np
from numpy.random import choice
import pandas as pd
import sys
import math
sys.path.insert(0, 'scripts/')
from create_movielens_dataset import get_ratings

# taking guidance from this https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/

def score(history, df, t, batch_size, recs):
	# https://arxiv.org/pdf/1003.5956.pdf
	# replay score. reward if rec matches logged data, ignore otherwise
	actions = df[t:t+batch_size]
	actions = actions.loc[actions['movieId'].isin(recs)]
	# add row to history if recs match logging policy
	history = history.append(actions)
	action_liked = actions[['movieId', 'liked']]
	#print('{} : {}'.format(t, t+batch_size))
	return history, action_liked

def distr(weights, gamma=0.0):
    weight_sum = float(sum(weights))
    id_mapping = {movieId: }
    return tuple((1.0 - gamma) * (w / weight_sum) + (gamma / len(weights)) for w in weights)


def draw(probability_distribution, n_recs=1):
	arm = choice(df.movieId.unique(), size=n_recs,
              p=probability_distribution, replace=False)
	return arm

def update_weights(weights, movieId_weight_mapping, probability_distribution, actions):
	# iter through actions. up to n updates / rec
	if actions.shape[0] == 0:
		return weights
	for a in range(actions.shape[0]):
		#actions = actions.reset_index()
		action = actions[a:a+1]
		weight_idx = movieId_weight_mapping[action.movieId.values[0]]
		estimated_reward = 1.0 * action.liked.values[0] / probability_distribution[weight_idx]
		weights[weight_idx] *= math.exp(estimated_reward * gamma / num_arms)
	return weights

df = get_ratings(min_number_of_reviews=20000)


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



n = 5 # slate size (it's more correct to server one at a time, but trains faster to recommend slates of top n)

rewards = []
num_arms = df.movieId.unique().shape[0]
max_time = 1000000 # total number of ratings to evaluate using the bandit
batch_size = 100 # number of ratings to observe for each iteration of the bandit before generating new recs
weights = [1.0] * df.movieId.unique().shape[0] # initialize one weight per arm
movieId_weight_mapping = dict(map(lambda t: (t[1], t[0]), enumerate(df.movieId.unique())))
gamma = .07
i = 1
for t in range(max_time//batch_size): #df.t:
	t = t * batch_size
	probability_distribution = distr(weights, gamma)
	recs = draw(probability_distribution, n_recs=n)
	history, action_score = score(history, df, t, batch_size, recs)
	weights = update_weights(weights, movieId_weight_mapping, probability_distribution, action_score)
	action_score = action_score.liked.tolist()
	#print(weights)
	rewards.extend(action_score)


print(np.mean(rewards))
print(len(rewards))

plt.plot(pd.Series(rewards).rolling(200).mean(), label='eps3')
plt.legend()
plt.show()