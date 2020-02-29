import numpy as np
import pandas as pd

def score(history, df, t, batch_size, recs):
    # https://arxiv.org/pdf/1003.5956.pdf
    # replay score. reward if rec matches logged data, ignore otherwise
    #actions = df.copy()[t:t+batch_size]
    actions = df[t:t+batch_size]
    actions = actions.loc[actions['movieId'].isin(recs)]
    actions['scoring_round'] = t
    # add row to history if recs match logging policy
    history = history.append(actions)
    action_liked = actions[['movieId', 'liked']]
    return history, action_liked

def summarise():
    '''
    summarise model performance
    - trailing avg. reward
    - cumulative avg. reward
    - total reward 
    - number of trials
    '''
    pass
