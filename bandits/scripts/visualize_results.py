import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import glob

path = r'/Users/jamesledoux/Documents/bandits/results' # use your path
all_files = glob.glob(path + "/*.csv")

all_dataframes = []

for filename in all_files:
    if '_raw' not in filename:
        df = pd.read_csv(filename, index_col=None, header=0)
        if 'epsilon_greedy' in filename:
            df['algorithm'] = 'epsilon_greedy'
        if 'ucb1' in filename:
            df['algorithm'] = 'ucb1'
        if 'bayesian' in filename:
            df['algorithm'] = 'bayesian_ucb'
        if 'exp3' in filename:
            df['algorithm'] = 'exp3'
        all_dataframes.append(df)

df = pd.concat(all_dataframes, axis=0, ignore_index=True)

epsilon = df.loc[(df['algorithm']=='epsilon_greedy') & (df['batch_size']==10000) & (df[' slate_size']==5)]
epsilon = epsilon.sort_values(' epsilon')
epsilon.plot(x=' epsilon', y=' mean_reward', kind='bar', title='Mean Reward by Epsilon Value',
        legend=False)
plt.tight_layout()
plt.savefig(path + '/epsilon_plot.png', dpi = 300)

exp = df.loc[(df['algorithm']=='exp3') & (df['batch_size']==10000) & (df[' slate_size']==5)]
exp = exp.sort_values(' gamma')
exp.plot(x=' gamma', y=' mean_reward', kind='bar', title='Mean Reward by UCB Scale Parameter',
        legend=False)
plt.tight_layout()
plt.savefig(path + '/exp_plot.png', dpi = 300)


ucb = df.loc[(df['algorithm']=='ucb1') & (df['batch_size']==10000) & (df[' slate_size']==5)]
ucb = ucb.sort_values(' ucb_multiplier')
ucb[' ucb_multiplier'] = 'UCB1'


ucb_bayes = df.loc[(df['algorithm']=='bayesian_ucb') & (df['batch_size']==10000) & (df[' slate_size']==5)]
ucb_bayes = ucb_bayes.sort_values(' ucb_multiplier')
ucb_bayes[' ucb_multiplier'] = ucb_bayes[' ucb_multiplier'].astype(str)
ucb = pd.concat([ucb, ucb_bayes], axis=0)
ucb.plot(x=' ucb_multiplier', y=' mean_reward', kind='bar', title='Mean Reward by UCB Scale Parameter',
        legend=False)
plt.tight_layout()
plt.savefig(path + '/ucb_plot.png', dpi = 300)



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,7), sharey=True, constrained_layout=True)
fig.suptitle('Parameter Tuning for Epsilon Greedy, EXP3, and UCB Bandits')
exp.plot(x=' gamma', y=' mean_reward', kind='bar', title='EXP3',
        legend=False, ax=axes[0], ylim=(0,.6))
epsilon.plot(x=' epsilon', y=' mean_reward', kind='bar', title='Epsilon Greedy',
        legend=False, ax=axes[1], ylim=(0,.6))
ucb.plot(x=' ucb_multiplier', y=' mean_reward', kind='bar', title='UCB (Bayesian and UCB1)',
        legend=False, ax=axes[2], ylim=(0,.6))
axes[0].set(ylabel='Mean Reward')
plt.rc('font', size=12)

plt.savefig(path + '/all_plots.png')

best_epsilon = '/Users/jamesledoux/Documents/bandits/results/epsilon_greedy_100_5_0.1_1500_raw.csv'
best_ucb = '/Users/jamesledoux/Documents/bandits/results/bayesian_100_5_1.5_1500_raw.csv'
best_exp = '/Users/jamesledoux/Documents/bandits/results/exp3_100_5_0.1_1500_raw.csv'

epsilon = genfromtxt(best_epsilon, delimiter=',')
ucb = genfromtxt(best_ucb, delimiter=',')
exp = genfromtxt(best_exp, delimiter=',')
epsilon = epsilon[~np.isnan(epsilon)]
ucb = ucb[~np.isnan(ucb)]
exp = exp[~np.isnan(exp)]


cumulative_epsilon = np.cumsum(epsilon) / np.linspace(1, len(epsilon), len(epsilon))
cumulative_ucb = np.cumsum(ucb) / np.linspace(1, len(ucb), len(ucb))
cumulative_exp = np.cumsum(exp) / np.linspace(1, len(exp), len(exp))

plt.plot(pd.Series(cumulative_epsilon).rolling(200).mean(), label='Epsilon Greedy')
plt.plot(pd.Series(cumulative_exp).rolling(200).mean(), label='EXP3')
plt.plot(pd.Series(cumulative_ucb).rolling(200).mean(), label='Bayesian UCB')
plt.title('200-Round Rolling Mean Reward')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.legend()
plt.savefig(path + '/trailing_average_rewards.png')
plt.clf()

cumulative_epsilon = np.cumsum(epsilon) 
cumulative_ucb = np.cumsum(ucb) 
cumulative_exp = np.cumsum(exp)

plt.plot(pd.Series(cumulative_epsilon), label='Epsilon Greedy')
plt.plot(pd.Series(cumulative_exp), label='EXP3')
plt.plot(pd.Series(cumulative_ucb), label='Bayesian UCB')

plt.title('Cumulative Reward Over Time')
plt.xlabel('Time Step')
plt.ylabel('Reward ("Liked" Movies)')
plt.legend()
plt.savefig(path + '/cumulative_rewards.png')





fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,9))

lw=2

cumulative_epsilon = np.cumsum(epsilon) 
cumulative_ucb = np.cumsum(ucb) 
cumulative_exp = np.cumsum(exp)

axes[1].plot(pd.Series(cumulative_epsilon), lw=lw, label='Epsilon Greedy')
axes[1].plot(pd.Series(cumulative_exp), lw=lw, label='EXP3')
axes[1].plot(pd.Series(cumulative_ucb), lw=lw, label='Bayesian UCB')

axes[1].set_title('Cumulative Reward Over Time')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Cumulative Reward ("Liked" Movies)')
axes[1].legend()


cumulative_epsilon = np.cumsum(epsilon) / np.linspace(1, len(epsilon), len(epsilon))
cumulative_ucb = np.cumsum(ucb) / np.linspace(1, len(ucb), len(ucb))
cumulative_exp = np.cumsum(exp) / np.linspace(1, len(exp), len(exp))

axes[0].plot(pd.Series(cumulative_epsilon).rolling(200).mean(), lw=lw, label='Epsilon Greedy')
axes[0].plot(pd.Series(cumulative_exp).rolling(200).mean(), lw=lw, label='EXP3')
axes[0].plot(pd.Series(cumulative_ucb).rolling(200).mean(), lw=lw, label='Bayesian UCB')
axes[0].set_title('200-Round Rolling Mean Reward')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Average Reward')
axes[0].legend()

fig.tight_layout()

plt.savefig(path + '/final_bandit_results.png')