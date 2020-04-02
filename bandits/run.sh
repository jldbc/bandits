#!/bin/sh

# epsilon greedy parameter tuning
python3 epsilon_greedy.py --n=5 --epsilon=.01 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.05 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.1 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.15 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.25 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.33 --batch_size=10000 --min_review_count=1500

# ucb parameter tuning
python3 ucb.py --n=5 --ucb_scale=.5 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=1 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=1.5 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=2 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=2.5 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=3 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --batch_size=10000 --min_review_count=1500 --bayesian=FALSE

# exp3 parameter tuning
python3 exp3.py --n=5 --gamma=.01 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.05 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.1 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.15 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.25 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.33 --batch_size=10000 --min_review_count=1500

# final comparsons
python3 ucb.py --n=5 --ucb_scale=1.5 --batch_size=100 --min_review_count=1500 --bayesian='TRUE'
python3 exp3.py --n=5 --gamma=.1 --batch_size=100 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.1 --batch_size=100 --min_review_count=1500

# produce figures used in blog post
python3 visualize_results.py
