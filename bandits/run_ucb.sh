#!/bin/sh
python3 ucb.py --n=5 --ucb_scale=.5 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=1 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=1.5 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=2 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=2.5 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'
python3 ucb.py --n=5 --ucb_scale=3 --batch_size=10000 --min_review_count=1500 --bayesian='TRUE'

python3 ucb.py --n=5 --batch_size=10000 --min_review_count=1500 --bayesian=FALSE

#python3 ucb.py --n=1 --ucb_scale=2 --batch_size=10000 --min_review_count=1500
#python3 ucb.py --n=3 --ucb_scale=2 --batch_size=10000 --min_review_count=1500
#python3 ucb.py --n=5 --ucb_scale=2 --batch_size=10000 --min_review_count=1500
#python3 ucb.py --n=7 --ucb_scale=2 --batch_size=10000 --min_review_count=1500
#python3 ucb.py --n=9 --ucb_scale=2 --batch_size=10000 --min_review_count=1500

#python3 ucb.py --n=5 --ucb_scale=1.96 --batch_size=100 --min_review_count=1500
#python3 ucb.py --n=5 --ucb_scale=1.96 --batch_size=1000 --min_review_count=1500
#python3 ucb.py --n=5 --ucb_scale=1.96 --batch_size=10000 --min_review_count=1500
