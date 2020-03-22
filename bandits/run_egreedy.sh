#!/bin/sh
python3 epsilon_greedy.py --n=5 --epsilon=.01 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.05 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.1 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.15 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.25 --batch_size=10000 --min_review_count=1500
python3 epsilon_greedy.py --n=5 --epsilon=.33 --batch_size=10000 --min_review_count=1500