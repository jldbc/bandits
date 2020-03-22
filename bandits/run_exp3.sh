#!/bin/sh
python3 exp3.py --n=5 --gamma=.01 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.05 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.1 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.15 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.25 --batch_size=10000 --min_review_count=1500
python3 exp3.py --n=5 --gamma=.33 --batch_size=10000 --min_review_count=1500