#!/bin/bash
for i in {4..5}
do
   python sklearn-models.py --seed $i
done