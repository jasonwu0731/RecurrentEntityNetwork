# Dialog State Tracking Challenge 6 (DSTC6) Track1 
# End-to-End Goal Oriented Dialog Learning
from Hong Kong University of Science and Technology(HKUST) Human Language Technology Center

## Setup
* Clone the repo and the dataset
* Run ```python REN.py --train --task=1``` to begin train on task 1
* Run ```python REN.py --train --task=1 --record``` to begin train on task 1 with recorded delexicalization data 
* Try ```--augment``` to increase the dataset by partial dialog

## Major Dependencies
- tensorflow

## Testing Sets for Competition 
1. uses the same KB as for the train dialogs, and the same set of slots in the queries
2. uses the different KB (with disjoint sets of restaurants, locations, cuisines, etc.), termed Out-Of-Vocabulary (OOV), but the same set of slots in the queries
3. uses the same KB as for the train dialogs, but one additional slot for the queries
4. uses the different KB (OOV) and an additional required slot

