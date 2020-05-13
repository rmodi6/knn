# k Nearest Neighbors
kNN classification algorithm implemented in Python.

### Usage
```bash
usage: knn.py [-h] --dataset DATASET_PATH [--k K] [--test_size TEST_SIZE]
              [--random_state RANDOM_STATE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET_PATH
                        path to dataset
  --k K                 k value for kNN (Default: 3)
  --test_size TEST_SIZE
                        size of test data in fraction (Default: 0.2)
  --random_state RANDOM_STATE
                        random state for train_test_split (Default: None)
```
### Example
```bash
python knn.py --dataset dataset/Breast_cancer_data.csv --k 3 --test_size 0.2 --random_state 10
```