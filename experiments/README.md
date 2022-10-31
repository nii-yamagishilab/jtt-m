# jtt-m experiments

## Prepare the data

```bash
cd data
sh 01_prepare.sh
cd ..
```

The script `01_prepare.sh` downloads [Schuster et al. (2021)](https://arxiv.org/abs/2103.08541)'s [preprocessing](https://github.com/TalSchuster/talschuster.github.io/tree/master/static) of the FEVER and MultiNLI datasets. After the files are downloaded and unzipped, you should see something like

```
+ wc -l fever/train.jsonl fever/dev.jsonl fever/test.jsonl
  178059 fever/train.jsonl
   11620 fever/dev.jsonl
   11710 fever/test.jsonl
  201389 total

+ python ../../group_stats.py fever/train.jsonl
S & 99303 (61.5) & 1267 (7.7) & 100570 (56.5)
R & 27575 (17.1) & 14275 (86.3) & 41850 (23.5)
N & 34633 (21.4) & 1006 (6.1) & 35639 (20.0)
```

for FEVER. For MultiNLI, you will see something like 

```
+ wc -l mnli/train.jsonl mnli/dev.jsonl mnli/test.jsonl
   392702 mnli/train.jsonl
     9832 mnli/dev.jsonl
     9832 mnli/test.jsonl
   412366 total
+ python ../../group_stats.py mnli/train.jsonl
S & 118554 (36.7) & 12345 (17.7) & 130899 (33.3)
R & 88180 (27.3) & 42723 (61.2) & 130903 (33.3)
N & 116185 (36.0) & 14715 (21.1) & 130900 (33.3)
```



## Reproduce JTT-m

### Step 1: Do the first training to obtain the error set

Below we go through the procedure for the  **FEVER** experiments. The process for the MNLI experiments is the same, except that we only evaluate on MultiNLI's test set, since its development set and test set are identical in this preprocessing version.

To train the model:

```bash
cd fever+sgd
sh 01_train.sh
```
Note that the script contains SLURM directives that specify our GPU resource requirements for the `sbtach` command, but it can also be run with the `sh` command.

### Step 2: Prepare filtered error set for upweighting (JTT)

After training is finished, run

```bash
sh 02_predict.sh
```

This step 

(a) makes predictions on the training set and saves the penultimate-layer embeddings of the training set for outlier removal later, and 

(b) makes predictions on the development and test set (if the test set is available). 

You can view the prediction results in `bert-base-uncased-128-out/eval.{train,dev,test}.txt`, and the group accuracies in  `bert-base-uncased-128-out/eval.groups.{train,dev,test}.txt` .

The results should look something like this (showing the dev set results):

```

tail -n 11 bert-base-uncased-128-out/eval.dev.txt

S     R     N
S  3788   118    58
R   705  2574  1044
N   530   494  2309

          S     R     N
Prec:  75.4  80.8  67.7
Rec:   95.6  59.5  69.3
F1:    84.3  68.6  68.5

Acc: 74.6
```

```
tail bert-base-uncased-128-out/eval.groups.dev.txt

Total 11620, correct 8671, wrong 2949
Avg acc: 74.6 (8671/11620)
Worst group acc: 14.0
(S, no neg): 96.0 (3777/3934)
(S, neg): 36.7 (11/30)
(R, no neg): 43.7 (1339/3067)
(R, neg): 98.3 (1235/1256)
(N, no neg): 70.9 (2296/3240)
(N, neg): 14.0 (13/93)
```

Once all the predictions are finished, run

```bash
sh 03_calc_mahal.sh 
```

(This script does not require a GPU.)

This calculates the Mahalanobis distances of the penultimate layer embeddings and saves the distances calculated in `bert-base-uncased-128-out/train.mahal.npy`.

### Step 3: Upweight examples

Run

```bash
sh 04_augment.sh
```

(This script does not require a GPU.)

This script upweights the training data in two different ways, and the upweighted training data are saved in corresponding subfolders (with the same name as their experiment folders) in `../data`:

1. **JTT-m**: Upweights incorrectly-predicted training set examples with outliers removed (by the Mahalanobis distance method) from the error set
    
    The folder is `fever_sgd_df5_up3`, meaning it uses `sgd` in the ERM training, filters the error set from the ERM training with degree of freedom `df` 5, and upweights (`up`) the filtered error set for `3` times. The second training uses `adamw` as its optimizer.
    
2. **JTT**: Upweights incorrectly-predicted training set examples
    
    Its folder is `fever_sgd_thres1.0_up3`. `thres1.0` sets the threshold to `1.0` for filtering out incorrect examples by their predicted probabilities. The default threshold is 1.0, so no examples will be filtered out for JTT. The examples are upweighted `3` times.


    

### Step 4: Retrain the model on upweighted training set

We use `fever_sgd_df5_up3+adamw` (the JTT-m experiment) as an example. The procedure is the same for `fever_sgd_thres1.0_up3+adamw` (the JTT experiment).

```bash
cd ..
cd fever_sgd_df5_up3+adamw
sh 01_train.sh
```

Once the training is finished, run

```bash
sh 02_predict.sh
```

The prediction results will be saved in `bert-base-uncased-128-out` as `eval.{dev,test}.txt` and `eval.groups.{dev,test}.txt` . You should see something like this (using the dev set as example):

```
==> bert-base-uncased-128-out/eval.dev.txt <==
S  3763   109    92
R   250  3748   325
N   257   359  2717

          S     R     N
Prec:  88.1  88.9  86.7
Rec:   94.9  86.7  81.5
F1:    91.4  87.8  84.0

Acc: 88.0

==> bert-base-uncased-128-out/eval.groups.dev.txt <==
Total 11620, correct 10228, wrong 1392
Avg acc: 88.0 (10228/11620)
Worst group acc: 50.0
(S, no neg): 95.3 (3748/3934)
(S, neg): 50.0 (15/30)
(R, no neg): 83.3 (2554/3067)
(R, neg): 95.1 (1194/1256)
(N, no neg): 82.2 (2664/3240)
(N, neg): 57.0 (53/93)
```

## Reproduce ERM results

The ERM models are trained with a different optimizer than the first training in Step 1. Instead of using SGD, AdamW is used. The training and prediction scripts are in `fever+adamw` and `mnli+adamw`. 

To train, run 

```
sh 01_train.sh
```

Run 

```
sh 02_predict.sh
```

to obtain ERM model results for the test set (and dev set, if available).

## Model checkpoints
We also release model checkpoints (as well as example outputs and preprocessed data) at: https://doi.org/10.5281/zenodo.7260028.
