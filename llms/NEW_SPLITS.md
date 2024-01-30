# Naive Random Unique
* Concat `naive_random` dataset's train,val, and test splits
* random shuffle
* drop duplicates in the targets
* split again into 85% train, and 15% test

Output statistics:
```json
DatasetDict({
    train: Dataset({
        features: ['labels', 'clue'],
        num_rows: 47844
    })
    test: Dataset({
        features: ['labels', 'clue'],
        num_rows: 8444
    })
})
```

# Word Init Disjoint Unique
* Concat `word_init_disjoint` dataset's train,and val splits to create `concat_train_val_disjoint` (this will be the new `train` split)
* random shuffle
* drop duplicates in the targets for `concat_train_val_disjoint`, and `word_init_disjoint` test split

Output statistics:
```json
DatasetDict({
    train: Dataset({
        features: ['labels', 'clue'],
        num_rows: 42793
    })
    test: Dataset({
        features: ['clue', 'labels'],
        num_rows: 13495
    })
})
```

# Word Init Disjoint Half 
* Concat `word_init_disjoint` dataset's train,and val splits to create `concat_train_val_disjoint` (this will be the new `train` split)
* random shuffle
* take only the first 51% of the targets for `concat_train_val_disjoint`, and `word_init_disjoint` test split
* we make it 51% to make sure we take the targets if it occur only one time (a way to do rounding up)
Output statistics:
```json
DatasetDict({
    train: Dataset({
        features: ['labels', 'clue'],
        num_rows: 69339
    })
    test: Dataset({
        features: ['clue', 'labels'],
        num_rows: 21707
    })
})
```



# Sanity checks:
* confirmed that the length of the `naive_random_unique` is the same as calculating the length of the unique entires in `naive_random` dataset
* confirmed that that length of the `word_init_disjoint_unique` train, and test sets are the same for  the concatination of `word_init_disjoint` train and dev, and the test splits
* confirmed that the `word_init_disjoint_half` has an average of **0.48%** for the targets in the original `word_init_disjoint`  

## Notes
* we concat the train, and val splits together, as we don't do any hyperparameters tuning, so we can utilize the extra examples in the dev set
* the intuition about creating `word_init_disjoint_unique` is to keep targets that share the same first two letters, in the same splits.