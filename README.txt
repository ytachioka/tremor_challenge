## step0: data linking 
Link the raw data file and label file based on the time stamps

# for training data
`cd data/TrainingDataPD25`
# modified labels are produced
`python mod_labels.py`
# data with labels are generated
`python gen_data.py`

# for test data
`cd data/TestData`
# the same process is needed
`python mod_labels.py`
`python gen_data.py`


## step1: pretrain conditional variational autoencoder for data augmentation
`python pretrain_autoencoder.py` with different seeds

## step2: evoluationary computation to find the best parameters
`python run_opt.py` for each seed
This script runs data generation (3_gen_data_prop.py) and human activity recogntion (HAR) model training (2_train_evaluateHAR.py) internally to find the best settings.

## step3: reproduce results with the best parameters
`python rep_opt.py` for each seed

## step4: to submit papers, labels are predicted by combining three HAR models trained on the original and augmented data
`python 4_evaluateHAR.py`

## step5: to assign multiple labels for events at duplicate time stamps
`python 5_make_label.py`

