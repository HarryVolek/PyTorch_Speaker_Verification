# PyTorch_Speaker_Verification

PyTorch implementation of speech embedding net and loss described here: https://arxiv.org/pdf/1710.10467.pdf

![training loss](https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/Results/Loss.png)

The TIMIT speech corpus was used to train the model, found here: https://catalog.ldc.upenn.edu/LDC93S1,
or here, https://github.com/philipperemy/timit

# Preprocessing

Change the following config.yaml key to a regex containing all .WAV files in your downloaded TIMIT dataset.
```yaml
unprocessed_data: './TIMIT/*/*/*/*.WAV'
```
Run the preprocessing script:
```
./data_preprocess.py 
```
Two folders will be created, train_tisv and test_tisv, containing .npy files containing numpy ndarrays of speaker utterances with a 90%/10% training/testing split.

# Training

To train the speaker verification model, run:
```
./train_speech_embedder.py 
```
with the following config.yaml key set to true:
```yaml
training: !!bool "true"
```
for testing, set the key value to:
```yaml
training: !!bool "false"
```
The log file and checkpoint save locations are controlled by the following values:
```yaml
log_file: './speech_id_checkpoint/Stats'
checkpoint_dir: './speech_id_checkpoint'
```
Only TI-SV is implemented.

# Performance

```
EER across 10 epochs: 0.0518
```

# Disclaimer

The embeddings produced by this project are NOT currently compatible with https://github.com/google/uis-rnn
