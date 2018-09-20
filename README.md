# PyTorch_Speaker_Verification

PyTorch implementation of speech embedding net and loss described here: https://arxiv.org/pdf/1710.10467.pdf

![training loss](https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/Results/Loss.png)

The TIMIT speech corpus was used to train the model, found here: https://catalog.ldc.upenn.edu/LDC93S1,
or here, https://github.com/philipperemy/timit

The command line arguments for usage are found in configuration.py. To use the model, run train_speech_embedder.py
with the training flag and either a regex for the unpreprocessed input wav files or the path to the directory created by
the data_preprocess.py script. The latter is recommended.

To preprocess the TIMIT wav files, place the files into the folder and run data_preprocess.py.

Only TI-SV is implemented.
