#Modified from https://github.com/JanhHyun/Speaker_Verification
import argparse

parser = argparse.ArgumentParser()    # make parser

# get arguments
def get_config():
    config, unparsed = parser.parse_known_args()
    return config

# return bool type of argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Data Preprocess Arguments
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--train_path', type=str, default='./train_tisv', help="train dataset directory")
data_arg.add_argument('--train_path_unprocessed', type=str, default='./TIMIT/TRAIN/*/*/*.WAV', 
                      help="Regex for all wav files")
data_arg.add_argument('--test_path', type=str, default='./test_tisv', help="test dataset directory")
data_arg.add_argument('--test_path_unprocessed', type=str, default='./TIMIT/TEST/*/*/*.WAV', 
                      help="test dataset directory")
data_arg.add_argument('--sr', type=int, default=16000, help="sampling rate")
data_arg.add_argument('--nfft', type=int, default=512, help="fft kernel size")
data_arg.add_argument('--window', type=int, default=0.025, help="window length (ms)")
data_arg.add_argument('--hop', type=int, default=0.01, help="hop size (ms)")
data_arg.add_argument('--n_mels', type=int, default=40, help="Number of mel energies")
data_arg.add_argument('--tisv_frame', type=int, default=180, help="max frame number of utterances of tisv")

# Model Parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--hidden', type=int, default=128, help="hidden state dimension of lstm")
model_arg.add_argument('--proj', type=int, default=64, help="projection dimension of lstm")
model_arg.add_argument('--num_layer', type=int, default=3, help="number of lstm layers")
model_arg.add_argument('--restore', type=str2bool, default=False, help="restore model or not")
model_arg.add_argument('--model_path', type=str, default='./model.model', help="model directory to save or load")
model_arg.add_argument('--num_workers', type=int, default=8, help="number of dataloader workers")
model_arg.add_argument('--data_preprocessed', type=str2bool, default=True, 
                       help="check if data has already been preprocessed with data preprocess script")

# Training Parameters
train_arg = parser.add_argument_group('Training')
train_arg.add_argument('--train', type=str2bool, default=True, help="train session or not(test session)")
train_arg.add_argument('--N', type=int, default=4, help="number of speakers of batch")
train_arg.add_argument('--M', type=int, default=5, help="number of utterances per speaker")
train_arg.add_argument('--optim', type=str.lower, default='sgd', help="optimizer type")
train_arg.add_argument('--lr', type=float, default=1e-2, help="learning rate")
train_arg.add_argument('--epochs', type=int, default=950, help="max epoch")
train_arg.add_argument('--test_epochs', type=int, default=10, help="number of dataset iterations for calculating EER")
train_arg.add_argument('--device', type=str, default='cuda',help='Compute device for training/testing')
train_arg.add_argument('--log_interval', type=int, default=30,help='Print progress after x iterations')
train_arg.add_argument('--checkpoint_interval', type=int, default=120,help='Save model after x iterations')
train_arg.add_argument('--log_file', type=str, default='./speech_id_checkpoint/Stats',help='File to write logs to')
train_arg.add_argument('--checkpoint_dir', type=str, default='./speech_id_checkpoint',help='Directory to save model weights')
train_arg.add_argument('--comment', type=str, default='', help="any comment")
