''' Configuration File.
'''

SEED = 0
CUDA_VISIBLE_DEVICES = 4
NUM_TRAIN = 2000 # N \Fashion MNIST 60000, cifar 10/100 50000
# NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 28 # B
SUBSET    = NUM_TRAIN # M
ADDENDUM  = int(NUM_TRAIN/5) # K
ADD_CONST = 1 # 1 ~ 9

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda

TRIALS = 10
CYCLES = 5

EPOCH = 200
EPOCH_GCN = 200
LR = 1e-1
LR_GCN = 1e-3
MILESTONES = [160, 240]
EPOCHL = 120#20 #120 # After 120 epochs, stop 
EPOCHV = 100 # VAAL number of epochs

MOMENTUM = 0.9
WDECAY = 5e-4#2e-3# 5e-4

INIT_PARA = True
INIT_TRAINSET = False