class DeviceConstants(object):
    CUDA = 'cuda'
    CPU = 'cpu'

class TrainingConstants(object):
    OBJECTIVE = 'erm'
    TASK = 'classification'
    NUM_EPOCHS = 100
    BASE_LR = 0.01
    DEVICE = DeviceConstants.CPU
    LOG_INTERVAL = 1  # Batches.
    LOG_DIR = 'logs'
    NET_SAVE_FNAME = 'net_'
    SAVE_FREQUENCY = 5 # Epochs.

class ObjectiveConstants(object):
    HUMAN_ALIGNED_RISK_ALPHA = 0.4
    HUMAN_ALIGNED_RISK_BETA = 0.3
    ENTROPIC_TILT = 10
    TRIMMED_ALPHA = 0.05
    CVAR_ALPHA = 0.05
    MEANVAR_C = 1