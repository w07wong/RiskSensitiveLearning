class DeviceConstants(object):
    CUDA = 'cuda'
    CPU = 'cpu'

class TrainingConstants(object):
    OBJECTIVE = 'erm'
    NUM_EPOCHS = 100
    BASE_LR = 0.01
    DEVICE = DeviceConstants.CPU
    LOG_INTERVAL = 1  # Batches.
    LOG_DIR = 'logs'