import argparse
import torch.nn as nn
from model import TwoLayerMLP
from data import Dataset
from training_utils import Trainer, DeviceConstants, TrainingConstants, ObjectiveConstants
from objectives import human_aligned_risk, entropic_risk, trimmed_risk, cva, mean_variance

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a risk sensitive algorithm')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the training data.'
    )
    parser.add_argument(
        '--objective',
        type=str,
        default=TrainingConstants.OBJECTIVE,
        help='[erm|human|entropic|trimmed|cvar|meanvar] \
        empricial risk minimization [erm], human-aligned risk [hrm], \
        trimmed risk [trimmed], conditional value at risk [cvar], \
        mean variance [meanvar]'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=TrainingConstants.TASK,
        help='[classification|regression] Classification or regression.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=TrainingConstants.NUM_EPOCHS,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--total_size',
        type=float,
        default=TrainingConstants.TOTAL_SIZE,
        help='The proportion of the data to use.'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=TrainingConstants.VAL_SIZE,
        help='The proportion of the data to use for validation.'
    )
    parser.add_argument(
        '--bsz',
        type=int,
        default=TrainingConstants.BSZ,
        help='Training batch size.'
    )
    parser.add_argument(
        '--base_lr',
        type=float,
        default=TrainingConstants.BASE_LR,
        help='Base learning rate.'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Enable CUDA support and utilize GPU devices.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=TrainingConstants.LOG_INTERVAL,
        help='Log interval in batches.'
    )

    parser.add_argument(
        '--human_alpha',
        type=float,
        default=ObjectiveConstants.HUMAN_ALIGNED_RISK_ALPHA,
        help='alpha parameter for human aligned risk'
    )
    parser.add_argument(
        '--human_beta',
        type=float,
        default=ObjectiveConstants.HUMAN_ALIGNED_RISK_BETA,
        help='beta parameter for human aligned risk'
    )
    parser.add_argument(
        '--entropic_tilt',
        type=float,
        default=ObjectiveConstants.ENTROPIC_TILT,
        help='tilt parameter for entropic risk'
    )
    parser.add_argument(
        '--trimmed_alpha',
        type=float,
        default=ObjectiveConstants.TRIMMED_ALPHA,
        help='alpha parameter for trimmed risk'
    )
    parser.add_argument(
        '--cvar_alpha',
        type=float,
        default=ObjectiveConstants.CVAR_ALPHA,
        help='alpha parameter for CVaR risk'
    )
    parser.add_argument(
        '--meanvar_c',
        type=float,
        default=ObjectiveConstants.MEANVAR_C,
        help='c parameter for mean variance risk'
    )

    # Training Constants
    args = parser.parse_args()
    data_dir = args.data_dir
    objective = args.objective
    task = args.task
    num_epochs = args.num_epochs
    total_size = args.total_size
    val_size = args.val_size
    bsz = args.bsz
    base_lr = args.base_lr
    log_interval = args.log_interval

    assert objective in ['erm', 'human', 'entropic', 'trimmed', 'cvar', 'meanvar']
    assert task in ['classification', 'regression']

    if task == 'classification':
        criteiron = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = nn.MSELoss(reduction='mean')

    if objective == 'human':
        criterion = human_aligned_risk(a=args.human_alpha, b=args.human_beta, criterion=criterion)
    elif objective == 'entropic':
        criterion = entropic_risk(t=args.entropic_tilt, criterion=criteiron)
    elif objective == 'trimmed':
        criteiron = trimmed_risk(a=args.trimmed_alpha, criteiron=criteiron)
    elif objective == 'cvar':
        criteiron = cvar(a=args.cvar_alpha, criteiron=criteiron)
    elif objective == 'meanvar':
        criteiron = mean_variance(c=args.meanvar_c, criteiron=criteiron)

    cuda = args.cuda
    if cuda:
        device = DeviceConstants.CUDA
    else:
        device = DeviceConstants.CPU

    dataset = Dataset(data_dir)
    # Specifiy network manually
    net = TwoLayerMLP(dataset.input_size, dataset.hidden_size, dataset.output_size)
    
    trainer = Trainer(net,
                      dataset,
                      criterion
                      num_epochs,
                      total_size,
                      val_size,
                      bsz,
                      device,
                      log_interval)

    trainer.train()
