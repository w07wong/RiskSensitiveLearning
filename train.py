import argparse
from TwoLayerMLP import TwoLayerMLP
from trainer import Trainer
from constants import DeviceConstants

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a risk sensitive algorithm')
    # parser.add_argument(
    #     'data_dir',
    #     type=str,
    #     help='Path to the training data.'
    # )
    parser.add_argument(
        '--objective',
        type=str,
        default=TrainingConstants.OBJECTIVE,
        help='Risk sensitive objective to evaluate.'
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

    args = parser.parse_args()
    # data_dir = args.data_dir
    objective = args.objective
    num_epochs = args.num_epochs
    total_size = args.total_size
    val_size = args.val_size
    bsz = args.bsz
    base_lr = args.base_lr
    log_interval = args.log_interval

    cuda = args.cuda
    if cuda:
        device = DeviceConstants.CUDA
    else:
        device = DeviceConstants.CPU

    dataset = Dataset(X, y)
    net = TwoLayerMLP(dataset.input_size, dataset.hidden_size, dataset.output_size)
    trainer = Trainer(net,
                      dataset,
                      num_epochs,
                      total_size,
                      val_size,
                      bsz,
                      device,
                      log_interval)

    trainer.train()
