import torch
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from constants import TrainingConstants
from logger import Logger

class Trainer(object):
    def __init__(self,
                 net,
                 dataset,
                 num_epochs=TrainingConstants.NUM_EPOCHS,
                 total_size=TrainingConstants.TOTAL_SIZE,
                 val_size=TrainingConstants.VAL_SIZE,
                 bsz=TrainingConstants.BSZ,
                 base_lr=TrainingConstants.BASE_LR,
                 device=TrainingConstants.DEVICE,
                 log_interval=TrainingConstants.LOG_INTERVAL):
        self._net = net
        self._dataset = dataset
        self._num_epochs = num_epochs
        self._total_size = total_size
        self._val_size = val_size
        self._bsz = bsz
        self._base_lr = base_lr
        self._device = device
        self._log_interval = log_interval

        self._native_logger = logging.getLogger(self.__class__.__name__)

    def _setup(self):
        self._logger = Logger(
            os.path.join(self._output_dir, TrainingConstants.LOG_DIR)
        )
        
        ind = np.arange(len(self._dataset))
        np.random.shuffle(ind)
        ind = ind[:ceil(self._total_size * len(ind))]
        train_ind = ind[:ceil((1 - self._val_size) * len(ind))]
        val_ind = ind[ceil((1 - self._val_size) * len(ind)):]

        train_sampler = SubsetRandomSampler(train_ind)
        val_sampler = SubsetRandomSampler(val_ind)
        self._train_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=1,
                                    pin_memory=True,
                                    sampler=train_sampler
                                 )
        self._val_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=1,
                                    pin_memory=True,
                                    sampler=val_sampler
                               )

        self._device = torch.device(self._device)
        self._net = self._net.to(self._device)
        self._optimizer = optim.Adadelta(self._net.parameters(), lr=self._base_lr)

    def _log_metric(self, epoch, metric_name, data):
        self._native_logger.info('Logging {} ...'.format(metric_name))

        if not isinstance(data, (list, np.ndarray)):
            data = [data]
        data = np.asarray(data)

        logs = OrderedDict()
        logs['{}_average'.format(metric_name)] = np.mean(data)
        logs['{}_stddev'.format(metric_name)] = np.std(data)
        logs['{}_max'.format(metric_name)] = np.max(data)
        logs['{}_min'.format(metric_name)] = np.min(data)

        # Write TensorFlow summaries.
        for key, value in logs.items():
            self._native_logger.info('\t{} : {}'.format(key, value))
            self._logger.log_scalar(value, key, epoch)
        self._logger.flush()

    def _train(self, epoch):
        self._net.train()
        
        num_batches = len(self._train_data_loader)
        train_losses = []
        for batch_idx, (X, y) in enumerate(self._train_data_loader):
            X = X.to(self._device)
            y = y.to(self._device)
            
            self._optimizer.zero_grad()

            output = self._net(X)
            loss = #TODO
            loss.backward()

            self._optimizer.step()

            if batch_idx % self._log_interval == 0:
                self._native_logger.info(
                    'Train Epoch: {} [Batch {}/{} ({:.0f}%)]\tLoss: {:.6f}\t'
                    'LR: {:.6f}'.format(
                        epoch,
                        batch_idx + 1,
                        num_batches,
                        100 * (batch_idx + 1) / num_batches,
                        loss.item(),
                        self._optimizer.param_groups[0]['lr']
                    )
                )
                train_losses.append(loss.item())

        self._log_metric(epoch, 'train/epoch_ce_loss', train_losses)

    def _eval(self, epoch):
        self._net.eval()

        eval_loss = 0
        eval_losses = []

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self._val_data_loader):
                X = X.to(self._device)
                y = y.to(self._device)

                output = self._net(X)

                loss = #TODO
                eval_loss += loss.item()
                eval_losses.append(loss.item())

        num_batches = len(self._val_data_loader)
        eval_loss /= num_batches
        self._log_metric(epoch, 'eval/epoch_ce_loss', eval_losses)

    def train(self):
        self._setup()
        for epoch in range(1, self._num_epochs + 1):
            self._train(epoch)
            self._eval(epoch)
            self._native_logger.info('')