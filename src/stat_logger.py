#!/usr/local/bin/python
import metrics as metrics
from ConfigLoader import logger


def print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc, train_batch_size):
    iter_str = '\titer: {0}'.format(n_iter)
    loss_str = 'batch loss: {:.6f}'.format(train_batch_loss) if type(train_batch_loss) is float else 'batch loss: {}'.format(train_batch_loss)
    train_batch_acc = metrics.eval_acc(n_acc=train_batch_n_acc, total=train_batch_size)
    acc_str = 'batch acc: {:.6f}'.format(train_batch_acc)
    logger.info(', '.join((iter_str, loss_str, acc_str)))


def print_epoch_stat(epoch_loss, epoch_acc):
    epoch_stat_pattern = 'Epoch: loss: {0:.6f}, acc: {1:.6f}'
    logger.info(epoch_stat_pattern.format(epoch_loss, epoch_acc))


def print_eval_res(result_dict, use_mcc=None):
    iter_str = '\tEval'
    info_list = [iter_str] + ['{}: {:.6f}'.format(k, float(v)) for k, v in result_dict.iteritems()]
    logger.info(', '.join(info_list))
