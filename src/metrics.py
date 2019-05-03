#!/usr/local/bin/python
import tensorflow as tf
import numpy as np
import math


def n_accurate(y, y_):
    """
        y, y_: Tensor, shape: [batch_size, y_size];
    """
    correct_y_batch = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    n_accurate = tf.reduce_sum(tf.cast(correct_y_batch, tf.float32))  # similar to numpy.count_nonzero()
    return n_accurate


def eval_acc(n_acc, total):
    return float(n_acc) / total


def create_confusion_matrix(y, y_, is_distribution=True):
    """
        By batch. shape: [n_batch, batch_size, y_size]
    """
    n_samples = float(y_.shape[0])   # get dimension list
    if is_distribution:
        label_ref = np.argmax(y_, 1)  # 1-d array of 0 and 1
        label_hyp = np.argmax(y, 1)
    else:
        label_ref, label_hyp = y, y_

    # p & n in prediction
    p_in_hyp = np.sum(label_hyp)
    n_in_hyp = n_samples - p_in_hyp

    # Positive class: up
    tp = np.sum(np.multiply(label_ref, label_hyp))  # element-wise, both 1 can remain
    fp = p_in_hyp - tp  # predicted positive, but false

    # Negative class: down
    tn = n_samples - np.count_nonzero(label_ref + label_hyp)  # both 0 can remain
    fn = n_in_hyp - tn  # predicted negative, but false

    return float(tp), float(fp), float(tn), float(fn)


def eval_mcc(tp, fp, tn, fn):
    core_de = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(core_de) if core_de else None

def recall_pos(output, target):
    pred = np.argmax(output, axis=1)
    target = np.argmax(target, axis=1)
    assert pred.shape[0] == len(target)
    
    correct = np.sum((pred == target) * (target == 1))
    base = np.sum(target == 1)
    
    if base == 0:
        return 0
    return 1. * correct / base

def recall_neg(output, target):
    pred = np.argmax(output, axis=1)
    target = np.argmax(target, axis=1)
    assert pred.shape[0] == len(target)
    
    correct = np.sum((pred == target) * (target == 2))
    base = np.sum(target == 2)
    
    if base == 0:
        return 0
    return 1. * correct / base

def recall_neu(output, target):
    pred = np.argmax(output, axis=1)
    target = np.argmax(target, axis=1)
    assert pred.shape[0] == len(target)
    
    correct = np.sum((pred == target) * (target == 0))
    base = np.sum(target == 0)
    
    if base == 0:
        return 0
    return 1. * correct / base

def precision_pos(output, target):
    pred = np.argmax(output, axis=1)
    target = np.argmax(target, axis=1)
    assert pred.shape[0] == len(target)
    
    correct =  np.sum((pred == target) * (pred == 1))
    base = np.sum(pred == 1)
    
    if base == 0:
        return 0
    return 1. * correct / base

def precision_neg(output, target):
    pred = np.argmax(output, axis=1)
    target = np.argmax(target, axis=1)
    assert pred.shape[0] == len(target)
    
    correct =  np.sum((pred == target) * (pred == 2))
    base = np.sum(pred == 2)
    
    if base == 0:
        return 0
    return 1. * correct / base

def precision_neu(output, target):
    pred = np.argmax(output, axis=1)
    target = np.argmax(target, axis=1)
    assert pred.shape[0] == len(target)
    
    correct =  np.sum((pred == target) * (pred == 0))
    base = np.sum(pred == 0)
    
    if base == 0:
        return 0
    return 1. * correct / base

def eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_, use_mcc=None):
    gen_acc = eval_acc(n_acc=gen_n_acc, total=gen_size)
    gen_loss = np.average(gen_loss_list)

    gen_y, gen_y_ = np.vstack(y_list), np.vstack(y_list_)

    results = {'loss': gen_loss,
               'acc': gen_acc,
               'precision_neu': precision_neu(gen_y, gen_y_),
               'precision_pos': precision_pos(gen_y, gen_y_),
               'precision_neg': precision_neg(gen_y, gen_y_),
               'recall_neu': recall_neu(gen_y, gen_y_),
               'recall_pos': recall_pos(gen_y, gen_y_),
               'recall_neg': precision_neg(gen_y, gen_y_),
               }

    # if use_mcc:
    #     gen_y, gen_y_ = np.vstack(y_list), np.vstack(y_list_)
    #     tp, fp, tn, fn = create_confusion_matrix(y=gen_y, y_=gen_y_)
    #     results['mcc'] = eval_mcc(tp, fp, tn, fn)

    return results


def basic_train_stat(train_batch_loss_list, train_epoch_n_acc, train_epoch_size):
    train_epoch_loss = np.average(train_batch_loss_list)
    train_epoch_acc = eval_acc(n_acc=train_epoch_n_acc, total=train_epoch_size)
    return train_epoch_loss, train_epoch_acc


