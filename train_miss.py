import os
import time
import os
import time
import numpy as np
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


# 如果路径不存在，则创建该路径
def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 评估模型，val_iter是验证集迭代器，is_save决定是否保存结果，phase表示是测试阶段还是其他
def eval(model, val_iter, is_save=False, phase='test'):
    model.eval()  # 设置模型为评估模式
    total_pred = []  # 存储预测结果
    total_label = []  # 存储真实标签
    total_miss_type = []  # 存储缺失类型
    # 遍历一个epoch内的数据
    for _, data in enumerate(val_iter):
        model.set_input(data)  # 解包数据集并应用预处理
        model.test()  # 进行测试
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()  # 获取预测结果
        label = data['label']  # 获取真实标签
        miss_type = np.array(data['miss_type'])  # 获取缺失类型
        total_pred.append(pred)  # 累加预测结果
        total_label.append(label)  # 累加真实标签
        total_miss_type.append(miss_type)  # 累加缺失类型

    # 计算指标
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)
    acc = accuracy_score(total_label, total_pred)  # 准确率
    uar = recall_score(total_label, total_pred, average='macro')  # 宏平均召回率
    f1 = f1_score(total_label, total_pred, average='macro')  # 宏平均F1分数
    cm = confusion_matrix(total_label, total_pred)  # 混淆矩阵

    if is_save:
        # 保存测试的全部结果
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

        # 保存部分结果
        for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
            part_index = np.where(total_miss_type == part_name)
            part_pred = total_pred[part_index]
            part_label = total_label[part_index]
            acc_part = accuracy_score(part_label, part_pred)
            uar_part = recall_score(part_label, part_pred, average='macro')
            f1_part = f1_score(part_label, part_pred, average='macro')
            np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
            np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
            if phase == 'test':
                recorder_lookup[part_name].write_result_to_tsv({
                    'acc': acc_part,
                    'uar': uar_part,
                    'f1': f1_part
                }, cvNo=opt.cvNo)

    model.train()  # 将模型设置回训练模式
    return acc, uar, f1, cm  # 返回计算的指标和混淆矩阵


# 清理检查点，保留指定epoch的模型，删除其他epoch的模型
def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

if __name__ == '__main__':
    opt = Options().parse()                             # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo)) # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.mkdir(logger_path)

    result_dir = os.path.join(opt.log_dir, opt.name, 'results')
    if not os.path.exists(result_dir):                  # make sure result path exists
        os.mkdir(result_dir)

    total_cv = 10 if opt.corpus_name == 'IEMOCAP' else 12
    recorder_lookup = {                                 # init result recoreder
        "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv),
        "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'), total_cv=total_cv),
        "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'), total_cv=total_cv),
        "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'), total_cv=total_cv),
        "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'), total_cv=total_cv),
        "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'), total_cv=total_cv),
        "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv),
    }

    suffix = '_'.join([opt.model, opt.dataset_mode])    # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger

    if opt.has_test:                                    # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    best_eval_epoch = -1           # record the best eval epoch
    best_eval_acc, best_eval_uar, best_eval_f1 = 0, 0, 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)           # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.

        # eval
        acc, uar, f1, cm = eval(model, val_dataset)
        logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (epoch, opt.niter + opt.niter_decay, acc, uar, f1))
        logger.info('\n{}'.format(cm))

        # show test result for debugging
        if opt.has_test and opt.verbose:
            acc, uar, f1, cm = eval(model, tst_dataset)
            logger.info('Tst result of epoch %d acc %.4f uar %.4f f1 %.4f' % (epoch, acc, uar, f1))
            logger.info('\n{}'.format(cm))

        # record epoch with best result
        if opt.corpus_name == 'IEMOCAP':
            if uar > best_eval_uar:
                best_eval_epoch = epoch
                best_eval_uar = uar
                best_eval_acc = acc
                best_eval_f1 = f1
            select_metric = 'uar'
            best_metric = best_eval_uar
        elif opt.corpus_name == 'MSP':
            if f1 > best_eval_f1:
                best_eval_epoch = epoch
                best_eval_uar = uar
                best_eval_acc = acc
                best_eval_f1 = f1
            select_metric = 'f1'
            best_metric = best_eval_f1
        else:
            raise ValueError(f'corpus name must be IEMOCAP or MSP, but got {opt.corpus_name}')

    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))
    # test
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        _ = eval(model, val_dataset, is_save=True, phase='val')
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
        logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        logger.info('\n{}'.format(cm))
        recorder_lookup['total'].write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1
        }, cvNo=opt.cvNo)
    else:
        recorder_lookup['total'].write_result_to_tsv({
            'acc': best_eval_acc,
            'uar': best_eval_uar,
            'f1': best_eval_f1
        }, cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
