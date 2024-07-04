import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

import sys

sys.path.append('../')
import config


def make_path(path):
    # 检查path指定的路径是否存在
    if not os.path.exists(path):
        # 如果路径不存在，则使用os.makedirs创建该路径（包括所有必要的父目录）
        os.makedirs(path)


# 定义一个评估函数eval，接收参数：优化器选项（opt），模型（model）和数据集（dataset）
def eval(opt, model, dataset):
    # 将模型设置为评估模式
    model.eval()

    # 初始化空列表以存储结果
    total_name = []
    total_pred = []
    total_label = []
    total_recon = []
    total_hidden = []

    # 遍历数据集中的每个样本
    for ii, data in enumerate(dataset):
        # 设置模型输入并进行预处理
        model.set_input(data)

        # 运行模型测试
        model.test()

        # 获取预测值、重构损失、隐藏层状态
        pred = model.pred.detach().cpu().numpy()
        loss_recon = model.loss_recon.detach().cpu().numpy()
        hiddens = model.hiddens.detach().cpu().numpy()
        name = data['int2name']
        label = data['label']

        # 将结果添加到对应的列表中
        total_name.append(name)
        total_pred.append(pred)
        total_label.append(label)
        total_recon.append(loss_recon)
        total_hidden.append(hiddens)

    # 将列表转换为numpy数组
    total_name = np.concatenate(total_name)  # [sample_num, ]
    total_pred = np.concatenate(total_pred)  # [sample_num, num_classes]
    total_label = np.concatenate(total_label)  # [sample_num, ]
    total_hidden = np.concatenate(total_hidden)  # [sample_num, featdim]
    # 计算重构损失的平均值
    total_recon = np.sum(total_recon) / len(total_pred)  # [1]

    # 将模型恢复为训练模式
    model.train()

    # 根据数据集类型计算准确率和F1分数
    dataset = opt.dataset_mode.split('_')[0]
    if dataset in ['cmumosi', 'cmumosei']:
        # 对非零标签进行操作，移除0和mask
        non_zeros = np.array([i for i, e in enumerate(total_label) if e != 0])
        acc = accuracy_score((total_label[non_zeros] > 0), (total_pred[non_zeros] > 0))
        f1 = f1_score((total_label[non_zeros] > 0), (total_pred[non_zeros] > 0), average='weighted')
    elif dataset in ['iemocapfour', 'iemocapsix']:
        acc = accuracy_score(total_label, np.argmax(total_pred, 1))
        f1 = f1_score(total_label, np.argmax(total_pred, 1), average='weighted')

    # 返回F1分数、准确率、重构损失及所有标签、预测、隐藏层状态和名称
    return f1, acc, total_recon, [total_label, total_pred, total_hidden, total_name]


# 定义一个清理检查点文件的函数，根据给定的实验名称和存储周期
def clean_chekpoints(expr_name, store_epoch):
    # 设置根目录为'checkpoints'文件夹下以expr_name命名的子目录
    root = os.path.join('checkpoints', expr_name)

    # 遍历该目录下的所有文件
    for checkpoint in os.listdir(root):
        # 检查文件名是否不符合指定的存储周期格式（不以store_epoch开头且以'.pth'结尾）
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            # 如果不符合，删除该文件
            os.remove(os.path.join(root, checkpoint))


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, 'logger')  # get logger path '/logs/mmin/1'
    if not os.path.exists(logger_path): os.mkdir(logger_path)

    result_dir = os.path.join(opt.log_dir, opt.name, 'results')  # '/logs/mmin/results'
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    suffix = '_'.join([opt.model, opt.dataset_mode])  # get logger suffix 'mmin_multimodal_miss'
    logger = get_logger(logger_path, suffix)  # get logger

    dataset = opt.dataset_mode.split('_')[0]
    if dataset in ['cmumosi', 'cmumosei']:
        opt.output_dim = 1
        num_folder = 1
    elif dataset == 'iemocapfour':
        opt.output_dim = 4
        num_folder = 5
    elif dataset == 'iemocapsix':
        opt.output_dim = 6
        num_folder = 5

    folder_acc = []  # save best epoch
    folder_f1 = []  # save best epoch
    folder_recon = []  # save best epoch
    folder_save = []  # save best epoch
    for index in range(num_folder):
        print(f'>>>>> Cross-validation: training on the {index + 1} folder >>>>>')
        opt.cvNo = index + 1

        ## create dataset
        trn_dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])

        ## create model
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        best_eval_f1 = 0
        best_eval_epoch = -1  # record the best eval epoch

        for epoch in range(opt.epoch_count,
                           opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            for i, data in enumerate(trn_dataset):  # inner loop within one epoch
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights
            model.save_networks(epoch)
            model.update_learning_rate(logger)  # update learning rates at the end of every epoch.

            # validation
            f1, acc, recon_loss, [val_label, val_pred, val_hidden, val_name] = eval(opt, model, val_dataset)
            print(
                f'Epoch {epoch}/{opt.niter + opt.niter_decay}  Val result: f1:{f1:2.2%} acc:{acc:2.2%} recon_loss:{recon_loss:.4f}')

            # record epoch with best result
            if f1 > best_eval_f1:
                best_eval_f1 = f1
                best_eval_epoch = epoch

        # test on best eval
        model.load_networks(best_eval_epoch)
        f1, acc, recon_loss, [tst_label, tst_pred, tst_hidden, tst_name] = eval(opt, model, tst_dataset)
        clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)

        folder_f1.append(f1)
        folder_acc.append(acc)
        folder_recon.append(recon_loss)
        folder_save.append(
            {'test_labels': tst_label, 'test_preds': tst_pred, 'test_hiddens': tst_hidden, 'test_names': tst_name})
        print(f'>>>>> Finish: training on the {index + 1} folder >>>>>')

    ## save results
    # gain suffix_name
    model_name = opt.model.split('_')[-1]
    suffix_name = f'{dataset}_{model_name}_mask:{opt.mask_rate:.1f}'

    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    mean_recon = np.mean(np.array(folder_recon))
    res_name = f'f1:{mean_f1:2.2%}_acc:{mean_acc:2.2%}_reconloss:{mean_recon:.4f}'

    save_root = config.MODEL_DIR
    if not os.path.exists(save_root): os.makedirs(save_root)
    save_path = f'{save_root}/{suffix_name}_{res_name}_{time.time()}.npz'
    print(f'save results in {save_path}')
    np.savez_compressed(save_path,
                        folder_save=np.array(folder_save, dtype=object)  # save non-structure type
                        )
    print(f' =========== finish =========== ')

