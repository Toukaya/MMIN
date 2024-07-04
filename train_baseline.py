import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


def make_path(path):
    """
    参数:
    path (str): 需要创建的路径字符串

    返回:
    None
    """
    # 检查路径是否存在
    # 检查路径是否存在
    if not os.path.exists(path):
        # 如果路径不存在，创建该路径
        os.makedirs(path)


"""
此函数用于创建指定的路径。
如果路径不存在，它将使用os.makedirs()方法来创建路径及其所有必要的父目录。
"""


def make_path(path):
    """
    参数:
    path (str): 需要创建的路径字符串

    返回:
    None
    """
    # 检查路径是否存在
    if not os.path.exists(path):
        # 如果路径不存在，创建该路径
        os.makedirs(path)


# 定义一个评估模型的函数，传入模型、验证数据迭代器、是否保存结果和当前阶段（'test'或其它）
def eval(model, val_iter, is_save=False, phase='test'):
    # 将模型设置为评估模式
    model.eval()

    # 初始化预测值和标签的列表
    total_pred = []
    total_label = []

    # 遍历验证数据集
    for i, data in enumerate(val_iter):
        # 设置模型输入并进行预处理
        model.set_input(data)
        # 前向传播并获取结果
        pred = model.pred.detach().cpu().numpy()
        label = data['label']
        # 将预测值和标签添加到列表中
        total_pred.append(pred)
        total_label.append(label)
    # 合并所有样本的预测值和标签
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)

    # 根据数据集类型计算准确率和F1分数
    dataset = opt.dataset_mode.split('_')[0]
    if dataset in ['cmumosi', 'cmumosei']:
        # 对于非零标签，计算准确率和F1分数
        non_zeros = np.array([i for i, e in enumerate(total_label) if e != 0])
        acc = accuracy_score((total_label[non_zeros] > 0), (total_pred[non_zeros] > 0))
        f1 = f1_score((total_label[non_zeros] > 0), (total_pred[non_zeros] > 0), average='weighted')
    elif dataset in ['iemocapfour', 'iemocapsix']:
        # 对于IEMOCAP数据集，取预测值的最大概率作为类别，然后计算准确率和F1分数
        total_pred = np.argmax(total_pred, 1)
        acc = round(accuracy_score(total_label, total_pred), 2)
        f1 = round(f1_score(total_label, total_pred, average='weighted'), 2)

    # 如果需要保存结果，将预测值和标签保存到指定目录
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    # 将模型恢复为训练模式
    model.train()

    # 返回准确率和F1分数
    return acc, f1


# 定义一个清理检查点的函数，根据给定的实验名称和存储的epoch数
def clean_chekpoints(expr_name, store_epoch):
    # 将'checkpoints'目录与实验名称拼接成完整路径
    root = os.path.join('checkpoints', expr_name)

    # 遍历该路径下的所有文件
    for checkpoint in os.listdir(root):
        # 检查文件名是否以指定的epoch号开头，并以'.pth'结尾
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            # 如果不匹配，删除该文件
            os.remove(os.path.join(root, checkpoint))


if __name__ == '__main__':
    opt = TrainOptions().parse()  # 获取训练选项
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))  # 获取日志路径
    if not os.path.exists(logger_path):  # 确保日志路径存在
        os.mkdir(logger_path)  # 创建日志路径

    result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'),
                                     total_cv=12)  # 初始化结果记录器
    suffix = '_'.join([opt.model, opt.dataset_mode])  # 获取日志后缀：utt_fusion_multimodal
    logger = get_logger(logger_path, suffix)  # 获取日志记录器

    dataset = opt.dataset_mode.split('_')[0]  # 分割数据集模式，获取数据集名称
    if dataset in ['cmumosi', 'cmumosei']:  # 如果是CMU MOSI或MOSEI数据集
        assert opt.output_dim == 1  # 确保输出维度为1
        num_folder = 1  # 设置文件夹数量为1
    elif dataset == 'iemocapfour':  # 如果是IEMOCAP四分类
        assert opt.output_dim == 4  # 确保输出维度为4
        num_folder = 5  # 设置文件夹数量为5
    elif dataset == 'iemocapsix':  # 如果是IEMOCAP六分类
        assert opt.output_dim == 6  # 确保输出维度为6
        num_folder = 5  # 设置文件夹数量为5

    folder_acc = []  # 初始化存储每个文件夹准确率的列表
    folder_f1 = []  # 初始化存储每个文件夹F1分数的列表
    folder_save = []  # 初始化存储每个文件夹保存信息的列表

    for index in range(num_folder):  # 遍历每个文件夹
        print(f'>>>>> Cross-validation: training on the {index + 1} folder >>>>>')  # 打印当前正在训练的文件夹信息
        opt.cvNo = index + 1  # 设置当前交叉验证的文件夹编号

        ## 读取数据
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])  # 创建数据集

        dataset_size = len(dataset)  # 获取数据集大小
        logger.info('The number of training samples = %d' % dataset_size)  # 记录训练样本数量
        model = create_model(opt)  # 创建模型
        model.setup(opt)  # 设置模型，包括加载网络和创建调度器
        total_iters = 0  # 初始化总迭代次数
        best_eval_uar = 0  # 记录最佳评估UAR
        best_epoch_acc, best_epoch_f1 = 0, 0  # 记录最佳epoch的准确率和F1分数
        best_eval_epoch = -1  # 记录最佳评估epoch

        ## epoch循环开始
        for epoch in range(opt.epoch_count,
                           opt.niter + opt.niter_decay + 1):  # 对不同的epoch进行循环；我们通过<epoch_count>和<epoch_count>+<save_latest_freq>保存模型
            epoch_start_time = time.time()  # 记录整个epoch的开始时间

            for i, data in enumerate(dataset):  # 在一个epoch内部循环
                iter_start_time = time.time()  # 记录每次迭代的开始时间
                total_iters += 1  # 增加迭代计数
                model.set_input(data)  # 设置模型输入并进行预处理
                model.optimize_parameters(epoch)  # 计算损失函数，获取梯度，更新网络权重

                if total_iters % opt.print_freq == 0:  # 每隔一定迭代次数，打印训练损失并保存日志信息到磁盘
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size

                if total_iters % opt.save_latest_freq == 0:  # 每隔一定迭代次数，缓存我们的最新模型
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

            if epoch % opt.save_epoch_freq == 0:  # 每隔一定的epoch数，缓存我们的模型
                model.save_networks('latest')
                model.save_networks(epoch)

            logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
                epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))  # 记录每个epoch结束时的信息
            model.update_learning_rate(logger)  # 在每个epoch结束时更新学习率

            # 评估验证集
            acc, f1 = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f f1 %.4f' % (
            epoch, opt.niter + opt.niter_decay, acc, f1))  # 记录验证集的准确率和F1分数

            if f1 > best_epoch_f1:  # 如果当前epoch的F1分数超过之前的最佳分数
                best_eval_epoch = epoch
                best_epoch_acc = acc
                best_epoch_f1 = f1

        # 测试最佳epoch
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)  # 加载在验证集上找到的最佳模型
        model.load_networks(best_eval_epoch)
        acc, f1 = eval(model, tst_dataset, is_save=True, phase='test')  # 在测试集上评估模型
        folder_acc.append(acc)
        folder_f1.append(f1)
        clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)  # 清理检查点

        print(f'>>>>> Finish: training on the {index + 1} folder >>>>>')

