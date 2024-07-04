import os
import json
import numpy as np
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from models.utils.config import OptConfig


def eval_miss(model, val_iter):
    # 将模型设置为评估模式
    model.eval()
    # 初始化预测结果、真实标签和缺失类型列表
    total_pred = []
    total_label = []
    total_miss_type = []

    # 遍历验证集数据
    for _, data in enumerate(val_iter):  # 内部循环，遍历一个epoch的数据
        # 设置模型输入（解包数据并进行预处理）
        model.set_input(data)
        # 运行模型测试
        model.test()
        # 获取模型预测的类别（取最大概率的维度）
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        # 获取真实标签
        label = data['label']
        # 获取缺失类型
        miss_type = np.array(data['miss_type'])
        # 将预测结果、真实标签和缺失类型添加到列表中
        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)

    # 合并所有数据
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)
    # 计算整体准确率、宏平均召回率和宏平均F1分数
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    # 计算混淆矩阵
    cm = confusion_matrix(total_label, total_pred)

    # 打印整体性能指标
    print(f'Total acc:{acc:.4f} uar:{uar:.4f} f1:{f1:.4f}')
    # 遍历每种缺失类型并计算其性能指标
    for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
        # 获取对应缺失类型的索引
        part_index = np.where(total_miss_type == part_name)
        # 提取该类型的数据
        part_pred = total_pred[part_index]
        part_label = total_label[part_index]
        # 计算该类型缺失的准确率、宏平均召回率和宏平均F1分数
        acc_part = accuracy_score(part_label, part_pred)
        uar_part = recall_score(part_label, part_pred, average='macro')
        f1_part = f1_score(part_label, part_pred, average='macro')
        # 打印该类型缺失的性能指标
        print(f'{part_name}, acc:{acc_part:.4f}, {uar_part:.4f}, {f1_part:.4f}')

    # 返回整体性能指标
    return acc, uar, f1, cm

# 定义一个评估模型在验证集上的函数
def eval_all(model, val_iter):
    # 将模型设置为评估模式
    model.eval()
    # 初始化预测和真实标签列表
    total_pred = []
    total_label = []

    # 遍历验证集数据
    for _, data in enumerate(val_iter):  # 内部循环，遍历一个epoch内的所有样本
        # 设置模型输入（解包数据并进行预处理）
        model.set_input(data)
        # 执行模型测试
        model.test()
        # 获取模型预测结果，取最大概率对应的类别，并转换为numpy数组
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        # 获取真实标签
        label = data['label']
        # 将预测和真实标签添加到列表中
        total_pred.append(pred)
        total_label.append(label)

    # 合并所有预测和真实标签
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    # 计算准确率
    acc = accuracy_score(total_label, total_pred)
    # 计算宏平均召回率
    uar = recall_score(total_label, total_pred, average='macro')
    # 计算宏平均F1分数
    f1 = f1_score(total_label, total_pred, average='macro')
    # 计算混淆矩阵
    cm = confusion_matrix(total_label, total_pred)

    # 打印评估指标
    print(f'Total acc:{acc:.4f} uar:{uar:.4f} f1:{f1:.4f}')
    # 返回评估指标
    return acc, uar, f1, cm


if __name__ == '__main__':
    test_miss = True
    test_base = False
    in_men = True
    total_cv = 10
    gpu_ids = [0]
    ckpt_path = "checkpoints/CAP_utt_fusion_AVL_run1"
    config = json.load(open(os.path.join(ckpt_path, 'train_opt.conf')))
    opt = OptConfig()
    opt.load(config)
    if test_base:
        opt.dataset_mode = 'multimodal'
    if test_miss:
        opt.dataset_mode = 'multimodal_miss'
        
    opt.gpu_ids = gpu_ids
    setattr(opt, 'in_mem', in_men)
    model = create_model(opt)
    model.setup(opt)
    results = []
    for cv in range(1, 1+total_cv):
        opt.cvNo = cv
        tst_dataloader = create_dataset_with_args(opt, set_name='tst')
        model.load_networks_cv(os.path.join(ckpt_path, str(cv)))
        model.eval()
        if test_base:
            acc, uar, f1, cm = eval_all(model, tst_dataloader)
        if test_miss:
            acc, uar, f1, cm = eval_miss(model, tst_dataloader)
        results.append([acc, uar, f1])
    
    mean_acc = sum([x[0] for x in results])/total_cv
    mean_uar = sum([x[1] for x in results])/total_cv
    mean_f1 = sum([x[2] for x in results])/total_cv
    print(f'Avg acc:{mean_acc:.4f} uar:{mean_uar:.4f} f1:{mean_f1:.4f}')
    