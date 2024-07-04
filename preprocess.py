import re
import os
import copy
import tqdm
import glob
import json
import math
import shutil
import random
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import config

# t: ms
def convert_time(t):
    # 将输入的时间（毫秒）转换为整数
    t = int(t)

    # 计算毫秒部分
    ms = t % 1000

    # 计算并舍去毫秒后的时间（秒）
    t = math.floor(t / 1000)

    # 计算小时数
    h = math.floor(t / 3600)

    # 计算剩余分钟数
    m = math.floor((t - h * 3600) / 60)

    # 计算剩余秒数
    s = t - 3600 * h - 60 * m

    # 格式化时间字符串，返回小时:分钟:秒.毫秒
    return '%02d:%02d:%02d.%03d' % (h, m, s, ms)


# 定义一个函数，用于选择CMUMOSEI数据集中的视频
def select_videos_for_cmumosei():
    # 设置数据根目录
    data_root = '../emotion-data/CMUMOSEI'

    # 转录文件路径
    trans_file = os.path.join(data_root, 'transcription.csv')

    # 原始视频存储路径
    video_root = os.path.join(data_root, 'whole_video')

    # 存储子视频的目标路径
    save_root = os.path.join(data_root, 'subvideo')
    if not os.path.exists(save_root): os.makedirs(save_root)

    # 如果目标路径不存在，则创建
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 初始化错误行列表
    error_lines = []

    # 读取转录文件为DataFrame
    df = pd.read_csv(trans_file)

    # 遍历DataFrame的每一行
    for idx, row in tqdm.tqdm(df.iterrows()):
        # 获取当前行的视频名称
        name = row['name']

        # 获取当前行的句子文本
        sentence = row['sentence']

        # 构建原始视频的完整路径
        video_path = os.path.join(video_root, name + '.mp4')

        # 检查视频是否存在，若不存在则添加到错误行列表
        if not os.path.exists(video_path):
            error_lines.append(name)
        else:
            # 构建子视频的保存路径
            save_path = os.path.join(save_root, name + '.mp4')

            # 复制视频到目标路径
            cmd = f'cp {video_path} {save_path}'
            os.system(cmd)

    # 输出错误样本数量
    print(f'error samples: {len(error_lines)}')


def feature_compressed(feature_root, save_root):
    # 获取特征根目录下的所有文件名
    names = os.listdir(feature_root)

    # 遍历每个文件名，处理特征并存储 (names, speakers) => features
    features = []
    feature_dim = -1  # 初始化特征维度为-1

    # 对每个文件名进行处理
    for ii, name in enumerate(names):
        # 打印处理进度
        print(f'Process {name}: {ii + 1}/{len(names)}')

        # 获取当前文件夹路径和其中的面部特征文件名
        feature = []
        feature_dir = os.path.join(feature_root, name)
        facenames = os.listdir(feature_dir)

        # 按名称排序并处理每个面部特征文件
        for facename in sorted(facenames):
            # 加载面部特征数据
            facefeat = np.load(os.path.join(feature_dir, facename))

            # 更新最大特征维度
            feature_dim = max(feature_dim, facefeat.shape[-1])

            # 将面部特征添加到列表中
            feature.append(facefeat)

        # 将列表转换为numpy数组并压缩
        single_feature = np.array(feature).squeeze()

        # 如果数组为空，用全零数组填充
        if len(single_feature) == 0:
            single_feature = np.zeros((feature_dim,))
        # 如果特征是二维（序列长度和特征维度），则按列平均
        elif len(single_feature.shape) == 2:
            single_feature = np.mean(single_feature, axis=0)

        # 添加处理后的单个特征到列表
        features.append(single_feature)

    # 创建保存根目录（如果不存在）
    if not os.path.exists(save_root): os.makedirs(save_root)

    # 保存每个处理后的特征文件
    for ii in range(len(names)):
        # 构建保存路径并保存numpy数组
        save_path = os.path.join(save_root, names[ii] + '.npy')
        np.save(save_path, features[ii])


# 定义一个名为 feature_compressed_iemocap 的函数，接收两个参数：feature_root 和 save_root
def feature_compressed_iemocap(feature_root, save_root):
    # 获取 feature_root 目录下的文件名列表
    names = os.listdir(feature_root)

    # 初始化空列表，用于存储处理后的特征 (names, speakers) => features
    features = []
    # 初始化特征维度为 -1
    feature_dim = -1

    # 遍历每个文件名
    for ii, name in enumerate(names):
        # 打印处理进度
        print(f'Process {name}: {ii + 1}/{len(names)}')

        # 初始化字典，存储不同性别的特征
        feature = {'F': [], 'M': []}
        # 获取当前文件名对应目录
        feature_dir = os.path.join(feature_root, name)
        # 获取该目录下所有文件名
        facenames = os.listdir(feature_dir)

        # 对每个文件进行排序并处理
        for facename in sorted(facenames):
            # 检查文件名是否包含 'F' 或 'M'
            assert facename.find('F') >= 0 or facename.find('M') >= 0
            # 加载特征数据
            facefeat = np.load(os.path.join(feature_dir, facename))
            # 更新最大特征维度
            feature_dim = max(feature_dim, facefeat.shape[-1])
            # 根据文件名中的 'F' 或 'M' 将特征添加到相应性别列表
            if facename.find('F') >= 0:
                feature['F'].append(facefeat)
            else:
                feature['M'].append(facefeat)

        # 对每个性别的特征进行处理
        for speaker in feature:
            # 将列表转换为数组并挤压维度
            single_feature = np.array(feature[speaker]).squeeze()
            # 如果数组为空，用全零数组填充
            if len(single_feature) == 0:
                single_feature = np.zeros((feature_dim,))
            # 如果数组是二维，取平均值
            elif len(single_feature.shape) == 2:
                single_feature = np.mean(single_feature, axis=0)
            # 更新字典中的特征
            feature[speaker] = single_feature
        # 将处理后的特征添加到 features 列表中
        features.append(feature)

    # 创建保存目录，如果不存在
    if not os.path.exists(save_root): os.makedirs(save_root)
    # 遍历每个文件名
    for ii in range(len(names)):
        # 构建子目录路径
        save_subroot = os.path.join(save_root, names[ii])
        # 创建子目录，如果不存在
        if not os.path.exists(save_subroot): os.makedirs(save_subroot)
        # 获取当前文件名对应的特征
        feature = features[ii]
        # 遍历每个性别
        for speaker in feature:
            # 构建保存路径
            save_path = os.path.join(save_subroot, f'compress_{speaker}.npy')
            # 保存特征数据
            np.save(save_path, feature[speaker])


# 定义一个函数，用于生成IEMOCAP数据集的转录文件
def generate_transcription_files_IEMOCAP():
    # 获取所有转录音频的名称和句子
    names = []
    sentences = []
    # IEMOCAP数据集的根目录
    data_root = '../emotion-data/IEMOCAP'

    # 遍历每个会话
    for session_name in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
        # 转录文件的路径
        transcription_root = os.path.join(data_root, session_name, 'dialog/transcriptions')

        # 查找并处理所有转录文件
        for trans_path in glob.glob(transcription_root + '/S*.txt'):
            with open(trans_path, encoding='utf8') as f:
                # 读取文件中的每一行并移除空白字符
                lines = [line.strip() for line in f]

            # 过滤掉空行
            lines = [line for line in lines if len(line) != 0]

            # 处理每一行转录信息
            for line in lines:  # line: Ses05F_script03_1_F033 [241.6700-243.4048]: You knew there was nothing.
                try:
                    # 提取子名称、开始时间、结束时间和句子
                    subname = line.split(' [')[0]
                    start = line.split('[')[1].split('-')[0]
                    end = line.split('-')[1].split(']')[0]
                    # 将时间转换为毫秒
                    start = convert_time(float(start) * 1000)
                    end = convert_time(float(end) * 1000)
                    sentence = line.split(']:')[1].strip()

                    # 添加到列表中
                    names.append(subname)
                    sentences.append(sentence)
                except:
                    # 忽略无法处理的行
                    continue

    # 将数据写入CSV文件
    csv_file = 'dataset/IEMOCAP/transcription.csv'
    # CSV文件的列名
    columns = ['name', 'sentence']
    # 创建数据矩阵
    data = np.column_stack([names, sentences])

    # 创建DataFrame并设置列类型为字符串
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)

    # 保存DataFrame到CSV文件，不包含索引
    df.to_csv(csv_file, index=False)


# 定义生成转录文件的函数
def generate_transcription_files_CMUMOSI():
    # 读取pkl文件
    pkl_path = 'dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'  # 指定pkl文件路径
    # 解析pickle文件，加载数据
    names, sentences = [], []
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, valVid, testVid = pickle.load(open(pkl_path, "rb"),
                                                                                                  encoding='latin1')

    # 提取并整理姓名和句子信息
    for vid in videoIDs:
        uids = videoIDs[vid]
        sens = videoSentences[vid]
        # 确保姓名和句子数量匹配
        assert len(uids) == len(sens)
        for ii in range(len(uids)):
            names.append(uids[ii])
            sentences.append(sens[ii])

    # 写入csv文件
    csv_file = 'dataset/CMUMOSI/transcription.csv'  # 指定csv文件路径
    # 定义列名
    columns = ['name', 'sentence']
    # 构建数据矩阵
    data = np.column_stack([names, sentences])
    # 创建DataFrame对象
    df = pd.DataFrame(data=data, columns=columns)
    # 将所有列转换为字符串类型
    df[columns] = df[columns].astype(str)
    # 保存到csv文件，不包含索引
    df.to_csv(csv_file, index=False)


# 定义一个函数generate_transcription_files_CMUMOSEI，用于处理CMUMOSEI数据集的转录文件
def generate_transcription_files_CMUMOSEI():
    # 读取pkl文件
    pkl_path = 'dataset/CMUMOSEI/CMUMOSEI_features_raw_2way.pkl'
    names, sentences = [], []  # 初始化空列表，用于存储用户ID和句子

    # 加载pkl文件中的数据
    with open(pkl_path, "rb") as f:
        videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, valVid, testVid = pickle.load(f,
                                                                                                      encoding='latin1')

    # 遍历videoIDs字典，将用户ID和对应句子添加到列表中
    for vid in videoIDs:
        uids = videoIDs[vid]
        sens = videoSentences[vid]
        assert len(uids) == len(sens), "用户ID和句子数量不匹配"
        for ii in range(len(uids)):
            names.append(uids[ii])
            sentences.append(sens[ii])

    # 将数据写入csv文件
    csv_file = 'dataset/CMUMOSEI/transcription.csv'
    columns = ['name', 'sentence']  # 定义csv文件的列名
    data = np.column_stack([names, sentences])  # 横向堆叠名字和句子列表
    df = pd.DataFrame(data=data, columns=columns)  # 创建DataFrame对象

    # 将DataFrame中所有列转换为字符串类型
    df[columns] = df[columns].astype(str)

    # 保存DataFrame到csv文件，不包含索引
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    import fire

    fire.Fire()