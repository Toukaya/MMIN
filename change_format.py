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
import soundfile as sf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def name2feat(feature_root):
    # 获取feature_root目录下的文件或文件夹名称
    names = os.listdir(feature_root)
    # 初始化一个空列表，用于存储特征
    features = []
    # 初始化特征维度为-1
    feature_dim = -1

    # 使用tqdm进行进度条显示，遍历names列表
    for ii, name in tqdm.tqdm(enumerate(names)):
        # 初始化一个空列表，用于存储当前文件或文件夹的特征
        feature = []
        # 构建当前文件或文件夹的完整路径
        feature_path = os.path.join(feature_root, name)

        # 如果是文件
        if os.path.isfile(feature_path):
            # 尝试加载特征文件
            try:
                single_feature = np.load(feature_path)
            # 如果遇到EOFError（文件结束错误）
            except EOFError:
                print(f"EOFError: No data left in file {feature_path}")
                # 设置特征为空数组
                single_feature = np.array([])

            # 去除单个特征的额外维度
            single_feature = single_feature.squeeze()
            # 将单个特征添加到feature列表中
            feature.append(single_feature)
            # 更新最大特征维度
            feature_dim = max(feature_dim, single_feature.shape[-1])

        # 如果是文件夹
        else:
            # 获取文件夹下所有文件名
            facenames = os.listdir(feature_path)
            # 按名称排序
            for facename in sorted(facenames):
                # 构建面部特征文件的完整路径
                facefeat_path = os.path.join(feature_path, facename)
                # 尝试加载面部特征文件
                try:
                    facefeat = np.load(facefeat_path)
                # 如果遇到EOFError（文件结束错误）
                except EOFError:
                    print(f"EOFError: No data left in file {facefeat_path}")
                    # 设置特征为空数组
                    facefeat = np.array([])

                # 更新最大特征维度
                feature_dim = max(feature_dim, facefeat.shape[-1])
                # 将面部特征添加到feature列表中
                feature.append(facefeat)

        # 将feature列表转换为一维数组
        single_feature = np.array(feature).squeeze()

        # 如果一维数组长度为0，用全零数组填充
        if len(single_feature) == 0:
            single_feature = np.zeros((feature_dim,))
        # 如果一维数组是二维的，取平均值转为一维
        elif len(single_feature.shape) == 2:
            single_feature = np.mean(single_feature, axis=0)

        # 将处理后的特征添加到features列表中
        features.append(single_feature)

    # 打印特征信息
    print(f'Input feature {os.path.basename(feature_root)} ===> dim is {feature_dim}; No. sample is {len(names)}')

    # 检查名字和特征数量是否一致
    assert len(names) == len(features), f'Error: len(names) != len(features)'

    # 创建字典，键为文件或文件夹名称，值为对应的特征
    name2feats = {}
    # 遍历names列表
    for ii in range(len(names)):
        # 处理文件名，移除.npy或.npz后缀
        name = names[ii]
        if name.endswith('.npy') or name.endswith('.npz'):
            name = name[:-4]

        # 将名字和特征添加到字典中
        name2feats[name] = features[ii]

    # 返回name2feats字典
    return name2feats


# 定义一个函数change_feat_format_cmumosei，用于转换CMUMOSEI数据集特征格式
def change_feat_format_cmumosei():
    # 定义数据路径
    label_pkl = '../dataset/CMUMOSEI/CMUMOSEI_features_raw_2way.pkl'
    feat_root = '../dataset/CMUMOSEI/features'
    save_root = './CMUMOSEI_features_2021'

    # 定义不同特征名称
    nameA = 'wav2vec-large-c-UTT'
    nameV = 'manet_UTT'
    nameL = 'deberta-large-4-UTT'

    # 加载数据标签
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, valVid, testVid = pickle.load(open(label_pkl, "rb"),
                                                                                                 encoding='latin1')

    # 获取不同特征的路径
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)

    # 定义函数，根据特征名称获取特征
    def name2feat(featroot):
        # 省略具体实现，假设该函数返回对应名称的特征
        pass

    # 遍历训练、验证和测试数据
    for item1, item2 in [(trainVid, 'trn'), (valVid, 'val'), (testVid, 'tst')]:
        # 初始化特征列表
        all_A = []
        all_V = []
        all_L = []
        label = []
        int2name = []

        # 遍历视频ID
        for vid in tqdm.tqdm(item1):
            # 获取视频ID对应的名称
            int2name.extend(videoIDs[vid])
            # 获取视频ID对应的标签
            label.extend(videoLabels[vid])

            # 遍历每个视频的帧
            for ii in range(len(videoIDs[vid])):
                # 获取帧名称
                name = videoIDs[vid][ii]
                # 获取各特征
                featA = name2featA[name]
                featV = name2featV[name]
                featL = name2featL[name]

                # 将特征添加到列表
                all_A.append(featA)
                all_V.append(featV)
                all_L.append(featL)

        # 转换特征列表为numpy数组
        all_A = np.array(all_A)
        all_V = np.array(all_V)
        all_L = np.array(all_L)

        # 保存特征到指定路径
        save_path = f"{save_root}/A/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_A)

        save_path = f"{save_root}/V/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_V)

        save_path = f"{save_root}/L/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_L)

        # 保存标签和名称到指定路径
        save_path = f"{save_root}/target/1/{item2}_label.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, label)

        save_path = f"{save_root}/target/1/{item2}_int2name.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, int2name)


# 定义一个函数change_feat_format_cmumosi，用于转换CMUMOSI数据集的特征格式
def change_feat_format_cmumosi():
    # 定义数据路径和保存路径
    label_pkl = '../dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'  # 标签pickle文件路径
    feat_root = '../dataset/CMUMOSI/features'  # 特征根目录
    save_root = './CMUMOSI_features_2021'  # 保存处理后特征的目录

    # 定义不同特征的名称
    nameA = 'wav2vec-large-c-UTT'  # 音频特征名称
    nameV = 'manet_UTT'  # 视频特征名称
    nameL = 'deberta-large-4-UTT'  # 文本特征名称

    # 加载标签数据
    videoIDs, videoLabels, videoSpeakers, videoSentence, trainVid, valVid, testVid = pickle.load(open(label_pkl, "rb"),
                                                                                                 encoding='latin1')

    # 获取不同特征的路径
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)

    # 定义函数，根据特征根目录获取特征
    def name2feat(feature_root):
        # 省略具体实现，该函数用于根据特征名称和根目录获取特征数据
        pass

    # 分别获取音频、视频和文本特征
    name2featA = name2feat(featrootA)
    name2featV = name2feat(featrootV)
    name2featL = name2feat(featrootL)

    # 遍历训练、验证和测试数据
    for item1, item2 in [(trainVid, 'trn'), (valVid, 'val'), (testVid, 'tst')]:
        # 初始化特征列表
        all_A = []
        all_V = []
        all_L = []
        label = []
        int2name = []

        # 遍历每个视频ID
        for vid in tqdm.tqdm(item1):
            # 获取视频ID对应的名称
            int2name.extend(videoIDs[vid])
            # 获取视频标签
            label.extend(videoLabels[vid])
            # 遍历每个样本
            for ii in range(len(videoIDs[vid])):
                name = videoIDs[vid][ii]
                # 获取音频、视频和文本特征
                featA = name2featA[name]
                featV = name2featV[name]
                featL = name2featL[name]
                # 将特征添加到列表中
                all_A.append(featA)
                all_V.append(featV)
                all_L.append(featL)

        # 转换为numpy数组
        all_A = np.array(all_A)
        all_V = np.array(all_V)
        all_L = np.array(all_L)

        # 保存音频特征
        save_path = f"{save_root}/A/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_A)

        # 保存视频特征
        save_path = f"{save_root}/V/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_V)

        # 保存文本特征
        save_path = f"{save_root}/L/1/{item2}.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, all_L)

        # 保存标签
        save_path = f"{save_root}/target/1/{item2}_label.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, label)

        # 保存名称与索引映射
        save_path = f"{save_root}/target/1/{item2}_int2name.npy"
        save_temp = os.path.split(save_path)[0]
        if not os.path.exists(save_temp): os.makedirs(save_temp)
        np.save(save_path, int2name)


# 定义一个函数change_feat_format_iemocapfour，用于转换IEMOCAP数据集的特征格式
def change_feat_format_iemocapfour():
    # 定义数据路径
    label_pkl = '../dataset/IEMOCAP/IEMOCAP_features_raw_4way.pkl'
    feat_root = '../dataset/IEMOCAP/features'
    save_root = './IEMOCAPFOUR_features_2021'

    # 定义不同特征名称
    nameA = 'wav2vec-large-c-UTT'
    nameV = 'manet_UTT'
    nameL = 'deberta-large-4-UTT'

    # 加载标签文件
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid = pickle.load(open(label_pkl, "rb"),
                                                                                          encoding='latin1')

    # 获取不同特征的根目录
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)

    # 创建字典，将特征名称映射到其对应的特征文件
    name2featA = name2feat(featrootA)
    name2featV = name2feat(featrootV)
    name2featL = name2feat(featL)

    # 生成五个文件夹
    num_folder = 5
    vids = sorted(list(trainVid | testVid))

    # 根据视频ID的会话编号创建索引
    session_to_idx = {}
    for idx, vid in enumerate(vids):
        session = int(vid[4]) - 1
        if session not in session_to_idx: session_to_idx[session] = []
        session_to_idx[session].append(idx)
    # 检查是否正确分为五个文件夹
    assert len(session_to_idx) == num_folder, f'Must split into five folder'

    # 分割训练和测试数据
    train_test_idxs = []
    for ii in range(num_folder):  # ii in [0, 4]
        test_idxs = session_to_idx[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii: train_idxs.extend(session_to_idx[jj])
        train_test_idxs.append([train_idxs, test_idxs])

    # 遍历每个文件夹
    for ii in range(len(train_test_idxs)):
        train_idxs = train_test_idxs[ii][0]
        test_idxs = train_test_idxs[ii][1]

        # 分割训练和测试视频
        trainVid = np.array(vids)[train_idxs]
        testVid = np.array(vids)[test_idxs]

        # 处理训练和测试数据
        for item1, item2 in [(trainVid, 'trn'), (testVid, 'val'), (testVid, 'tst')]:
            # 收集utterance级别的特征
            all_A = []
            all_V = []
            all_L = []
            label = []
            int2name = []
            for vid in tqdm.tqdm(item1):
                int2name.extend(videoIDs[vid])
                label.extend(videoLabels[vid])
                for jj in range(len(videoIDs[vid])):
                    name = videoIDs[vid][jj]
                    featA = name2featA[name]
                    featV = name2featV[name]
                    featL = name2featL[name]
                    all_A.append(featA)
                    all_V.append(featV)
                    all_L.append(featL)
            # 转换为numpy数组
            all_A = np.array(all_A)
            all_V = np.array(all_V)
            all_L = np.array(all_L)

            # 保存特征
            save_path = f"{save_root}/A/{ii + 1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_A)

            save_path = f"{save_root}/V/{ii + 1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_V)

            save_path = f"{save_root}/L/{ii + 1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_L)

            # 保存标签和名称映射
            save_path = f"{save_root}/target/{ii + 1}/{item2}_label.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, label)

            save_path = f"{save_root}/target/{ii + 1}/{item2}_int2name.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, int2name)


# 定义一个函数change_feat_format_iemocapsix，用于处理IEMOCAP数据集的特征格式
def change_feat_format_iemocapsix():
    # 定义数据路径
    label_pkl = '../dataset/IEMOCAP/IEMOCAP_features_raw_6way.pkl'
    feat_root = '../dataset/IEMOCAP/features'
    save_root = './IEMOCAPSIX_features_2021'

    # 定义不同模型的特征名称
    nameA = 'wav2vec-large-c-UTT'
    nameV = 'manet_UTT'
    nameL = 'deberta-large-4-UTT'

    # 加载数据
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid = pickle.load(open(label_pkl, "rb"),
                                                                                          encoding='latin1')

    # 获取不同模型的特征路径
    featrootA = os.path.join(feat_root, nameA)
    featrootV = os.path.join(feat_root, nameV)
    featrootL = os.path.join(feat_root, nameL)

    # 创建字典，将视频ID映射到特征
    name2featA = name2feat(featrootA)
    name2featV = name2feat(featrootV)
    name2featL = name2feat(featrootL)

    # 将视频ID按会话分组，生成5个文件夹
    num_folder = 5
    vids = sorted(list(trainVid | testVid))

    session_to_idx = {}
    for idx, vid in enumerate(vids):
        session = int(vid[4]) - 1
        if session not in session_to_idx: session_to_idx[session] = []
        session_to_idx[session].append(idx)
    assert len(session_to_idx) == num_folder, f'必须分为5个文件夹'

    # 生成训练和测试数据的索引
    train_test_idxs = []
    for ii in range(num_folder): # ii in [0, 4]
        test_idxs = session_to_idx[ii]
        train_idxs = []
        for jj in range(num_folder):
            if jj != ii: train_idxs.extend(session_to_idx[jj])
        train_test_idxs.append([train_idxs, test_idxs])

    # 遍历每个文件夹
    for ii in range(len(train_test_idxs)):
        train_idxs = train_test_idxs[ii][0]
        test_idxs = train_test_idxs[ii][1]
        trainVid = np.array(vids)[train_idxs]
        testVid = np.array(vids)[test_idxs]

        # 处理训练和测试数据
        for item1, item2 in [(trainVid, 'trn'), (testVid, 'val'), (testVid, 'tst')]:
            # 收集每个样本的特征
            all_A = []
            all_V = []
            all_L = []
            label = []
            int2name = []
            for vid in tqdm.tqdm(item1):
                int2name.extend(videoIDs[vid])
                label.extend(videoLabels[vid])
                for jj in range(len(videoIDs[vid])):
                    name = videoIDs[vid][jj]
                    featA = name2featA[name]
                    featV = name2featV[name]
                    featL = name2featL[name]
                    all_A.append(featA)
                    all_V.append(featV)
                    all_L.append(featL)

            # 转换为numpy数组
            all_A = np.array(all_A)
            all_V = np.array(all_V)
            all_L = np.array(all_L)

            # 保存特征
            save_path = f"{save_root}/A/{ii + 1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_A)

            save_path = f"{save_root}/V/{ii + 1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_V)

            save_path = f"{save_root}/L/{ii + 1}/{item2}.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, all_L)

            # 保存标签
            save_path = f"{save_root}/target/{ii + 1}/{item2}_label.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, label)

            # 保存视频ID与名称的映射
            save_path = f"{save_root}/target/{ii + 1}/{item2}_int2name.npy"
            save_temp = os.path.split(save_path)[0]
            if not os.path.exists(save_temp): os.makedirs(save_temp)
            np.save(save_path, int2name)



if __name__ == '__main__':
    change_feat_format_cmumosi()


