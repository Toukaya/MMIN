import os
import numpy as np

def read_results(file):
    # 初始化一个空列表，用于存储结果
    ans = []

    # 打开文件并读取所有行，存储在lines列表中
    lines = open(file).readlines()

    # 遍历文件中的每一行
    for line in lines:
        # 如果行不以'0'开头，跳过此行
        if not line.startswith('0'):
            continue

        # 提取以'0'开头的行，去除首尾空白符，然后按制表符分割，将结果转换为浮点数列表并添加到ans
        ans.append(list(map(lambda x: float(x), line.strip().split('\t'))))

    # 将ans列表转换为numpy数组，并将其数据类型转换为浮点数
    data = np.array(ans).astype(np.float)

    # 检查数据的行数是否为24，确保数据正确
    assert data.shape[0] == 24

    # 分割数据，前10行为验证数据（val_data）
    val_data = data[0: 10]

    # 分割数据，第12行至第22行为测试数据（tst_data）
    tst_data = data[12: 22]

    # 返回验证数据和测试数据
    return val_data, tst_data


def combine(result1, result2):  # 定义一个名为combine的函数，接收两个参数result1和result2
    result = result1 * (result1 >= result2) + result2 * (
                result1 < result2)  # 计算组合结果：如果result1大于等于result2，则取result1，否则取result2
    return result  # 返回计算得到的组合结果


# 定义一个函数，将两个文件的结果合并并写入到output文件中
def combine_file(file1, file2, output):
    # 从file1读取验证集和测试集数据
    val_data1, tst_data1 = read_results(file1)
    # 从file2读取验证集和测试集数据
    val_data2, tst_data2 = read_results(file2)

    # 合并两个文件的验证集数据
    val_data = combine(val_data1, val_data2)

    # 计算验证集数据的均值和标准差，并扩展维度
    val_mean = np.expand_dims(np.mean(val_data, axis=0), 0)
    val_std = np.expand_dims(np.std(val_data, axis=0), 0)

    # 将验证集数据、均值和标准差垂直堆叠
    val_data = np.vstack([val_data, val_mean, val_std])

    # 合并两个文件的测试集数据
    tst_data = combine(tst_data1, tst_data2)

    # 计算测试集数据的均值和标准差，并扩展维度
    tst_mean = np.expand_dims(np.mean(tst_data, axis=0), 0)
    tst_std = np.expand_dims(np.std(tst_data, axis=0), 0)

    # 将测试集数据、均值和标准差垂直堆叠
    tst_data = np.vstack([tst_data, tst_mean, tst_std])

    # 打开output文件准备写入
    f = open(output, 'w')

    # 写入文件名
    f.write(output.split('/')[-1] + '\n')

    # 写入验证集数据
    f.write('val:\n')
    # 遍历验证集数据，格式化并写入
    for d in val_data:
        line = '\t'.join(list(map(lambda x:'{:.4f}'.format(x), d))) + '\n'
        f.write(line)

    # 提取验证集的均值和标准差，计算acc、uar和f1
    val_mean = val_mean[0]
    val_std = val_std[0]
    acc = '{:.4f}±{:.4f}'.format(val_mean[0], val_std[0])
    uar = '{:.4f}±{:.4f}'.format(val_mean[1], val_std[1])
    f1 = '{:.4f}±{:.4f}'.format(val_mean[2], val_std[2])
    # 写入验证集结果
    f.write('VAL result:\nacc %s uar %s f1 %s\n\n' % (acc, uar, f1))

    # 写入测试集数据
    f.write('tst:\n')
    # 遍历测试集数据，格式化并写入
    for d in tst_data:
        line = '\t'.join(list(map(lambda x:'{:.4f}'.format(x), d))) + '\n'
        f.write(line)

    # 提取测试集的均值和标准差，计算acc、uar和f1
    tst_mean = tst_mean[0]
    tst_std = tst_std[0]
    acc = '{:.4f}±{:.4f}'.format(tst_mean[0], tst_std[0])
    uar = '{:.4f}±{:.4f}'.format(tst_mean[1], tst_std[1])
    f1 = '{:.4f}±{:.4f}'.format(tst_mean[2], tst_std[2])
    # 写入测试集结果
    f.write('TEST result:\nacc %s uar %s f1 %s\n' % (acc, uar, f1))
    
# print(val_data)
# print(tst_data)
root = 'today_tasks/results'
save_root = 'today_tasks/results_combine'
# run_idx1 = 'A_ts_Adnn512,256,128_lossBCE_kd1.0_temp2.0_ce0.0_mmd0.0_run1'
# run_idx2 = 'A_ts_Adnn512,256,128_lossBCE_kd1.0_temp2.0_ce0.0_mmd0.0_run2'
# out = 'A_ts_Adnn512,256,128_lossBCE_kd1.0_temp2.0_ce0.0_mmd0.0'
# combine_file(os.path.join(root, run_idx1), os.path.join(root, run_idx2), os.path.join(save_root, out))
total_file = os.listdir(root)
name_set = set()
for file in total_file:
    name = '_'.join(file.split('_')[:-1])
    name_set.add(name)

for name in name_set:
    run_idx1 = name + '_run1'
    run_idx2 = name + '_run2'
    out = name
    combine_file(os.path.join(root, run_idx1), os.path.join(root, run_idx2), os.path.join(save_root, out))