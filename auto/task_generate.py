import os

def make_grid(params):
    # 初始化总长度为1，用于计算所有参数组合的总数
    total_length = 1
    # 遍历参数字典，将每个参数值的长度乘以总长度，得到所有可能的组合数
    for key, value in params.items():
        total_length *= len(value)

    # 创建一个空列表，用于存储结果网格
    ans = []
    # 用所有组合数填充列表，创建空字典作为网格行
    for _ in range(total_length):
        ans.append({})

    # 计算当前参数的组合数
    combo_num = total_length
    # 遍历参数字典，依次处理每个参数
    for key, value in params.items():
        # 更新当前参数的组合数，除以下一个参数的值的长度
        combo_num = combo_num // len(value)
        # 遍历当前网格的所有行
        for i in range(0, total_length, combo_num):
            # 遍历当前参数的值
            for j in range(combo_num):
                # 将当前参数的值填入网格中对应的位置
                ans[i + j][key] = value[i // combo_num % len(value)]

    # 返回生成的网格
    return ans


def make_task(parameters):
    # 根据参数生成超参数网格
    param_grid = make_grid(parameters)

    # 创建命令行模板，包含任务脚本和参数占位符
    template = 'sh ' + task_script + ' ' + ' '.join(['{' + key + '}' for key in parameters.keys()])

    # 生成所有可能的命令行组合
    total_cmd = []
    for param in param_grid:
        cmd = template.format(**param)  # 使用参数填充模板
        total_cmd.append(cmd)

    # 将GPU平均分配给每个命令
    cmd_with_gpu = []
    for i in range(len(avialable_gpus)):
        # 计算每个GPU应处理的任务数量
        task_num = len(total_cmd) / len(avialable_gpus)
        cmds = total_cmd[int(i * task_num):int((i + 1) * task_num)]
        for cmd in cmds:
            # 添加GPU编号到命令行
            cmd_with_gpu.append(cmd + ' ' + str(avialable_gpus[i]))

    # 为每个会话创建任务文件并写入命令
    for i in range(num_sessions):
        # 生成会话名称
        session_name = '{}_{}'.format(screen_name, i)
        # 生成任务文件路径
        task_file = os.path.join(auto_script_dir, f'{i}_task.sh')
        f = open(task_file, 'w')
        f.write('screen -dmS {}\n'.format(session_name))
        task_num = len(cmd_with_gpu) / num_sessions
        cmds = cmd_with_gpu[int(i * task_num):int((i + 1) * task_num)]
        for cmd in cmds:
            _cmd = "screen -x -S {} -p 0 -X stuff '{}\n'\n".format(session_name, cmd)
            f.write(_cmd)
        f.write("screen -x -S {} -p 0 -X stuff 'exit\n'\n".format(session_name))
   
if __name__ == '__main__':
    auto_script_dir = 'auto/tmp'  # 生成脚本路径
    script_root = 'auto/scripts'
    task_script = script_root + '/' + 'mmin.sh'  # 执行script路径
    avialable_gpus = [0, 1, 2, 3, 4, 5]  # 可用GPU有哪些
    num_sessions = 6  # 一共开多少个session同时执行
    avialable_gpus = avialable_gpus[:num_sessions]
    screen_name = 'mmin'
    parameters = {  # 一共有哪些参数
        'mse_weight': [0.1, 0.15, 0.2],
        'cycle_weight': [0.05, 0.1, 0.2],
        'run_idx': [1, 2]
    }
    make_task(parameters)

    for i in range(num_sessions):
        cmd = 'sh {}/{}_task.sh'.format(auto_script_dir, i)
        print(cmd)
        os.system(cmd)