# *_*coding:utf-8 *_*
import os
import socket

def get_host_ip():
    """
    获取本地主机IP地址的函数
    使用socket创建UDP套接字并连接到指定的IP和端口，然后获取套接字的本地地址
    """

    try:
        # 创建一个IPv4类型的UDP套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 连接到指定的IP（10.0.0.1）和端口（8080），此操作不会发送数据，仅用于获取本地IP
        s.connect(('10.0.0.1', 8080))

        # 获取套接字的本地地址信息，其中[0]代表IP地址
        ip = s.getsockname()[0]
    finally:
        # 关闭套接字
        s.close()

    # 返回获取到的IP地址
    return ip

############ For LINUX ##############
# path
DATA_DIR = {
	'CMUMOSI': '/share/home/lianzheng/gcnet-master/dataset/CMUMOSI',   # for nlpr
	'CMUMOSEI': '/share/home/lianzheng/gcnet-master/dataset/CMUMOSEI',# for nlpr
	'IEMOCAPSix': '/share/home/lianzheng/gcnet-master/dataset/IEMOCAP', # for nlpr
	'IEMOCAPFour': '/share/home/lianzheng/gcnet-master/dataset/IEMOCAP', # for nlpr
}
PATH_TO_RAW_AUDIO = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subaudio'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
}
PATH_TO_RAW_FACE = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subvideofaces'), # without openfac
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideofaces'),
}
PATH_TO_TRANSCRIPTIONS = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription.csv'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription.csv'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'transcription.csv'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'features'),
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}
# dir
SAVED_ROOT = os.path.join('../saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
