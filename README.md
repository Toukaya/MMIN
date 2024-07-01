# MMIN

此代码库实现了缺失模态想象网络（MMIN），用于以下论文：
"带有不确定缺失模态的情绪识别中的缺失模态想象网络"

# 环境

```
python 3.7.0
pytorch >= 1.0.0
```

# 使用方法

首先，您应该在`data/config`中更改数据文件夹路径，并按照`preprocess/`中的代码预处理您的数据。

特征的预处理是手工完成的，我们将在下一个更新中使其成为自动运行脚本。您可以下载预处理后的特征来运行代码。

+ 对IEMOCAP进行MMIN训练：

    首先，使用所有音频、视觉和词汇模态训练一个模型融合模型作为预训练的编码器。

    ```bash
    bash scripts/CAP_utt_fusion.sh AVL [num_of_expr] [GPU_index]
    ```

    然后

    ```bash
    bash scripts/CAP_mmin.sh [num_of_expr] [GPU_index]
    ```

+ 对MSP-improv进行MMIN训练：

    ```bash
    bash scripts/MSP_utt_fusion.sh AVL [num_of_expr] [GPU_index]
    ```

    ```bash
    bash scripts/MSP_mmin.sh [num_of_expr] [GPU_index]
    ```

请注意，您可以使用shell脚本中定义的默认超参数运行代码，如需更改这些参数，请参考`options/get_opt.py`和您选择的每个模型的`modify_commandline_options`方法。

# 下载特征
百度云链接
IEMOCAP A V L模态特征
链接：https://pan.baidu.com/s/1WmuqNlvcs5XzLKfz5i4iqQ 提取码：gn6w

Google Drive 链接
https://drive.google.com/file/d/1X5wjY-eMnLPV2qkFaaRi9ZPkrMcCAv7Q/view?usp=sharing

百度云链接
MSP A V L模态特征
链接：https://pan.baidu.com/s/17E44x84pdR2AQIts0aJfKg 提取码：6dzq

# 许可证
MIT许可证。

版权所有 (c) 2021 中国人民大学信息学院AIM3-RUC实验室。

# 引用
如果您发现我们的论文和此代码有用，请考虑引用
```
@inproceedings{zhao2021missing,
  title={Missing modality imagination network for emotion recognition with uncertain missing modalities},
  author={Zhao, Jinming and Li, Ruichen and Jin, Qin},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={2608--2618},
  year={2021}
}
```
