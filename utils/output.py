import torch
import torch.nn as nn
import os

# 定义日志文件的路径
def logset(log_file):
    log_file = log_file
    # 写入标题信息到日志文件
    if os.path.exists(log_file):
        os.remove(log_file)  # 如果文件已存在，先删除它
    with open(log_file, 'a') as f:
        f.write("Epoch, Loss, CORR, Train/Test\n")

import os

# 定义日志函数
def logset(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        f.write('Epoch, Loss, RMSE, CORR, Phase\n')  # 创建文件并写入表头

def log_output(log_file, epoch, avg_loss, epoch_rmse, avg_corr, phase):
    with open(log_file, 'a') as f:
        f.write(f"{epoch + 1}, {avg_loss:.4f}, {epoch_rmse:.4f}, {avg_corr:.4f}, {phase}\n")
