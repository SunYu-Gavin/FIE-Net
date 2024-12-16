import torch
def compute_corr(y_true, y_pred):
    """ 计算 Pearson 相关系数 """
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)

    cov = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    std_y_true = torch.std(y_true)
    std_y_pred = torch.std(y_pred)
#    if std_y_true == 0 or std_y_pred == 0:
#        return torch.tensor(0.0)  # 相关系数为0
    corr = cov / (std_y_true * std_y_pred)
    return corr