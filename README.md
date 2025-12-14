# GAN实验详细使用说明

## 目录
1. [环境配置](#环境配置)
2. [数据准备](#数据准备)
3. [训练模型](#训练模型)
4. [评估模型](#评估模型)
5. [生成图像](#生成图像)
6. [结果分析](#结果分析)
7. [超参数调优](#超参数调优)
8. [常见问题](#常见问题)

---

## 环境配置

### 1. 安装依赖

```bash

# 安装PyTorch（根据你的CUDA版本选择）
# CPU版本
pip install torch torchvision

# GPU版本 (CUDA 11.3)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 验证安装

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

---

## 数据准备

### 1. 下载数据集

**Anime Face数据集**:
- 链接: https://www.modelscope.cn/datasets/yanghaitao/AnimeFace128
- 大小: 57221张动漫头像


### 2. 组织数据目录

对于Anime Face数据集：
```
project/
├── data/
│   └── faces/           # 注意：只有一个子文件夹
│       ├── 0000fdee4208b8b7e12074c920bc6166-0.jpg
│       ├── 0001a0fca4e9d2193afea712421693be-0.jpg
│       └── ...
├── main.py
├── model.py
└── ...
```

**重要**: PyTorch的`ImageFolder`要求数据目录下有至少一个子文件夹！

### 3. 数据验证

```python
import torchvision as tv
from torch.utils.data import DataLoader

# 检查数据集
dataset = tv.datasets.ImageFolder('data/')
print(f"数据集大小: {len(dataset)}")
print(f"类别数: {len(dataset.classes)}")

# 查看一张图片
img, label = dataset[0]
print(f"图片尺寸: {img.size}")
```

---

## 训练模型

### 1. 基础训练

```bash
# 使用GPU训练，启用可视化
python main.py train --gpu --vis=True

# 使用CPU训练，不启用可视化
python main.py train --nogpu --vis=False
```

### 2. 启动可视化服务（可选）

在另一个终端运行：
```bash
python -m visdom.server
```

然后在浏览器打开: http://localhost:8097

### 3. 自定义训练参数

```bash
# 示例1: 快速测试（少量epoch）
python main.py train --gpu --max_epoch=50 --save_every=5 --eval_every=5

# 示例2: 小batch size（显存不足时）
python main.py train --gpu --batch_size=64 --num_workers=2

# 示例3: 完整训练
python main.py train --gpu \
    --batch_size=256 \
    --lr1=2e-4 \
    --lr2=2e-4 \
    --max_epoch=200 \
    --save_every=10 \
    --eval_every=5
```

### 4. 训练输出说明

训练过程中会：
- 在`imgs/`目录保存生成的图像
- 在`checkpoints/`目录保存模型
- 在`logs/`目录保存训练日志
- 在visdom界面显示实时损失和图像（如果启用）

```
训练输出示例：
Epoch 1/200
100%|███████████| 195/195 [02:15<00:00,  1.44it/s]
Epoch 1 完成 - D_loss: 1.3845, G_loss: 2.1234

开始评估模型...
Inception Score: 2.34 ± 0.12
FID Score: 156.3
```

---

## 评估模型

### 1. 训练中自动评估

评估会在训练过程中自动进行（每`eval_every`个epoch）：
- 计算Inception Score (IS)
- 计算Fréchet Inception Distance (FID)
- 结果保存在`logs/eval_metrics.txt`

### 2. 单独评估模型

如果你想评估已保存的模型：

```python
# 在main.py中添加evaluate函数
import fire

def evaluate(**kwargs):
    """评估已保存的模型"""
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device = t.device('cuda') if opt.gpu else t.device('cpu')
    
    # 加载数据
    transforms = tv.transforms.Compose([...])
    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset, ...)
    
    # 加载模型
    netg = NetG(opt)
    netg.load_state_dict(t.load(opt.netg_path))
    netg.to(device)
    
    # 评估
    metrics = evaluate_model(netg, dataloader, device, opt)
    print(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")
    print(f"FID: {metrics['fid']:.4f}")

# 运行
python main.py evaluate --gpu \
    --netg-path=checkpoints/netg_200.pth \
    --num_eval_samples=5000
```

### 3. 理解评估指标

**Inception Score (IS)**:
- 范围: 1~10+
- 越高越好
- 衡量生成图像的质量和多样性
- 对于动漫头像，IS > 3.5 通常表示较好的质量

**FID Score**:
- 范围: 0~500+
- 越低越好
- 衡量生成分布与真实分布的距离
- 对于动漫头像，FID < 60 通常表示较好的质量

---

## 生成图像

### 1. 使用训练好的模型生成

```bash
# 基础生成（需要预先下载预训练模型）
python main.py generate --gpu \
    --netd-path=checkpoints/netd_200.pth \
    --netg-path=checkpoints/netg_200.pth \
    --gen-img=result.png \
    --gen-num=64

# 生成更多图片
python main.py generate --gpu \
    --netd-path=checkpoints/netd_200.pth \
    --netg-path=checkpoints/netg_200.pth \
    --gen-img=result_large.png \
    --gen-num=256 \
    --gen-search-num=2048
```

### 2. 参数说明

- `gen-num`: 最终保存的图片数量
- `gen-search-num`: 生成候选图片的数量（会从中选出最好的）
- `gen-mean`: 噪声均值（默认0）
- `gen-std`: 噪声标准差（默认1）

### 3. 生成多样性调整

```bash
# 更随机的生成（增大标准差）
python main.py generate --gpu \
    --netg-path=checkpoints/netg_200.pth \
    --netd-path=checkpoints/netd_200.pth \
    --gen-std=1.5

# 更集中的生成（减小标准差）
python main.py generate --gpu \
    --netg-path=checkpoints/netg_200.pth \
    --netd-path=checkpoints/netd_200.pth \
    --gen-std=0.7
```

---

## 结果分析

### 1. 运行分析脚本

```bash
python analyze.py
```

这会生成：
- `analysis/training_curves.png` - 训练损失曲线
- `analysis/metrics_evolution.png` - IS和FID演变
- `analysis/evolution_comparison.png` - 不同epoch的图像对比
- `analysis/hyperparameter_analysis.png` - 超参数影响分析
- `analysis/best_samples.png` - 最佳生成样本
- `analysis/summary_report.md` - 实验总结

### 2. 查看训练日志

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
df = pd.read_csv('logs/training_log.txt')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(df['D_Loss'], label='D Loss', alpha=0.7)
plt.plot(df['G_Loss'], label='G Loss', alpha=0.7)
plt.legend()
plt.show()
```

### 3. 查看评估指标

```bash
cat logs/eval_metrics.txt
```

输出示例：
```
Epoch 5: IS=2.34±0.12, FID=156.3
Epoch 10: IS=2.67±0.15, FID=124.5
Epoch 15: IS=2.98±0.14, FID=102.3
...
```

---

## 超参数调优

### 1. 学习率调优

```bash
# 实验1: 低学习率
python main.py train --gpu --lr1=1e-4 --lr2=1e-4 --env=GAN_lr_low

# 实验2: 标准学习率
python main.py train --gpu --lr1=2e-4 --lr2=2e-4 --env=GAN_baseline

# 实验3: 高学习率
python main.py train --gpu --lr1=5e-4 --lr2=5e-4 --env=GAN_lr_high
```

### 2. Batch Size调优

```bash
# 实验1: 小batch
python main.py train --gpu --batch_size=64 --env=GAN_bs64

# 实验2: 中batch
python main.py train --gpu --batch_size=128 --env=GAN_bs128

# 实验3: 大batch
python main.py train --gpu --batch_size=256 --env=GAN_bs256
```

### 3. 训练频率调优

```bash
# 实验1: 频繁更新生成器
python main.py train --gpu --d_every=1 --g_every=1 --env=GAN_freq_1_1

# 实验2: 平衡更新（推荐）
python main.py train --gpu --d_every=1 --g_every=5 --env=GAN_freq_1_5

# 实验3: 少更新生成器
python main.py train --gpu --d_every=1 --g_every=10 --env=GAN_freq_1_10
```

### 4. 网络容量调优

```bash
# 实验1: 小网络
python main.py train --gpu --ngf=32 --ndf=32 --env=GAN_small

# 实验2: 标准网络
python main.py train --gpu --ngf=64 --ndf=64 --env=GAN_baseline

# 实验3: 大网络
python main.py train --gpu --ngf=128 --ndf=128 --env=GAN_large
```

### 5. 记录实验结果

创建`logs/hyperparameter_experiments.csv`：
```csv
experiment,lr_g,lr_d,batch_size,g_every,nz,ngf,ndf,IS,FID,notes
baseline,2e-4,2e-4,256,5,100,64,64,3.89,52.1,最佳配置
lr_low,1e-4,1e-4,256,5,100,64,64,3.45,72.3,训练稳定但慢
lr_high,5e-4,5e-4,256,5,100,64,64,2.98,95.6,训练不稳定
...
```

---

## 常见问题

### 1. CUDA内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小batch size
python main.py train --gpu --batch_size=64

# 减小worker数量
python main.py train --gpu --num_workers=2

# 使用CPU（如果GPU内存真的不够）
python main.py train --nogpu --batch_size=128
```

### 2. 训练不稳定

**症状**: 损失剧烈波动，生成质量时好时坏

**解决方案**:
- 降低学习率: `--lr1=1e-4 --lr2=1e-4`
- 调整训练频率: `--g_every=10`
- 减小batch size: `--batch_size=128`

### 3. 模式崩溃

**症状**: 生成器只生成少数几种相似的图像

**解决方案**:
- 增加判别器训练频率: `--d_every=1 --g_every=5`
- 尝试不同的学习率比例: `--lr1=1e-4 --lr2=2e-4`
- 增加噪声维度: `--nz=200`

### 4. 生成图像模糊

**症状**: 图像缺乏细节，看起来模糊

**解决方案**:
- 增加训练轮数: `--max_epoch=300`
- 增加网络容量: `--ngf=128 --ndf=128`
- 调整学习率: `--lr1=1e-4`

### 5. 训练速度慢

**问题**: 训练时间过长

**优化方案**:
```bash
# 增加num_workers（数据加载并行）
python main.py train --gpu --num_workers=8

# 使用更大的batch size（如果显存允许）
python main.py train --gpu --batch_size=512

# 减少评估频率
python main.py train --gpu --eval_every=10

# 使用混合精度训练（需要修改代码）
# 在训练循环中使用torch.cuda.amp
```

### 6. Visdom可视化问题

**问题**: 无法连接到Visdom服务器

**解决方案**:
```bash
# 确保Visdom服务正在运行
python -m visdom.server

# 如果端口被占用，更改端口
python -m visdom.server -port 8098

# 在训练时指定端口
python main.py train --gpu --vis=True --port=8098

# 或者关闭可视化
python main.py train --gpu --vis=False
```

### 7. 数据加载错误

**问题**: `RuntimeError: Found 0 files in subfolders`

**原因**: ImageFolder要求数据目录下有子文件夹

**解决方案**:
```bash
# 正确的目录结构
data/
└── faces/          # 必须有这个子文件夹
    ├── img1.jpg
    └── img2.jpg

# 错误的目录结构
data/
├── img1.jpg       # 不能直接放在data下
└── img2.jpg
```

---

## 进阶技巧

### 1. 断点续训

```python
# 在main.py中使用预训练模型
python main.py train --gpu \
    --netd-path=checkpoints/netd_100.pth \
    --netg-path=checkpoints/netg_100.pth
```

### 2. 学习率衰减

在训练代码中添加学习率调度器：
```python
scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.5)
scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)

# 在每个epoch后
scheduler_g.step()
scheduler_d.step()
```

### 3. 梯度裁剪

防止梯度爆炸：
```python
torch.nn.utils.clip_grad_norm_(netg.parameters(), max_norm=5.0)
torch.nn.utils.clip_grad_norm_(netd.parameters(), max_norm=5.0)
```

### 4. 早停法

```python
best_fid = float('inf')
patience = 20
counter = 0

if fid < best_fid:
    best_fid = fid
    counter = 0
    # 保存最佳模型
else:
    counter += 1
    if counter >= patience:
        print("Early stopping!")
        break
```

---

## 实验建议

### 初学者流程
1. 先用小数据集和少量epoch测试（如50个epoch）
2. 确认代码能正常运行后，再进行完整训练
3. 关注IS和FID指标的变化趋势
4. 保存多个checkpoint以便对比

### 完整实验流程
1. **Baseline训练** (2-3小时): 使用默认参数训练200 epoch
2. **超参数调优** (6-8小时): 测试不同学习率、batch size等
3. **结果分析** (1小时): 运行分析脚本，生成图表
4. **撰写报告** (2-3小时): 基于模板完成实验报告

### 推荐配置
- GPU: NVIDIA GTX 1080 Ti 或更好
- 显存: 至少 8GB
- 内存: 至少 16GB
- 训练时间: 约 2-3 小时 (200 epochs)

---

## 资源链接

- **PyTorch文档**: https://pytorch.org/docs/
- **DCGAN论文**: https://arxiv.org/abs/1511.06434
- **GAN教程**: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- **Visdom文档**: https://github.com/fossasia/visdom

---

## 联系与反馈

如有问题，请：
1. 查看常见问题部分
2. 检查日志文件中的错误信息
3. 在GitHub上提issue

祝实验顺利！🎉
