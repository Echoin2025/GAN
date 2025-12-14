# 条件GAN (Conditional GAN) 实验指南

## 目录
1. [实验目标](#实验目标)
2. [理论基础](#理论基础)
3. [环境准备](#环境准备)
4. [数据准备](#数据准备)
5. [训练cGAN](#训练cgan)
6. [生成与评估](#生成与评估)
7. [对比分析](#对比分析)
8. [常见问题](#常见问题)

---

## 实验目标

1. ✅ 理解条件GAN的工作原理
2. ✅ 实现并训练条件GAN模型
3. ✅ 对比GAN和cGAN的生成效果
4. ✅ 分析条件生成的优势和局限
5. ✅ 撰写完整的对比分析报告

---

## 理论基础

### 什么是条件GAN？

**标准GAN**只能从随机噪声生成图像，无法控制生成什么内容。

**条件GAN**在生成过程中引入额外的条件信息（如类别标签），使生成过程可控。

### 核心思想

```
标准GAN:    x = G(z)           生成什么是随机的
条件GAN:    x = G(z, c)        生成内容由c控制

其中:
- z: 随机噪声
- c: 条件信息（如类别标签、文本描述等）
- x: 生成的图像
```

### 架构对比

#### 标准GAN
```
噪声z → 生成器G → 假图像x'
真图像x → 判别器D → 真假概率
假图像x' → 判别器D → 真假概率
```

#### 条件GAN
```
噪声z + 标签c → 生成器G → 假图像x'
真图像x + 标签c → 判别器D → 真假概率
假图像x' + 标签c → 判别器D → 真假概率
```

### 数学表达

**GAN目标函数:**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

**cGAN目标函数:**
```
min_G max_D V(D,G) = E_{x,c}[log D(x|c)] + E_{z,c}[log(1 - D(G(z|c)|c))]
```

关键区别：所有操作都以条件c为前提。

---

## 环境准备

### 文件结构

```
project/
├── model.py              # 标准GAN模型
├── main.py               # 标准GAN训练脚本
├── cgan_model.py         # 条件GAN模型（新增）
├── cgan_main.py          # 条件GAN训练脚本（新增）
├── compare_gan_cgan.py   # 对比分析脚本（新增）
├── analyze.py            # 分析脚本
├── data/                 # 数据集
│   └── faces/           # 图片文件
├── checkpoints/         # GAN模型
├── cgan_checkpoints/    # cGAN模型（新增）
├── imgs/                # GAN生成图片
├── cgan_imgs/           # cGAN生成图片（新增）
├── logs/                # GAN训练日志
├── cgan_logs/           # cGAN训练日志（新增）
└── comparison/          # 对比分析结果（新增）
```

### 依赖检查

所有依赖与标准GAN相同，无需额外安装。

---

## 数据准备

### 数据集要求

条件GAN需要**带标签的数据集**。有两种方式：

#### 方式1: 使用有类别的数据集

如果你的动漫头像数据按角色类型、发色等分类：

```
data/
    ├── class_0/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── class_1/
    │   ├── img3.jpg
    │   └── img4.jpg
    └── class_2/
        ├── img5.jpg
        └── img6.jpg
```

`ImageFolder`会自动为每个子文件夹分配一个类别标签。

#### 方式2: 人工划分类别

如果数据没有分类，可以手动创建子目录：

```bash
# 创建类别目录
mkdir -p data/faces/class_0
mkdir -p data/faces/class_1

# 将图片分配到不同类别（可以随机或按特征）
# 例如：前5000张放class_0，后5000张放class_1
```

**建议**：至少分成2-5个类别，太多类别会增加训练难度。

### 检查数据

```python
import torchvision as tv

dataset = tv.datasets.ImageFolder('data/')
print(f"类别数: {len(dataset.classes)}")
print(f"类别名称: {dataset.classes}")
print(f"每个类别的样本数:")
for i, class_name in enumerate(dataset.classes):
    count = sum(1 for _, label in dataset if label == i)
    print(f"  {class_name}: {count}")
```

---

## 训练cGAN

### 基础训练

```bash
# 使用默认参数训练
python cgan_main.py train --gpu --vis=True

# 自定义类别数（根据你的数据集）
python cgan_main.py train --gpu --num_classes=5

# 完整参数
python cgan_main.py train \
    --gpu \
    --num_classes=5 \
    --batch_size=256 \
    --max_epoch=200 \
    --lr1=2e-4 \
    --lr2=2e-4 \
    --save_every=10
```

### 重要参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| num_classes | 类别数量 | 根据数据集 |
| label_emb_dim | 标签嵌入维度 | 50 |
| use_simple_model | 使用简化模型 | False |
| nz | 噪声维度 | 100 |
| batch_size | 批大小 | 256 |

### 训练监控

训练过程中观察：

1. **损失曲线**：
   - D_Loss应逐渐稳定
   - G_Loss应逐渐下降
   - 相比GAN，cGAN损失可能更稳定

2. **生成样本**：
   - 检查不同类别是否有明显差异
   - 同类别内样本是否风格一致

3. **Visdom可视化**：
   ```bash
   # 在另一个终端
   python -m visdom.server
   # 浏览器打开 http://localhost:8097
   ```

### 训练时间估计

- GPU (RTX 3090): 约2-3小时 (200 epochs)
- GPU (GTX 1080 Ti): 约4-5小时
- CPU: 不推荐（太慢）

---

## 生成与评估

### 1. 基础生成（每个类别生成多张）

```bash
python cgan_main.py generate \
    --gpu \
    --netg-path=cgan_checkpoints/netg_200.pth \
    --num_classes=4 \
    --gen_num=50 \
    --gen_img=cgan_result.png
```

这会为每个类别生成 `gen_num // num_classes` 张图片。

### 2. 类别控制生成

```bash
python cgan_main.py generate_with_specific_labels \
    --gpu \
    --netg-path=cgan_checkpoints/netg_200.pth \
    --num_classes=4
```

生成 `cgan_by_class.png`，每行显示一个类别。

### 3. 与GAN对比

```bash
# 先生成GAN样本
python main.py generate \
    --gpu \
    --netg-path=checkpoints/netg_200.pth \
    --netd-path=checkpoints/netd_200.pth \
    --gen_img=gan_result.png

# 再生成cGAN样本
python cgan_main.py generate \
    --gpu \
    --netg-path=cgan_checkpoints/netg_200.pth \
    --gen_img=cgan_result.png

# 对比两张图片
```

### 4. 定量评估

虽然cGAN主要优势在可控性，但也可以评估质量：

```python
# 在cgan_main.py中添加评估功能
# 类似main.py中的evaluate_model函数
# 计算IS和FID
```

---

## 对比分析

### 自动对比分析

```bash
python compare_gan_cgan.py
```

这会生成：
1. 训练曲线对比图
2. 生成样本对比图
3. 详细的对比分析报告

### 手动对比清单

#### 1. 视觉质量对比

**观察点**：
- [ ] 图像清晰度：哪个更清晰？
- [ ] 细节丰富度：哪个细节更多？
- [ ] 色彩自然度：哪个色彩更自然？
- [ ] 整体真实感：哪个更像真实图片？

#### 2. 多样性对比

**GAN**：
- [ ] 生成的图像风格是否多样？
- [ ] 是否覆盖了数据集的主要模式？
- [ ] 是否出现模式崩溃？

**cGAN**：
- [ ] 不同类别间差异是否明显？
- [ ] 同类别内是否风格一致？
- [ ] 类别控制是否准确？

#### 3. 可控性对比

**GAN**：
- ❌ 无法指定生成什么
- ❌ 每次生成完全随机

**cGAN**：
- ✅ 可以指定生成的类别
- ✅ 同类别生成风格可预测

#### 4. 训练特性对比

| 特性 | GAN | cGAN |
|------|-----|------|
| 训练稳定性 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 训练速度 | 快 | 中等 |
| 收敛速度 | 中等 | 较快 |
| 模式崩溃风险 | 高 | 低 |

---

## 撰写对比报告

### 报告结构建议

```markdown
# GAN vs cGAN 对比实验报告

## 1. 实验背景
- 问题描述
- 实验目的

## 2. 理论基础
- GAN原理回顾
- cGAN的改进点
- 条件生成的意义

## 3. 实验设置
- 数据集描述
- 模型架构
- 超参数配置
- 训练环境

## 4. 实验过程
### 4.1 标准GAN训练
- 训练过程
- 遇到的问题
- 最终结果

### 4.2 条件GAN训练
- 数据标签准备
- 训练过程
- 类别控制效果

## 5. 实验结果
### 5.1 训练曲线对比
- 插入对比图
- 分析差异

### 5.2 生成样本对比
- 插入生成样本
- 视觉质量评价

### 5.3 定量评估
- IS对比
- FID对比
- 其他指标

## 6. 优缺点分析
### 6.1 GAN
- 优点
- 缺点
- 适用场景

### 6.2 cGAN
- 优点
- 缺点
- 适用场景

## 7. 结论
- 主要发现
- 实用价值
- 改进方向

## 8. 参考文献
```

### 关键分析点

1. **架构差异**：
   - 标签如何融入网络
   - 参数量变化
   - 计算复杂度

2. **训练差异**：
   - 收敛速度
   - 稳定性
   - 需要的epoch数

3. **生成质量**：
   - 主观评价（视觉）
   - 客观评价（IS, FID）
   - 可控性评价

4. **实用价值**：
   - 哪些场景适合GAN
   - 哪些场景适合cGAN
   - 实际应用建议

---

## 常见问题

### 1. cGAN训练不稳定

**问题**：损失剧烈波动，生成质量差

**解决**：
```bash
# 降低学习率
python cgan_main.py train --lr1=1e-4 --lr2=1e-4

# 增加判别器训练频率
python cgan_main.py train --d_every=1 --g_every=5

# 使用简化模型
python cgan_main.py train --use_simple_model=True
```

### 2. 类别控制不明显

**问题**：不同类别生成的图片看起来很相似

**原因**：
- 数据集类别划分不合理
- 标签嵌入维度太小
- 训练不充分

**解决**：
```bash
# 增加标签嵌入维度
python cgan_main.py train --label_emb_dim=100

# 延长训练
python cgan_main.py train --max_epoch=300

# 重新划分数据集类别（使类别间差异更大）
```

### 3. 内存不足

**问题**：`CUDA out of memory`

**解决**：
```bash
# 减小batch size
python cgan_main.py train --batch_size=128

# 使用简化模型
python cgan_main.py train --use_simple_model=True

# 减小标签嵌入维度
python cgan_main.py train --label_emb_dim=20
```

### 4. 数据集没有标签怎么办？

**方案1：随机分配**
```python
# 将数据随机分成N类
import os
import shutil
import random

images = os.listdir('data/faces')
random.shuffle(images)

n_classes = 5
imgs_per_class = len(images) // n_classes

for i in range(n_classes):
    os.makedirs(f'data/faces_labeled/class_{i}', exist_ok=True)
    start = i * imgs_per_class
    end = (i+1) * imgs_per_class if i < n_classes-1 else len(images)
    
    for img in images[start:end]:
        shutil.copy(
            f'data/faces/{img}',
            f'data/faces_labeled/class_{i}/{img}'
        )
```

**方案2：使用聚类**
```python
# 使用K-means对图像特征聚类
from sklearn.cluster import KMeans
import torch
import torchvision

# 提取图像特征
model = torchvision.models.resnet18(pretrained=True)
features = []
for img in images:
    # 提取特征
    feat = model(img)
    features.append(feat)

# 聚类
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(features)

# 按聚类结果分配类别
```

### 5. 如何选择类别数？

**建议**：
- 数据量 < 10,000: 2-3类
- 数据量 10,000-50,000: 3-5类
- 数据量 > 50,000: 5-10类

**原则**：
- 每个类别至少1000张图片
- 类别间应有明显视觉差异
- 不要太多类别（增加训练难度）

---

## 进阶技巧

### 1. 多条件生成

除了类别，还可以加入其他条件：

```python
# 例如：类别 + 颜色
class MultiCondNetG(nn.Module):
    def __init__(self):
        self.class_emb = nn.Embedding(num_classes, 50)
        self.color_emb = nn.Embedding(num_colors, 50)
        # ...
    
    def forward(self, z, class_label, color_label):
        class_feat = self.class_emb(class_label)
        color_feat = self.color_emb(color_label)
        cond = torch.cat([class_feat, color_feat], dim=1)
        # ...
```

### 2. 文本条件生成

使用文本描述作为条件：

```python
# 使用预训练的文本编码器
from transformers import CLIPTextModel, CLIPTokenizer

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# 将文本转为条件向量
def encode_text(text):
    tokens = tokenizer(text, return_tensors="pt")
    text_features = text_encoder(**tokens).last_hidden_state
    return text_features.mean(dim=1)
```

### 3. 渐进式训练

从低分辨率开始，逐步提高：

```python
# 32x32 -> 64x64 -> 96x96
# 每个阶段训练50 epochs
```

---

## 实验清单

完成以下步骤以完成整个实验：

- [ ] 1. 准备带标签的数据集
- [ ] 2. 训练标准GAN (200 epochs)
- [ ] 3. 训练条件GAN (200 epochs)
- [ ] 4. 生成GAN样本
- [ ] 5. 生成cGAN样本（包括类别控制）
- [ ] 6. 运行对比分析脚本
- [ ] 7. 对比训练曲线
- [ ] 8. 对比生成质量
- [ ] 9. 分析优缺点
- [ ] 10. 撰写对比报告

---

## 参考资源

- **原始论文**: Mirza & Osindero (2014). "Conditional Generative Adversarial Nets"
- **PyTorch教程**: https://pytorch.org/tutorials/
- **GAN Zoo**: https://github.com/hindupuravinash/the-gan-zoo

祝实验顺利！🎉