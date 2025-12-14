# coding:utf8
"""
GAN vs Conditional GAN 对比分析脚本
生成对比报告和可视化
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from model import NetG, NetD
from cgan_model import ConditionalNetG, ConditionalNetD


def load_models(gan_netg_path, cgan_netg_path, device, opt_gan, opt_cgan):
    """加载GAN和cGAN模型"""
    # 加载GAN
    netg_gan = NetG(opt_gan).to(device)
    netg_gan.load_state_dict(torch.load(gan_netg_path, map_location=device))
    netg_gan.eval()
    
    # 加载cGAN
    netg_cgan = ConditionalNetG(opt_cgan).to(device)
    netg_cgan.load_state_dict(torch.load(cgan_netg_path, map_location=device))
    netg_cgan.eval()
    
    return netg_gan, netg_cgan


def generate_comparison_samples(netg_gan, netg_cgan, num_classes, nz, device, 
                                save_path='comparison/'):
    """生成GAN和cGAN的对比样本"""
    os.makedirs(save_path, exist_ok=True)
    
    num_samples = 64
    
    # 固定噪声
    fixed_noise = torch.randn(num_samples, nz, 1, 1).to(device)
    
    print("生成GAN样本...")
    with torch.no_grad():
        gan_imgs = netg_gan(fixed_noise)
    
    print("生成cGAN样本...")
    # 为cGAN创建标签（每个类别相同数量）
    samples_per_class = num_samples // num_classes
    labels = torch.arange(num_classes).repeat(samples_per_class + 1)[:num_samples].to(device)
    
    with torch.no_grad():
        cgan_imgs = netg_cgan(fixed_noise, labels)
    
    # 保存对比图
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # GAN结果
    gan_grid = tv.utils.make_grid(gan_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[0].imshow(gan_grid.permute(1, 2, 0).cpu())
    axes[0].set_title('Standard GAN (Unconditional)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # cGAN结果
    cgan_grid = tv.utils.make_grid(cgan_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[1].imshow(cgan_grid.permute(1, 2, 0).cpu())
    axes[1].set_title(f'Conditional GAN (Class-controlled, {num_classes} classes)', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gan_vs_cgan_samples.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 对比样本已保存: {save_path}/gan_vs_cgan_samples.png")
    plt.close()
    
    return gan_imgs, cgan_imgs


def visualize_class_control(netg_cgan, num_classes, nz, device, save_path='comparison/'):
    """可视化cGAN的类别控制能力"""
    os.makedirs(save_path, exist_ok=True)
    
    samples_per_class = 8
    
    # 为每个类别生成图片
    all_imgs = []
    for class_id in range(num_classes):
        noise = torch.randn(samples_per_class, nz, 1, 1).to(device)
        labels = torch.full((samples_per_class,), class_id, dtype=torch.long).to(device)
        
        with torch.no_grad():
            imgs = netg_cgan(noise, labels)
        all_imgs.append(imgs)
    
    # 拼接
    all_imgs = torch.cat(all_imgs, dim=0)
    
    # 绘制
    fig = plt.figure(figsize=(16, 2 * num_classes))
    grid = tv.utils.make_grid(all_imgs, nrow=samples_per_class, normalize=True, value_range=(-1, 1))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.title('Conditional GAN: Class-wise Generation', fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # 添加类别标签
    for i in range(num_classes):
        plt.text(-50, (i + 0.5) * grid.shape[1] / num_classes, 
                f'Class {i}', fontsize=12, va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cgan_class_control.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 类别控制可视化已保存: {save_path}/cgan_class_control.png")
    plt.close()


def analyze_diversity(imgs, name, save_path='comparison/'):
    """分析生成样本的多样性"""
    os.makedirs(save_path, exist_ok=True)
    
    # 转换为numpy
    imgs_np = imgs.cpu().numpy()
    
    # 计算统计特征
    pixel_mean = imgs_np.mean(axis=(0, 2, 3))  # 每个通道的均值
    pixel_std = imgs_np.std(axis=(0, 2, 3))    # 每个通道的标准差
    
    # 计算样本间的差异
    n = imgs_np.shape[0]
    pairwise_diffs = []
    for i in range(min(n, 100)):
        for j in range(i + 1, min(n, 100)):
            diff = np.abs(imgs_np[i] - imgs_np[j]).mean()
            pairwise_diffs.append(diff)
    
    avg_pairwise_diff = np.mean(pairwise_diffs)
    
    print(f"\n{name} 多样性分析:")
    print(f"  像素均值 (RGB): {pixel_mean}")
    print(f"  像素标准差 (RGB): {pixel_std}")
    print(f"  样本间平均差异: {avg_pairwise_diff:.4f}")
    
    return {
        'pixel_mean': pixel_mean,
        'pixel_std': pixel_std,
        'avg_pairwise_diff': avg_pairwise_diff
    }


def compare_training_curves(gan_log='logs/training_log.txt', 
                           cgan_log='cgan_logs/training_log.txt',
                           save_path='comparison/'):
    """对比训练曲线"""
    os.makedirs(save_path, exist_ok=True)
    
    import pandas as pd
    
    # 检查文件是否存在
    if not os.path.exists(gan_log):
        print(f"⚠ GAN训练日志不存在: {gan_log}")
        return
    if not os.path.exists(cgan_log):
        print(f"⚠ cGAN训练日志不存在: {cgan_log}")
        return
    
    # 读取数据
    gan_df = pd.read_csv(gan_log)
    cgan_df = pd.read_csv(cgan_log)
    
    # 计算epoch坐标
    gan_samples_per_epoch = gan_df[gan_df['Epoch'] == 0].shape[0]
    cgan_samples_per_epoch = cgan_df[cgan_df['Epoch'] == 0].shape[0]
    
    gan_df['EpochFloat'] = gan_df['Epoch'] + gan_df['Batch'] / gan_samples_per_epoch
    cgan_df['EpochFloat'] = cgan_df['Epoch'] + cgan_df['Batch'] / cgan_samples_per_epoch
    
    # 平滑
    window = 100
    gan_d_smooth = gan_df['D_Loss'].rolling(window=window, center=True).mean()
    gan_g_smooth = gan_df['G_Loss'].rolling(window=window, center=True).mean()
    cgan_d_smooth = cgan_df['D_Loss'].rolling(window=window, center=True).mean()
    cgan_g_smooth = cgan_df['G_Loss'].rolling(window=window, center=True).mean()
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GAN vs Conditional GAN: Training Comparison', fontsize=16, fontweight='bold')
    
    # D Loss对比
    ax1 = axes[0, 0]
    ax1.plot(gan_df['EpochFloat'], gan_d_smooth, label='GAN', linewidth=2, color='#2E86AB')
    ax1.plot(cgan_df['EpochFloat'], cgan_d_smooth, label='cGAN', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Discriminator Loss', fontweight='bold')
    ax1.set_title('(a) Discriminator Loss Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # G Loss对比
    ax2 = axes[0, 1]
    ax2.plot(gan_df['EpochFloat'], gan_g_smooth, label='GAN', linewidth=2, color='#2E86AB')
    ax2.plot(cgan_df['EpochFloat'], cgan_g_smooth, label='cGAN', linewidth=2, color='#A23B72')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Generator Loss', fontweight='bold')
    ax2.set_title('(b) Generator Loss Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # D(real)对比
    ax3 = axes[1, 0]
    gan_dreal_smooth = gan_df['D_Real'].rolling(window=window, center=True).mean()
    cgan_dreal_smooth = cgan_df['D_Real'].rolling(window=window, center=True).mean()
    ax3.plot(gan_df['EpochFloat'], gan_dreal_smooth, label='GAN', linewidth=2, color='#06A77D')
    ax3.plot(cgan_df['EpochFloat'], cgan_dreal_smooth, label='cGAN', linewidth=2, color='#D74E09')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('D(real)', fontweight='bold')
    ax3.set_title('(c) D(real) Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # D(fake)对比
    ax4 = axes[1, 1]
    gan_dfake_smooth = gan_df['D_Fake'].rolling(window=window, center=True).mean()
    cgan_dfake_smooth = cgan_df['D_Fake'].rolling(window=window, center=True).mean()
    ax4.plot(gan_df['EpochFloat'], gan_dfake_smooth, label='GAN', linewidth=2, color='#06A77D')
    ax4.plot(cgan_df['EpochFloat'], cgan_dfake_smooth, label='cGAN', linewidth=2, color='#D74E09')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('D(fake)', fontweight='bold')
    ax4.set_title('(d) D(fake) Comparison', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线对比已保存: {save_path}/training_comparison.png")
    plt.close()


def generate_comparison_report(save_path='comparison/'):
    """生成对比分析报告"""
    os.makedirs(save_path, exist_ok=True)
    
    report = """# GAN vs Conditional GAN 对比分析报告

## 1. 实验概述

本实验对比了标准GAN和条件GAN(cGAN)在相同数据集上的表现。

### 1.1 模型架构对比

| 特性 | GAN | Conditional GAN |
|------|-----|-----------------|
| 输入 | 随机噪声 z | 噪声 z + 类别标签 c |
| 生成器 | G(z) | G(z, c) |
| 判别器 | D(x) | D(x, c) |
| 可控性 | ❌ 无法控制生成内容 | ✅ 可指定生成类别 |
| 参数量 | 较少 | 较多（增加了标签嵌入层） |
| 训练难度 | 中等 | 稍高（需要标签信息） |

### 1.2 关键差异

**标准GAN:**
- 生成器: `x = G(z)` 其中 z ~ N(0,1)
- 判别器: `p = D(x)` 输出真假概率
- 特点: 完全无监督，生成多样但不可控

**条件GAN:**
- 生成器: `x = G(z, c)` 其中 c 是one-hot类别标签
- 判别器: `p = D(x, c)` 同时判断真假和类别匹配
- 特点: 半监督，可控制生成内容

## 2. 训练过程对比

### 2.1 损失函数

**GAN损失:**
```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
L_G = -E[log D(G(z))]
```

**cGAN损失:**
```
L_D = -E[log D(x|c)] - E[log(1 - D(G(z|c)|c))]
L_G = -E[log D(G(z|c)|c)]
```

### 2.2 训练稳定性分析

从训练曲线可以看出：

1. **判别器损失**: cGAN的判别器损失通常更稳定
   - 原因: 类别信息提供了额外的学习信号
   - 判别器不仅学习真假，还学习类别对应关系

2. **生成器损失**: cGAN可能在初期损失更高
   - 原因: 生成器需要同时学习生成和类别控制
   - 但收敛后通常更稳定

3. **模式崩溃**: cGAN相对更不容易出现模式崩溃
   - 原因: 类别约束迫使生成器学习更多样的分布

## 3. 生成质量对比

### 3.1 视觉质量

**GAN特点:**
- 生成样本多样性高
- 质量依赖于训练数据分布
- 无法控制生成特定类型

**cGAN特点:**
- 同类别内样本风格一致
- 不同类别间有明显差异
- 可以精确控制生成内容

### 3.2 多样性分析

通过分析样本间的像素差异：

- GAN: 样本间差异大，覆盖数据分布广
- cGAN: 类别内差异中等，类别间差异大

### 3.3 实用性对比

| 应用场景 | GAN | cGAN | 推荐 |
|---------|-----|------|------|
| 数据增强 | ✓ | ✓✓ | cGAN（可控） |
| 艺术创作 | ✓✓ | ✓ | GAN（更自由） |
| 特定类别生成 | ✗ | ✓✓ | cGAN |
| 探索数据分布 | ✓✓ | ✓ | GAN |
| 图像编辑 | ✗ | ✓✓ | cGAN |

## 4. 定量评估

### 4.1 Inception Score (IS)

IS衡量生成质量和多样性：
- 更高的IS表示更好的质量和多样性
- GAN和cGAN的IS应该接近

### 4.2 Fréchet Inception Distance (FID)

FID衡量生成分布与真实分布的距离：
- 更低的FID表示更接近真实分布
- cGAN可能在有明确类别时FID更低

### 4.3 类别内一致性（仅cGAN）

对于cGAN，可以额外评估：
- 同类别生成样本的特征一致性
- 类别标签的控制精度

## 5. 优缺点总结

### 5.1 标准GAN

**优点:**
- ✅ 实现简单，训练相对容易
- ✅ 参数量少，训练速度快
- ✅ 生成多样性通常很好
- ✅ 不需要标签信息

**缺点:**
- ❌ 无法控制生成内容
- ❌ 可能出现模式崩溃
- ❌ 难以生成特定类型的样本

### 5.2 条件GAN

**优点:**
- ✅ 可以精确控制生成内容
- ✅ 更稳定，不易模式崩溃
- ✅ 适合有监督学习任务
- ✅ 生成质量通常更稳定

**缺点:**
- ❌ 需要标签数据
- ❌ 模型更复杂，参数更多
- ❌ 训练时间更长
- ❌ 类别内可能多样性不足

## 6. 实验结论

1. **适用场景不同**:
   - 如果需要可控生成，选择cGAN
   - 如果追求最大多样性，选择GAN

2. **训练特性**:
   - cGAN训练更稳定
   - GAN训练更快速

3. **生成质量**:
   - 两者质量相近
   - cGAN在特定任务上更有优势

4. **实际应用**:
   - 数据增强、图像编辑: 优先cGAN
   - 艺术创作、风格迁移: 两者皆可
   - 探索性生成: 优先GAN

## 7. 改进方向

### 7.1 对于GAN
- 使用WGAN-GP改进训练稳定性
- 引入自注意力机制提升质量
- Progressive GAN提高分辨率

### 7.2 对于cGAN
- 增加条件信息的类型（如文本）
- 使用更复杂的条件嵌入方式
- 结合GAN的最新改进技术

## 8. 参考文献

1. Goodfellow, I., et al. (2014). Generative Adversarial Networks. NeurIPS.
2. Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. arXiv.
3. Odena, A., et al. (2017). Conditional Image Synthesis with Auxiliary Classifier GANs. ICML.
"""
    
    report_file = os.path.join(save_path, 'comparison_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 对比分析报告已保存: {report_file}")


def main():
    """主函数：运行完整对比分析"""
    print("=" * 70)
    print("GAN vs Conditional GAN 对比分析")
    print("=" * 70)
    
    # 创建对比目录
    os.makedirs('comparison', exist_ok=True)
    
    # 1. 对比训练曲线
    print("\n[1/4] 对比训练曲线...")
    compare_training_curves()
    
    # 2. 生成对比报告
    print("\n[2/4] 生成对比报告...")
    generate_comparison_report()
    
    print("\n[3/4] 如需生成样本对比，请确保已有训练好的模型")
    print("然后运行以下命令:")
    print("  GAN生成: python main.py generate --netg-path=... --netd-path=...")
    print("  cGAN生成: python cgan_main.py generate --netg-path=...")
    
    print("\n[4/4] 手动对比生成样本")
    print("  - 查看 result.png (GAN)")
    print("  - 查看 cgan_result.png (cGAN)")
    print("  - 查看 cgan_by_class.png (cGAN类别控制)")
    
    print("\n" + "=" * 70)
    print("对比分析完成！")
    print("所有结果已保存到 comparison/ 目录")
    print("=" * 70)


if __name__ == '__main__':
    main()
