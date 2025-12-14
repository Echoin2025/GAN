# coding:utf8
"""
Conditional GAN模型定义
支持基于类别标签的条件生成
"""

import torch
from torch import nn


class ConditionalNetG(nn.Module):
    """
    条件生成器
    输入: 噪声z + 类别标签
    输出: 生成的图像
    """
    
    def __init__(self, opt):
        super(ConditionalNetG, self).__init__()
        self.nz = opt.nz  # 噪声维度
        self.num_classes = opt.num_classes  # 类别数
        self.label_emb_dim = opt.label_emb_dim  # 标签嵌入维度
        ngf = opt.ngf  # 生成器feature map数
        
        # 标签嵌入层：将类别标签映射到高维向量
        self.label_embedding = nn.Embedding(self.num_classes, self.label_emb_dim)
        
        # 输入是噪声z和标签嵌入的拼接
        input_dim = self.nz + self.label_emb_dim
        
        self.main = nn.Sequential(
            # 输入: (nz + label_emb_dim) x 1 x 1
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输出: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输出: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输出: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 输出: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
            # 输出: 3 x 96 x 96
        )
    
    def forward(self, noise, labels):
        """
        Args:
            noise: (batch_size, nz, 1, 1) 噪声向量
            labels: (batch_size,) 类别标签
        Returns:
            生成的图像 (batch_size, 3, 96, 96)
        """
        # 获取标签嵌入 (batch_size, label_emb_dim)
        label_emb = self.label_embedding(labels)
        # 重塑为 (batch_size, label_emb_dim, 1, 1)
        label_emb = label_emb.view(label_emb.size(0), label_emb.size(1), 1, 1)
        
        # 拼接噪声和标签嵌入
        gen_input = torch.cat([noise, label_emb], dim=1)
        
        # 生成图像
        return self.main(gen_input)


class ConditionalNetD(nn.Module):
    """
    条件判别器
    输入: 图像 + 类别标签
    输出: 真假概率
    """
    
    def __init__(self, opt):
        super(ConditionalNetD, self).__init__()
        self.num_classes = opt.num_classes
        self.image_size = opt.image_size
        ndf = opt.ndf
        
        # 标签嵌入层
        # 将标签映射到与图像相同大小的特征图
        self.label_embedding = nn.Embedding(self.num_classes, self.image_size * self.image_size)
        
        # 输入是图像(3通道) + 标签特征图(1通道) = 4通道
        self.main = nn.Sequential(
            # 输入: 4 x 96 x 96
            nn.Conv2d(4, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (ndf) x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: 1 x 1 x 1
        )
    
    def forward(self, img, labels):
        """
        Args:
            img: (batch_size, 3, 96, 96) 输入图像
            labels: (batch_size,) 类别标签
        Returns:
            判别结果 (batch_size,)
        """
        # 获取标签嵌入 (batch_size, image_size*image_size)
        label_emb = self.label_embedding(labels)
        # 重塑为 (batch_size, 1, image_size, image_size)
        label_emb = label_emb.view(label_emb.size(0), 1, self.image_size, self.image_size)
        
        # 拼接图像和标签特征图
        d_input = torch.cat([img, label_emb], dim=1)
        
        # 判别
        output = self.main(d_input)
        return output.view(-1)


class SimpleConditionalNetG(nn.Module):
    """
    简化版条件生成器（使用投影方式）
    更轻量，训练更快
    """
    
    def __init__(self, opt):
        super(SimpleConditionalNetG, self).__init__()
        self.nz = opt.nz
        self.num_classes = opt.num_classes
        ngf = opt.ngf
        
        # 简单的标签嵌入（投影到噪声维度）
        self.label_proj = nn.Embedding(self.num_classes, self.nz)
        
        self.main = nn.Sequential(
            # 输入是调制后的噪声
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        """使用标签调制噪声"""
        # 获取标签投影
        label_proj = self.label_proj(labels).view(noise.size(0), -1, 1, 1)
        # 调制噪声（element-wise乘法）
        modulated_noise = noise * (1 + label_proj)
        return self.main(modulated_noise)


class SimpleConditionalNetD(nn.Module):
    """
    简化版条件判别器（使用投影判别器）
    """
    
    def __init__(self, opt):
        super(SimpleConditionalNetD, self).__init__()
        self.num_classes = opt.num_classes
        ndf = opt.ndf
        
        # 主干网络提取特征
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 无条件判别头
        self.unconditional_head = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        
        # 标签嵌入（投影）
        self.label_proj = nn.Embedding(self.num_classes, ndf * 8)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img, labels):
        """投影判别器：D(x) + y^T * phi(x)"""
        # 提取特征 (batch_size, ndf*8, 4, 4)
        features = self.feature_extractor(img)
        
        # 无条件判别 (batch_size, 1, 1, 1)
        unconditional_out = self.unconditional_head(features)
        
        # 全局平均池化特征 (batch_size, ndf*8)
        pooled_features = torch.mean(features, dim=[2, 3])
        
        # 标签投影 (batch_size, ndf*8)
        label_proj = self.label_proj(labels)
        
        # 内积 (batch_size, 1)
        conditional_out = torch.sum(pooled_features * label_proj, dim=1, keepdim=True)
        
        # 组合
        output = unconditional_out.view(-1) + conditional_out.view(-1)
        
        return self.sigmoid(output)


# 辅助函数：生成one-hot标签
def to_one_hot(labels, num_classes):
    """
    将类别标签转换为one-hot编码
    Args:
        labels: (batch_size,) 类别标签
        num_classes: 总类别数
    Returns:
        one_hot: (batch_size, num_classes)
    """
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot
