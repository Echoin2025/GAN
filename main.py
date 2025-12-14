# coding:utf8
import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
import numpy as np
from model import NetG, NetD
from torchnet.meter import AverageValueMeter
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d


class Config(object):
    data_path = 'data/'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = True  # 是否使用GPU
    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    save_path = 'imgs/'  # 生成图片保存路径
    log_path = 'logs/'  # 日志保存路径

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = '/tmp/debuggan'  # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10  # 每10个epoch保存一次模型
    eval_every = 5  # 每5个epoch评估一次
    netd_path = None  # 预训练模型
    netg_path = None

    # 只测试不训练
    gen_img = 'result.png'
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差
    
    # 评估相关
    eval_batch_size = 50  # 评估时的batch size
    num_eval_samples = 5000  # 用于FID计算的样本数


opt = Config()


class InceptionV3(t.nn.Module):
    """用于计算IS和FID的Inception V3模型"""
    def __init__(self):
        super().__init__()
        inception = tv.models.inception_v3(pretrained=True)
        self.blocks = t.nn.ModuleList([
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            t.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            t.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            t.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 将图像从[-1,1]转换到[0,1]
        x = (x + 1) / 2
        # Resize到299x299
        x = t.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # 标准化
        x = (x - 0.5) / 0.5
        
        for block in self.blocks:
            x = block(x)
        
        return x.squeeze(-1).squeeze(-1)


def calculate_inception_score(imgs, inception_model, device, splits=10):
    """计算Inception Score"""
    N = len(imgs)
    
    # 获取预测
    preds = []
    batch_size = 32
    
    inception_model.eval()
    with t.no_grad():
        for i in range(0, N, batch_size):
            batch = imgs[i:i+batch_size].to(device)
            pred = inception_model(batch)
            pred = t.nn.functional.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # 计算IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py)))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算Frechet距离"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # 计算sqrt(sigma1 * sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 数值稳定性
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(real_imgs, fake_imgs, inception_model, device):
    """计算FID分数"""
    inception_model.eval()
    
    def get_activations(imgs):
        acts = []
        batch_size = 32
        with t.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size].to(device)
                act = inception_model(batch)
                acts.append(act.cpu().numpy())
        return np.concatenate(acts, axis=0)
    
    real_acts = get_activations(real_imgs)
    fake_acts = get_activations(fake_imgs)
    
    mu1, sigma1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    mu2, sigma2 = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)
    
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid


def evaluate_model(netg, dataloader, device, opt):
    """评估模型生成质量"""
    print("\n开始评估模型...")
    
    # 加载Inception模型
    inception_model = InceptionV3().to(device)
    
    # 生成假图片
    netg.eval()
    fake_imgs = []
    num_batches = opt.num_eval_samples // opt.eval_batch_size
    
    with t.no_grad():
        for _ in range(num_batches):
            noise = t.randn(opt.eval_batch_size, opt.nz, 1, 1).to(device)
            fake_img = netg(noise)
            fake_imgs.append(fake_img)
    
    fake_imgs = t.cat(fake_imgs, dim=0)
    
    # 获取真实图片
    real_imgs = []
    for i, (img, _) in enumerate(dataloader):
        real_imgs.append(img)
        if len(real_imgs) * img.size(0) >= opt.num_eval_samples:
            break
    real_imgs = t.cat(real_imgs, dim=0)[:opt.num_eval_samples]
    
    # 计算IS
    is_mean, is_std = calculate_inception_score(fake_imgs, inception_model, device)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # 计算FID
    fid_score = calculate_fid(real_imgs, fake_imgs, inception_model, device)
    print(f"FID Score: {fid_score:.4f}")
    
    netg.train()
    
    return {
        'is_mean': is_mean,
        'is_std': is_std,
        'fid': fid_score
    }


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    
    # 创建保存目录
    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True)

    # 网络
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # 真图片label为1，假图片label为0
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()
    
    # 日志文件
    log_file = open(os.path.join(opt.log_path, 'training_log.txt'), 'w')
    log_file.write('Epoch,Batch,D_Loss,G_Loss,D_Real,D_Fake\n')

    epochs = range(opt.max_epoch)
    
    for epoch in epochs:
        print(f"\nEpoch {epoch+1}/{opt.max_epoch}")
        
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                
                # 真图片
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()
                d_real_mean = output.mean().item()

                # 假图片
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                d_fake_mean = output.mean().item()
                
                optimizer_d.step()

                error_d = error_d_fake + error_d_real
                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())
            
            # 记录日志
            if ii % opt.plot_every == 0:
                log_file.write(f'{epoch},{ii},{error_d.item():.4f},{error_g.item():.4f},'
                             f'{d_real_mean:.4f},{d_fake_mean:.4f}\n')
                log_file.flush()

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                    
                with t.no_grad():
                    fix_fake_imgs = netg(fix_noises)
                    
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])

        # 每个epoch结束后的操作
        if (epoch + 1) % opt.save_every == 0:
            # 保存模型、图片
            with t.no_grad():
                fix_fake_imgs = netg(fix_noises)
            tv.utils.save_image(fix_fake_imgs.data[:64], 
                              f'{opt.save_path}/epoch_{epoch+1}.png', 
                              normalize=True, value_range=(-1, 1))
            t.save(netd.state_dict(), f'checkpoints/netd_{epoch+1}.pth')
            t.save(netg.state_dict(), f'checkpoints/netg_{epoch+1}.pth')
            print(f"模型已保存: epoch {epoch+1}")
        
        # 评估模型
        if (epoch + 1) % opt.eval_every == 0:
            metrics = evaluate_model(netg, dataloader, device, opt)
            if opt.vis:
                vis.plot('IS', metrics['is_mean'])
                vis.plot('FID', metrics['fid'])
            
            # 保存评估结果
            with open(os.path.join(opt.log_path, 'eval_metrics.txt'), 'a') as f:
                f.write(f"Epoch {epoch+1}: IS={metrics['is_mean']:.4f}±{metrics['is_std']:.4f}, "
                       f"FID={metrics['fid']:.4f}\n")
        
        # 重置meters
        errord_meter.reset()
        errorg_meter.reset()
        
        print(f"Epoch {epoch+1} 完成 - D_loss: {error_d.item():.4f}, G_loss: {error_g.item():.4f}")
    
    log_file.close()
    print("\n训练完成！")


@t.no_grad()
def generate(**kwargs):
    """随机生成动漫头像，并根据netd的分数选择较好的"""
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device = t.device('cuda') if opt.gpu else t.device('cpu')

    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    
    # 保存图片
    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, value_range=(-1, 1))
    print(f"已生成 {opt.gen_num} 张图片，保存至 {opt.gen_img}")


if __name__ == '__main__':
    import fire
    fire.Fire()