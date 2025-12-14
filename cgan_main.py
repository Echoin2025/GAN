# coding:utf8
import os
import torch as t
import torchvision as tv
import tqdm
import numpy as np
from cgan_model import ConditionalNetG, ConditionalNetD
from torchnet.meter import AverageValueMeter


class Config(object):
    data_path = 'data/'
    num_workers = 4
    image_size = 96
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    beta1 = 0.5
    gpu = True
    nz = 100  # 噪声维度
    ngf = 64
    ndf = 64
    
    # cGAN特有参数
    num_classes = 4  # 类别数（根据数据集调整）
    label_emb_dim = 50  # 标签嵌入维度
    use_simple_model = False  # 是否使用简化版模型
    
    save_path = 'cgan_imgs/'
    log_path = 'cgan_logs/'
    
    vis = True
    env = 'cGAN'
    plot_every = 20
    
    debug_file = '/tmp/debugcgan'
    d_every = 1
    g_every = 5
    save_every = 10
    eval_every = 5
    
    netd_path = None
    netg_path = None
    
    # 生成参数
    gen_img = 'cgan_result.png'
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1
    
    # 评估参数
    eval_batch_size = 50
    num_eval_samples = 5000


opt = Config()


def train(**kwargs):
    """训练条件GAN"""
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device = t.device('cuda') if opt.gpu else t.device('cpu')
    
    # 创建保存目录
    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs('cgan_checkpoints', exist_ok=True)
    
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)
    
    # 数据加载
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    
    # 检测实际类别数
    actual_num_classes = len(dataset.classes)
    if actual_num_classes != opt.num_classes:
        print(f"警告: 配置的类别数({opt.num_classes})与数据集类别数({actual_num_classes})不匹配")
        print(f"自动调整为 {actual_num_classes} 类")
        opt.num_classes = actual_num_classes
    
    print(f"数据集信息:")
    print(f"  图片总数: {len(dataset)}")
    print(f"  类别数: {opt.num_classes}")
    print(f"  类别名称: {dataset.classes}")
    
    dataloader = t.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        drop_last=True
    )
    
    # 初始化网络
    if opt.use_simple_model:
        from cgan_model import SimpleConditionalNetG, SimpleConditionalNetD
        netg = SimpleConditionalNetG(opt)
        netd = SimpleConditionalNetD(opt)
        print("使用简化版cGAN模型")
    else:
        netg = ConditionalNetG(opt)
        netd = ConditionalNetD(opt)
        print("使用标准cGAN模型")
    
    # 加载预训练模型（如果有）
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        # 修复：添加 weights_only=True 参数
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location, weights_only=True))
        print(f"加载判别器: {opt.netd_path}")
    if opt.netg_path:
        # 修复：添加 weights_only=True 参数
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location, weights_only=True))
        print(f"加载生成器: {opt.netg_path}")
    
    netd.to(device)
    netg.to(device)
    
    # 优化器
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)
    
    # 标签
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    
    # 固定的噪声和标签用于可视化
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    # 为每个类别生成样本（如果batch_size >= num_classes）
    if opt.batch_size >= opt.num_classes:
        fix_labels = t.arange(opt.num_classes).repeat(opt.batch_size // opt.num_classes + 1)[:opt.batch_size]
    else:
        fix_labels = t.randint(0, opt.num_classes, (opt.batch_size,))
    fix_labels = fix_labels.to(device)
    
    # 用于训练的噪声
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    
    # 损失记录
    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()
    
    # 日志文件
    log_file = open(os.path.join(opt.log_path, 'training_log.txt'), 'w')
    log_file.write('Epoch,Batch,D_Loss,G_Loss,D_Real,D_Fake\n')
    
    print("\n开始训练条件GAN...")
    print("=" * 70)
    
    for epoch in range(opt.max_epoch):
        print(f"\nEpoch {epoch + 1}/{opt.max_epoch}")
        
        for ii, (img, label) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)
            real_labels = label.to(device)
            
            # ============= 训练判别器 =============
            if ii % opt.d_every == 0:
                optimizer_d.zero_grad()
                
                # 真实图片
                output = netd(real_img, real_labels)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()
                d_real_mean = output.mean().item()
                
                # 生成假图片
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                # 随机采样类别标签
                fake_labels_for_g = t.randint(0, opt.num_classes, (opt.batch_size,)).to(device)
                fake_img = netg(noises, fake_labels_for_g).detach()
                
                # 判别假图片
                output = netd(fake_img, fake_labels_for_g)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                d_fake_mean = output.mean().item()
                
                optimizer_d.step()
                
                error_d = error_d_real + error_d_fake
                errord_meter.add(error_d.item())
            
            # ============= 训练生成器 =============
            if ii % opt.g_every == 0:
                optimizer_g.zero_grad()
                
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_labels_for_g = t.randint(0, opt.num_classes, (opt.batch_size,)).to(device)
                fake_img = netg(noises, fake_labels_for_g)
                
                output = netd(fake_img, fake_labels_for_g)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                
                errorg_meter.add(error_g.item())
            
            # 记录日志
            if ii % opt.plot_every == 0:
                log_file.write(f'{epoch},{ii},{error_d.item():.4f},{error_g.item():.4f},'
                             f'{d_real_mean:.4f},{d_fake_mean:.4f}\n')
                log_file.flush()
            
            # 可视化
            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                if os.path.exists(opt.debug_file):
                    import ipdb; ipdb.set_trace()
                
                with t.no_grad():
                    fix_fake_imgs = netg(fix_noises, fix_labels)
                
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])
        
        # 保存模型和图片
        if (epoch + 1) % opt.save_every == 0:
            with t.no_grad():
                fix_fake_imgs = netg(fix_noises, fix_labels)
            
            tv.utils.save_image(
                fix_fake_imgs.data[:64],
                f'{opt.save_path}/epoch_{epoch + 1}.png',
                normalize=True,
                value_range=(-1, 1),
                nrow=8
            )
            
            t.save(netd.state_dict(), f'cgan_checkpoints/netd_{epoch + 1}.pth')
            t.save(netg.state_dict(), f'cgan_checkpoints/netg_{epoch + 1}.pth')
            print(f"模型已保存: epoch {epoch + 1}")
        
        # 重置meters
        errord_meter.reset()
        errorg_meter.reset()
        
        print(f"Epoch {epoch + 1} 完成 - D_loss: {error_d.item():.4f}, G_loss: {error_g.item():.4f}")
    
    log_file.close()
    print("\n训练完成！")


@t.no_grad()
def generate(**kwargs):
    """
    条件生成：为每个类别生成图像
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device = t.device('cuda') if opt.gpu else t.device('cpu')
    
    # 加载模型
    if opt.use_simple_model:
        from cgan_model import SimpleConditionalNetG
        netg = SimpleConditionalNetG(opt).eval()
    else:
        netg = ConditionalNetG(opt).eval()
    
    # 修复：添加 weights_only=True 参数
    netg.load_state_dict(t.load(opt.netg_path, map_location=lambda storage, loc: storage, weights_only=True))
    netg.to(device)
    
    print(f"为 {opt.num_classes} 个类别生成图像...")
    
    # 为每个类别生成多张图片
    samples_per_class = opt.gen_num // opt.num_classes
    
    all_imgs = []
    for class_id in range(opt.num_classes):
        print(f"生成类别 {class_id}...")
        
        # 生成该类别的图片
        noises = t.randn(samples_per_class, opt.nz, 1, 1).to(device)
        labels = t.full((samples_per_class,), class_id, dtype=t.long).to(device)
        
        fake_imgs = netg(noises, labels)
        all_imgs.append(fake_imgs)
    
    # 合并所有图片
    all_imgs = t.cat(all_imgs, dim=0)
    
    # 保存
    tv.utils.save_image(
        all_imgs[:opt.gen_num],
        opt.gen_img,
        normalize=True,
        value_range=(-1, 1),
        nrow=opt.num_classes  # 每行显示一个类别
    )
    
    print(f"已生成 {opt.gen_num} 张图片，保存至 {opt.gen_img}")
    print(f"图片按类别排列，每行 {samples_per_class} 张")


@t.no_grad()
def generate_with_specific_labels(**kwargs):
    """
    根据指定的标签序列生成图像
    例如：labels=[0,0,0,1,1,1,2,2,2] 会生成3张类别0、3张类别1、3张类别2
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device = t.device('cuda') if opt.gpu else t.device('cpu')
    
    # 加载模型
    if opt.use_simple_model:
        from cgan_model import SimpleConditionalNetG
        netg = SimpleConditionalNetG(opt).eval()
    else:
        netg = ConditionalNetG(opt).eval()
    
    # 修复：添加 weights_only=True 参数
    netg.load_state_dict(t.load(opt.netg_path, map_location=lambda storage, loc: storage, weights_only=True))
    netg.to(device)
    
    # 创建标签序列：每个类别生成相同数量
    samples_per_class = 8
    labels_list = []
    for class_id in range(opt.num_classes):
        labels_list.extend([class_id] * samples_per_class)
    
    labels = t.tensor(labels_list, dtype=t.long).to(device)
    noises = t.randn(len(labels), opt.nz, 1, 1).to(device)
    
    # 生成
    fake_imgs = netg(noises, labels)
    
    # 保存
    tv.utils.save_image(
        fake_imgs,
        'cgan_by_class.png',
        normalize=True,
        value_range=(-1, 1),
        nrow=samples_per_class
    )
    
    print(f"已生成类别控制的图像，保存至 cgan_by_class.png")
    print(f"每行8张，每个类别一行")


if __name__ == '__main__':
    import fire
    fire.Fire()
