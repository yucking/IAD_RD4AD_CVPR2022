import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score , roc_curve
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import os

def cal_anomaly_map(ft_list, fs_list, out_size=224, amap_mode='mul'):# 特征差异图计算
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size]) # 点乘情况下 所有值初始化为 1;
    else:
        anomaly_map = np.zeros([out_size, out_size])# 加法情况下 所有值初始化为 0；
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i] # shape=[B,C,H,W] 
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(ft, fs) # shape=[B,H,W] 
        a_map = torch.unsqueeze(a_map, dim=1)   # shape=[B,1,H,W] 
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True) # 上采样到统一分辨率
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy() # 转为 NumPy 图（2D）
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    '''将原图 img 与异常热图 anomaly_map 叠加，形成彩色高亮图，用于可视化异常区域。'''
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255 # 转为 float32 并归一化至 0~1: 图像亮的区域 + 异常高响应的区域 → 更亮
    cam = cam / np.max(cam) # 重新归一化回 [0, 1] : 防止叠加后数值 >1，导致图像溢出（如出现白块）并 重新拉伸到最大值为 1
    return np.uint8(255 * cam) # 恢复为可显示的 uint8 图像，得到最终可显示的 RGB 图

def min_max_norm(image):
    '''对任意图像进行最小-最大归一化，使其值分布在 [0, 1]'''
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray): # 输入必须是 uint8 类型（0~255）
    '''将灰度图转换为彩色热图（红-黄-蓝）用于可视化异常响应'''
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



def evaluation(encoder, bn, decoder, dataloader,device,_class_=None):
    bn.eval()
    decoder.eval()
    gt_list_px = []  # 所有像素的 Ground Truth
    pr_list_px = []  # 所有像素的预测 anomaly score
    gt_list_sp = []  # 每张图的 ground truth 标签（0 或 1）
    pr_list_sp = []  # 每张图的 anomaly score（使用 max pooling）
    aupro_list = []  # 每张图的 PRO 区域准确性得分
    with torch.no_grad():
        for img, gt, label, _ in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0 # 正常
            # 计算 AUPRO（区域级指标，只有异常样本计算）
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        #ano_score = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
        #vis_data = {}
        #vis_data['Anomaly Score'] = ano_score
        #vis_data['Ground Truth'] = np.array(gt_list_sp)
        # print(type(vis_data))
        # np.save('vis.npy',vis_data)
        #with open('{}_vis.pkl'.format(_class_), 'wb') as f:
        #    pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)

def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '../mvtec/' + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device,_class_)
    print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)
    return auroc_px

def visualization(_class_, test_id=0):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './mvtec/' + _class_
    ckpt_dir = f'./checkpoints/{test_id}'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckp_path = f'{ckpt_dir}/wres50_{_class_}.pth'

    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)
    for k in list(ckp['bn'].keys()):
        if 'memory' in k:
            del ckp['bn'][k]
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    pr_list_sp = []
    gt_list_sp = []
    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map * 255)

            img_np = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img_np = np.uint8(min_max_norm(img_np) * 255)

            output_dir = f'./results_all/{test_id}/{_class_}'
            os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(f'{output_dir}/{count}_org.png', img_np)
            ano_map_vis = show_cam_on_image(img_np, ano_map)
            cv2.imwrite(f'{output_dir}/{count}_ad.png', ano_map_vis)

            gt_img = gt.cpu().numpy().astype(int)[0][0] * 255
            cv2.imwrite(f'{output_dir}/{count}_gt.png', gt_img.astype(np.uint8))

            pr_list_sp.append(np.max(anomaly_map))
            gt_list_sp.append(label.item())

            count += 1

    # Score histogram
    plt.figure(figsize=(8, 4))
    plt.hist([np.array(pr_list_sp)[np.array(gt_list_sp) == 0],
              np.array(pr_list_sp)[np.array(gt_list_sp) == 1]],
             label=['Normal', 'Anomaly'], bins=20, alpha=0.7, color=['blue', 'red'])
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'Score Distribution: {_class_}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_hist.png')
    plt.close()

    # ROC Curve
    if len(set(gt_list_sp)) > 1:
        fpr, tpr, thresholds = roc_curve(gt_list_sp, pr_list_sp)
        auc_score = roc_auc_score(gt_list_sp, pr_list_sp)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'AUROC = {auc_score:.3f}', color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {_class_}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve.png')
        plt.close()
    else:
        print(f"⚠️ ROC curve not plotted: only one class present in gt for {_class_}.")




def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            #if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            #print(inputs[-1].shape)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig('ad.png')
            #plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            #count += 1
            #if count>40:
            #    return 0
                #assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        #print(type(vis_data))
        #np.save('vis.npy',vis_data)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"], dtype=float)
    #df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        #df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])], ignore_index=True)


    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def detection(encoder, bn, decoder, dataloader,device,_class_):
    #_, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)


            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1


        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean