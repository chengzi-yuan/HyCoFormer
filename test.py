import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import hycoformer_s, hycoformer_b

MODEL_BUILDERS = {
    'hycoformer-s': hycoformer_s,
    'hycoformer-b': hycoformer_b,
}


def build_model(name):
    key = name.lower()
    if key not in MODEL_BUILDERS:
        raise ValueError('Unsupported model: ' + name)
    return MODEL_BUILDERS[key]()


parser = argparse.ArgumentParser(description='HyCoFormer testing')
parser.add_argument('--model', default='hycoformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='dataloader workers')
parser.add_argument('--data_dir', default=r'D:\Czy_Project\DF-LAgent2.0RLN\data', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()
        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '%.02f_%.04f.csv' % (PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
    network = build_model(args.model).cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if not os.path.exists(saved_model_dir):
        print('==> No existing trained model!')
        sys.exit(0)

    print('==> Start testing, current model name: ' + args.model)
    network.load_state_dict(single(saved_model_dir))

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    test(test_loader, network, result_dir)
