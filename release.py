import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from utils.tools import same_seeds, get_project_path
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import argparse
import math
import os
import numpy as np
import math

class GradCAMHelper:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self.acts = None
        self.grads = None
        self._fh = layer.register_forward_hook(self._forward_hook)
        self._bhs = []

    def _forward_hook(self, module, inp, out):
        if isinstance(out, tuple):
            self.acts = out[0]
            handle = out[0].register_hook(self._backward_hook)
        else:
            self.acts = out
            handle = out.register_hook(self._backward_hook)
        self._bhs.append(handle)

    def _backward_hook(self, grad):
        self.grads = grad

    def remove(self):
        try:
            self._fh.remove()
        except Exception:
            pass
        for handle in self._bhs:
             try:
                 handle.remove()
             except Exception:
                 pass
        self._bhs = []
        self.acts = None
        self.grads = None

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        initial_mode = self.model.training
        self.model.eval()

        self.acts = None
        self.grads = None
        out = self.model(x)

        if self.acts is None:
            print(f"Warning: Activations not captured for layer {self.layer}. Check model structure/hook registration. Returning zero CAM.")
            self.model.train(initial_mode)
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        if isinstance(out, tuple):
            out_for_grad = out[0]
        else:
            out_for_grad = out
        target = out_for_grad.max(1)[1]
        one_hot = torch.zeros_like(out_for_grad)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)

        try:
            out_for_grad.backward(gradient=one_hot, retain_graph=False)
        except RuntimeError as e:
            print(f"Error during backward pass for Grad-CAM: {e}. Returning zero CAM.")
            self.model.train(initial_mode)
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)


        if self.grads is None:
            print(f"Warning: Gradients not captured for layer {self.layer}. Backward hook might have failed. Returning zero CAM.")
            self.model.train(initial_mode)
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        n = cam.shape[0]
        cam_flat = cam.view(n, -1)
        maxv = cam_flat.max(dim=1, keepdim=True)[0].view(n, 1, 1, 1)
        minv = cam_flat.min(dim=1, keepdim=True)[0].view(n, 1, 1, 1)
        cam_normalized = (cam - minv) / (maxv - minv + 1e-12)

        self.model.train(initial_mode)

        return cam_normalized


def pick_last_conv(model: nn.Module) -> nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def _gaussian_kernel(size: int, sigma: float, device: torch.device):
    size = int(size) if size % 2 == 1 else int(size) + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g1d = g1d / (g1d.sum() + 1e-12)
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / (g2d.sum() + 1e-12)
    return g2d

try:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    TORCHMETRICS_AVAILABLE = True
    print("✅ torchmetrics found. PSNR and SSIM metrics are enabled.")
except Exception:
    try:
        from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as psnr
        from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim
        TORCHMETRICS_AVAILABLE = True
        print("✅ torchmetrics found (image.functional). PSNR and SSIM metrics are enabled.")
    except Exception:
        try:
            from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
            TORCHMETRICS_AVAILABLE = True
            print("✅ torchmetrics found (image). PSNR and SSIM metrics are enabled.")

            def psnr(preds, target, data_range=1.0):
                metric = PeakSignalNoiseRatio(data_range=data_range).to(preds.device)
                return metric(preds, target)

            def ssim(preds, target, data_range=1.0):
                metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(preds.device)
                return metric(preds, target)
        except Exception:
            print("⚠️ Warning: torchmetrics not available. PSNR/SSIM metrics will be skipped.")
            psnr, ssim = None, None
            TORCHMETRICS_AVAILABLE = False

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPSClass
    LPIPS_AVAILABLE = True
    LPIPS_METRIC = None
    def get_lpips_metric(device):
        global LPIPS_METRIC
        if LPIPS_METRIC is None or (getattr(LPIPS_METRIC, 'device', None) != device):
            LPIPS_METRIC = LPIPSClass(net_type='alex').to(device)
        return LPIPS_METRIC
    print("✅ LPIPS metric enabled (torchmetrics.image.lpip).")
except Exception:
    print("⚠️ Warning: LPIPS metric not available (torchmetrics.image.lpip). LPIPS will be skipped.")
    LPIPS_AVAILABLE = False
    LPIPS_METRIC = None

def srgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    thr = 0.04045
    linear = torch.where(img <= thr, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[:, 0, :, :], linear[:, 1, :, :], linear[:, 2, :, :]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta3, t.pow(1/3), (t / (3 * delta**2)) + (4.0/29.0))
    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b2 = 200.0 * (fy - fz)
    return torch.stack([L, a, b2], dim=1)

def delta_e76(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    lab1 = srgb_to_lab(img1)
    lab2 = srgb_to_lab(img2)
    d = lab1 - lab2
    de = torch.sqrt(torch.clamp(d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2, min=0))
    n = img1.shape[0]
    return de.view(n, -1).mean(dim=1).mean()

from utils.defenses import (
    BitDepthReduce, JPEGDefense, RandResizePad,
    PreprocessThenModel, NIPSR3Wrapper,
    FeatureDenoisingPlaceholder, ComDefendPlaceholder
)

from attack_method import MI_FGSM_SMER, I_FGSM_SMER, VMI_FGSM_SMER, SI_MI_FGSM_SMER

try:
    import matplotlib
    from matplotlib import cm as mpl_cm
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(description='SMER-NDI and Advanced Attacks')
    parser.add_argument('--dataset', type=str, default='imagenet_compatible', help='imagenet_compatible')
    parser.add_argument('--batch-size', type=int, default=25, help='Batch size')
    parser.add_argument('--image-size', type=int, default=224, help='image size of the dataloader')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--attack_method', type=str, default='MI_FGSM_SMER')
    parser.add_argument('--image-dir', type=str, required=True, help="Path to image directory")
    parser.add_argument('--image-info', type=str, required=True, help="Path to image info CSV")
    parser.add_argument('--image-ext', type=str, default='JPEG', help='Image file extension (e.g., JPEG, PNG)')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu_id')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max images to attack (0 means all)')
    parser.add_argument('--ens-models', type=str, default='vit_t,deit_t,resnet18,inc_v3',
                        help='Surrogate models (comma separated keys from model zoo)')

    parser.add_argument('--eps', type=float, default=16.0)
    parser.add_argument('--alpha', type=float, default=1.6)
    parser.add_argument('--iters', type=int, default=10, help="Number of outer iterations")
    parser.add_argument('--momentum', type=float, default=1.0, help='Momentum value')
    parser.add_argument('--beta', type=float, default=10, help="Step size for inner iterations")

    parser.add_argument('--nesterov', action='store_true', help='Enable Nesterov momentum')
    parser.add_argument('--use-di', action='store_true', help='Enable Diverse Inputs (DI)')
    parser.add_argument('--di-prob', type=float, default=0.7, help='Probability of applying DI')
    parser.add_argument('--use-ti', action='store_true', help='Enable Translation-Invariant (TI)')

    parser.add_argument('--vmi-beta', type=float, default=1.5, help='Beta for VMI variance tuning')
    parser.add_argument('--si-scales', type=int, default=5, help='Number of scales for SI')

    parser.add_argument("--defense", type=str, default="none",
                        choices=["none", "rnp", "nips_r3", "bitred", "jpeg", "fd", "comdefend"],
                        help="选择在受害模型前应用的输入防御。")
    parser.add_argument("--rnp_in", type=int, default=299, help="R&P 最小尺寸（原始输入尺寸）。")
    parser.add_argument("--rnp_max", type=int, default=331, help="R&P 最大pad尺寸。")
    parser.add_argument("--nips_passes", type=int, default=3, help="NIPS-r3 随机化次数。")
    parser.add_argument("--bitred_bits", type=int, default=5, help="Bit-Red 位深（1~8）。")
    parser.add_argument("--jpeg_quality", type=int, default=75, help="JPEG 质量（1~100）。")

    parser.add_argument('--save-adv', action='store_true', help='启用保存对抗样本到磁盘')
    parser.add_argument('--save-adv-dir', type=str, default=None, help='对抗样本输出目录（默认 outputs/adv_<attack_method>）')
    parser.add_argument('--save-adv-ext', type=str, default=None, help='保存文件扩展名（默认沿用 --image-ext）')

    parser.add_argument('--save-heatmap', action='store_true', help='保存对抗扰动的热力图')
    parser.add_argument('--save-heatmap-dir', type=str, default=None, help='热力图输出目录（默认 outputs/heatmap_<attack_method>）')
    parser.add_argument('--save-heatmap-ext', type=str, default=None, help='热力图文件扩展名（默认 png）')
    parser.add_argument('--heatmap-mode', type=str, default='l2', choices=['l2', 'linf', 'absmean'], help='像素级通道聚合方式')
    parser.add_argument('--heatmap-norm', type=str, default='eps', choices=['eps', 'per_image', 'none'], help='归一化策略')
    parser.add_argument('--heatmap-cmap', type=str, default='jet', choices=['jet','turbo','viridis','plasma','magma','inferno','gray'], help='伪彩色映射方案')
    parser.add_argument('--heatmap-overlay', action='store_true', help='将热力图与图像叠加保存')
    parser.add_argument('--heatmap-alpha', type=float, default=0.6, help='热力图叠加透明度')
    parser.add_argument('--heatmap-overlay-base', type=str, default='adv', choices=['adv','orig'], help='叠加底图：对抗图或原图')
    parser.add_argument('--heatmap-text', action='store_true', help='在图上标注数值')
    parser.add_argument('--heatmap-text-metric', type=str, default='ssim', choices=['ssim','psnr','lpips','l2','linf','l1'], help='文本标注指标')
    parser.add_argument('--heatmap-smooth', type=str, default='none', choices=['none','avg','max','gauss'], help='热力图平滑方式')
    parser.add_argument('--heatmap-smooth-k', type=int, default=11, help='平滑核大小，奇数')
    parser.add_argument('--heatmap-smooth-sigma', type=float, default=2.0, help='高斯平滑 sigma')
    parser.add_argument('--heatmap-thresh', type=float, default=0.0, help='可视化阈值，低于则置零')
    parser.add_argument('--heatmap-source', type=str, default='perturb', choices=['perturb','gradcam'], help='热力图来源')
    parser.add_argument('--heatmap-gradcam-model', type=str, default=None, help='用于 Grad-CAM 的模型键（如 resnet18）')

    parser.add_argument('--disable-rl-reweighing', action='store_true',
                        help='Disable the RL based reweighing mechanism (use fixed weights)')
    parser.add_argument('--disable-mb', action='store_true',
                        help='Disable the Stochastic Mini-Batch perturbing (use ensemble average gradient instead)')

    return parser.parse_args()


def attach_defense(models: dict, args):
    def infer_input_size(m) -> int:
        siz = getattr(m, 'default_cfg', {}).get('input_size', (None, None, 224))[-1]
        if hasattr(m, 'patch_embed') and hasattr(m.patch_embed, 'img_size'):
            try:
                sz = m.patch_embed.img_size
                siz = int(sz[0] if isinstance(sz, (tuple, list)) else sz)
            except Exception: pass
        return siz

    if args.defense == "none":
        for name, mdl in models.items():
            mdl.eval()
        return models

    wrapped = {}
    defense_map = {
        "bitred": BitDepthReduce(bits=args.bitred_bits),
        "jpeg": JPEGDefense(quality=args.jpeg_quality),
        "fd": FeatureDenoisingPlaceholder,
        "comdefend": ComDefendPlaceholder
    }

    print(f"🛡️ [DEFENSE] enabled: {args.defense}")
    for name, mdl in models.items():
        original_device = next(mdl.parameters()).device
        mdl_size = infer_input_size(mdl)

        if args.defense in defense_map:
            if args.defense in ["fd", "comdefend"]:
                 preprocessor = defense_map[args.defense]()
            else:
                 preprocessor = defense_map[args.defense]
            wrapped[name] = PreprocessThenModel(preprocessor, mdl)
        elif args.defense == "rnp":
            rnd = RandResizePad(in_size=args.rnp_in, max_size=args.rnp_max, out_size=mdl_size)
            wrapped[name] = PreprocessThenModel(rnd, mdl)
        elif args.defense == "nips_r3":
            rnd = RandResizePad(in_size=args.rnp_in, max_size=args.rnp_max, out_size=mdl_size)
            wrapped[name] = NIPSR3Wrapper(randomizer=rnd, model=mdl, passes=args.nips_passes)
        else:
            wrapped[name] = mdl

        if name in wrapped:
             wrapped[name] = wrapped[name].to(original_device).eval()
             print(f"  - {name}: Applied {type(wrapped[name]).__name__}")

    for name, mdl in models.items():
        if name not in wrapped:
            models[name].eval()
            wrapped[name] = models[name]


    return wrapped


def setup_attack(args, all_models):
    attack_methods = {
        'MI_FGSM_SMER': MI_FGSM_SMER,
        'I_FGSM_SMER': I_FGSM_SMER,
        'VMI_FGSM_SMER': VMI_FGSM_SMER,
        'SI_MI_FGSM_SMER': SI_MI_FGSM_SMER
    }
    if args.attack_method not in attack_methods:
        raise ValueError(f"Unknown attack method: {args.attack_method}. Available: {list(attack_methods.keys())}")

    attack_func = attack_methods[args.attack_method]

    ens_model_names = [s.strip() for s in args.ens_models.split(',') if s.strip()]
    unknown = [n for n in ens_model_names if n not in all_models]
    if unknown:
        raise ValueError(f'Unknown model names in --ens-models: {unknown}. Available: {list(all_models.keys())}')

    print(f'🎯 Ensemble (surrogate) models: {ens_model_names}')
    ens_models_list = []
    for name in ens_model_names:
        if name in all_models:
            model = all_models[name]
            model.eval()
            ens_models_list.append(model)

    return attack_func, ens_models_list


def run_evaluation_loop(args, dataloader, attack_func, ens_models, eval_models, metrix, device):

    results = {
        "l_inf_scores": [], "l2_scores": [], "psnr_scores": [], "ssim_scores": [],
        "l1_scores": [], "deltae_scores": [], "lpips_scores": []
    }

    total_to_process = len(dataloader.dataset)
    if args.max_samples and args.max_samples > 0 and args.max_samples < total_to_process:
        total_to_process = args.max_samples

    processed_samples = 0
    pbar = tqdm(total=total_to_process, desc="Attacking batches")
    grad_cam_helper = None

    for data, label, _ in dataloader:
        if processed_samples >= total_to_process:
            break

        n = min(label.size(0), total_to_process - processed_samples)
        data, label = data[:n].to(device), label[:n].to(device)

        adv_exp = attack_func(ens_models, data, label, args=args)

        if getattr(args, 'save_adv', False):
             out_dir = args.save_adv_dir or os.path.join(args.root_path, 'outputs', f"adv_{args.attack_method}")
             os.makedirs(out_dir, exist_ok=True)
             save_ext = (args.save_adv_ext or args.image_ext).lower()
             start_idx = processed_samples
             names = getattr(dataloader.dataset, 'image_name', None)
             for i in range(n):
                 if names is not None and start_idx + i < len(names): fname = names[start_idx + i]
                 else: fname = f"sample_{start_idx + i:06d}"
                 out_path = os.path.join(out_dir, f"{fname}.{save_ext}")
                 save_image(adv_exp[i].detach().cpu().clamp(0, 1), out_path)

        perturbation = adv_exp - data

        if getattr(args, 'save_heatmap', False):
            source = getattr(args, 'heatmap_source', 'perturb')
            hmap = None
            denom = 1.0

            if source == 'gradcam':
                mkey = getattr(args, 'heatmap_gradcam_model', None)
                cand = None
                all_available_models = {**eval_models, **{f"ens_{i}": m for i, m in enumerate(ens_models)}}

                if mkey and mkey in all_available_models: cand = all_available_models[mkey]
                elif 'resnet18' in all_available_models: cand = all_available_models['resnet18']
                else:
                    for k, m in all_available_models.items():
                        if pick_last_conv(m) is not None:
                            cand = m; print(f"Using model '{k}' for Grad-CAM as fallback."); break

                if cand is not None:
                    layer = pick_last_conv(cand)
                    if layer is not None:
                        if grad_cam_helper is None or grad_cam_helper.model != cand or grad_cam_helper.layer != layer:
                            if grad_cam_helper: grad_cam_helper.remove()
                            grad_cam_helper = GradCAMHelper(cand, layer)

                        base_for_cam = adv_exp if getattr(args, 'heatmap_overlay_base', 'adv') == 'adv' else data
                        try:
                            hmap = grad_cam_helper.compute(base_for_cam)
                        except Exception as e:
                            print(f"Error computing Grad-CAM: {e}. Falling back.")
                            hmap = None
                            if grad_cam_helper: grad_cam_helper.remove(); grad_cam_helper = None
                    else: print(f"Warning: No Conv2d layer found in Grad-CAM model. Falling back.")
                else: print("Warning: No suitable model for Grad-CAM. Falling back.")

            if hmap is None:
                mode = getattr(args, 'heatmap_mode', 'l2')
                if mode == 'linf': hmap = perturbation.abs().max(dim=1, keepdim=True)[0]; denom = args.eps / 255.0
                elif mode == 'absmean': hmap = perturbation.abs().mean(dim=1, keepdim=True); denom = args.eps / 255.0
                else: hmap = torch.norm(perturbation, p=2, dim=1, keepdim=True); denom = (args.eps / 255.0) * math.sqrt(3.0)

                norm_mode = getattr(args, 'heatmap_norm', 'eps')
                if norm_mode == 'eps': hmap = hmap / (denom + 1e-12) if denom > 1e-12 else hmap.fill_(0)
                elif norm_mode == 'per_image':
                    maxvals = hmap.view(n, -1).max(dim=1)[0].view(n, 1, 1, 1); hmap = hmap / (maxvals + 1e-12)
                hmap = hmap.clamp(0, 1)

            smooth = getattr(args, 'heatmap_smooth', 'none')
            k = int(getattr(args, 'heatmap_smooth_k', 11))
            sigma = float(getattr(args, 'heatmap_smooth_sigma', 2.0))
            thresh = float(getattr(args, 'heatmap_thresh', 0.0))
            if smooth != 'none':
                pad = k // 2
                if smooth == 'avg': hmap = F.avg_pool2d(hmap, kernel_size=k, stride=1, padding=pad)
                elif smooth == 'max': hmap = F.max_pool2d(hmap, kernel_size=k, stride=1, padding=pad)
                elif smooth == 'gauss':
                    gk = _gaussian_kernel(k, sigma, device=hmap.device).view(1, 1, k, k); hmap = F.conv2d(hmap, gk, stride=1, padding=pad)
            if thresh > 0.0: hmap = torch.where(hmap >= thresh, hmap, torch.zeros_like(hmap))
            hmap = hmap.clamp(0, 1)

            cmap_name = getattr(args, 'heatmap_cmap', 'jet')
            if MATPLOTLIB_AVAILABLE and cmap_name != 'gray':
                with torch.no_grad():
                     h_np = hmap.squeeze(1).cpu().numpy()
                h_rgb_np = matplotlib.colormaps.get_cmap(cmap_name)(h_np)[..., :3]
                h_rgb = torch.from_numpy(h_rgb_np).permute(0, 3, 1, 2).float()
            else: h_rgb = hmap.repeat(1, 3, 1, 1)

            overlay_on = bool(getattr(args, 'heatmap_overlay', False))
            alpha = float(getattr(args, 'heatmap_alpha', 0.6))
            base_choice = getattr(args, 'heatmap_overlay_base', 'adv')
            base_img_cpu = (adv_exp if base_choice == 'adv' else data).detach().cpu()
            to_save = h_rgb.cpu()
            if overlay_on: to_save = torch.clamp(alpha * to_save + (1.0 - alpha) * base_img_cpu, 0, 1)

            hm_dir = args.save_heatmap_dir or os.path.join(args.root_path, 'outputs', f"heatmap_{args.attack_method}")
            os.makedirs(hm_dir, exist_ok=True)
            hm_ext = (args.save_heatmap_ext or 'png').lower()
            start_idx = processed_samples
            names = getattr(dataloader.dataset, 'image_name', None)

            for i in range(n):
                if names is not None and start_idx + i < len(names): fname = names[start_idx + i]
                else: fname = f"sample_{start_idx + i:06d}"
                out_path = os.path.join(hm_dir, f"{fname}.{hm_ext}")

                metric_text = None
                if getattr(args, 'heatmap_text', False):
                    mname = getattr(args, 'heatmap_text_metric', 'ssim')
                    try:
                        adv_i, data_i = adv_exp[i:i+1], data[i:i+1]
                        if mname == 'ssim' and TORCHMETRICS_AVAILABLE: val = ssim(adv_i, data_i, data_range=1.0)
                        elif mname == 'psnr' and TORCHMETRICS_AVAILABLE: val = psnr(adv_i, data_i, data_range=1.0)
                        elif mname == 'lpips' and LPIPS_AVAILABLE:
                            lpips_metric = get_lpips_metric(device); val = lpips_metric(adv_i * 2 - 1, data_i * 2 - 1)
                        elif mname in ['l2', 'linf', 'l1']:
                            diff = (adv_i - data_i).view(1, -1)
                            if mname == 'l2': val = torch.norm(diff, p=2, dim=1)
                            elif mname == 'linf': val = torch.norm(diff, p=float('inf'), dim=1)
                            else: val = diff.abs().mean()
                        else: val = None

                        if val is not None: metric_text = f"{(val.item() if hasattr(val,'item') else float(val)):.4f}"
                    except Exception: metric_text = None

                if metric_text and PIL_AVAILABLE:
                    img_pil = to_pil_image(to_save[i])
                    draw = ImageDraw.Draw(img_pil)
                    try: font = ImageFont.truetype("arial.ttf", 15)
                    except IOError: font = ImageFont.load_default()
                    draw.text((8 + 1, 8 + 1), metric_text, fill=(0, 0, 0), font=font)
                    draw.text((8, 8), metric_text, fill=(255, 255, 255), font=font)
                    img_pil.save(out_path)
                else:
                    save_image(to_save[i], out_path)

        if TORCHMETRICS_AVAILABLE:
            try: ps = psnr(adv_exp, data, data_range=1.0); results["psnr_scores"].append(ps.item() if hasattr(ps, 'item') else float(ps))
            except Exception: pass
            try: ss = ssim(adv_exp, data, data_range=1.0); results["ssim_scores"].append(ss.item() if hasattr(ss, 'item') else float(ss))
            except Exception: pass
        if LPIPS_AVAILABLE:
            try:
                lpips_metric = get_lpips_metric(device)
                lpips_val = lpips_metric(adv_exp * 2.0 - 1.0, data * 2.0 - 1.0)
                results["lpips_scores"].append(lpips_val.item() if hasattr(lpips_val, 'item') else float(lpips_val))
            except Exception: pass
        try: results["deltae_scores"].append(delta_e76(adv_exp, data).item())
        except Exception: pass
        results["l_inf_scores"].append(torch.norm(perturbation.view(n, -1), p=float('inf'), dim=1).mean().item())
        results["l2_scores"].append(torch.norm(perturbation.view(n, -1), p=2, dim=1).mean().item())
        results["l1_scores"].append(torch.abs(perturbation).view(n, -1).mean(dim=1).mean().item())


        for model_name, model in eval_models.items():
            model.eval()
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)

                if isinstance(r_clean, tuple): r_clean = r_clean[0]
                if isinstance(r_adv, tuple): r_adv = r_adv[0]

                if model_name in ['inc_v3_ens3', 'inc_v3_ens4', 'incres_v2_ens']:
                    if r_clean.shape[1] > 1: r_clean = r_clean[:, 1:]
                    if r_adv.shape[1] > 1: r_adv = r_adv[:, 1:]

                pred_clean = r_clean.max(1)[1]
                correct_clean = (pred_clean == label).sum().item()
                pred_adv = r_adv.max(1)[1]
                correct_adv = (pred_adv == label).sum().item()

            metrix[model_name].update(correct_clean, correct_adv, n)

        processed_samples += n
        pbar.update(n)

    pbar.close()
    if grad_cam_helper:
         grad_cam_helper.remove()
    return results


def print_results(eval_models, metrix, imperceptibility_results):
    print("\n" + "="*80)
    print("📊 ATTACK EVALUATION RESULTS")
    print("="*80)

    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    print('-' * 73)
    total_samples_evaluated = 0
    total_successful_attacks_sum = 0
    total_initially_correct_sum = 0

    valid_model_names = [name for name, m in metrix.items() if hasattr(m, 'total_num') and m.total_num > 0]

    for model_name in valid_model_names:
        if not all(hasattr(metrix[model_name], attr) for attr in ['total_num', 'total_correct_clean', 'total_correct_adv']):
            print(f"| Skipping model {model_name.ljust(10)} due to missing metrics attributes.")
            continue

        total_num = metrix[model_name].total_num
        clean_acc_val = metrix[model_name].total_correct_clean / total_num
        adv_acc_val = metrix[model_name].total_correct_adv / total_num
        initially_correct = metrix[model_name].total_correct_clean
        successful_attacks = initially_correct - metrix[model_name].total_correct_adv
        asr_val = (successful_attacks / initially_correct) * 100 if initially_correct > 0 else 0.0

        print(f"|\t{model_name.ljust(10)}\t"
              f"|\t{round(clean_acc_val * 100, 2):<13.2f}\t"
              f"|\t{round(adv_acc_val * 100, 2):<13.2f}\t"
              f"|\t{round(asr_val, 2):<8.2f}\t|")

        total_successful_attacks_sum += successful_attacks
        total_initially_correct_sum += initially_correct

    print('-' * 73)

    if total_initially_correct_sum > 0:
        overall_avg_asr = (total_successful_attacks_sum / total_initially_correct_sum) * 100
        print(f"|\tOverall Avg ASR\t|\t-\t\t|\t-\t\t|\t{overall_avg_asr:<8.2f}\t|")
        print('-' * 73)


    print('\n--- Imperceptibility Metrics ---')
    print('-' * 65)
    print('|\tMetric\t\t\t\t|\tAverage Value\t|')
    print('-' * 65)

    def print_metric(name, scores_key, scale=1.0, unit=""):
        scores = imperceptibility_results.get(scores_key, [])
        if scores:
            avg_val = np.mean(scores) * scale
            print(f"|\t{name.ljust(24)}\t|\t{avg_val:<13.4f}\t| {unit}")
        else:
            print(f"|\t{name.ljust(24)}\t|\t{'N/A'.ljust(13)}\t| {unit}")

    print_metric("L-infinity (0-255)", "l_inf_scores", 255.0)
    print_metric("L2 Norm", "l2_scores")
    print_metric("L1 Mean Abs (0-255)", "l1_scores", 255.0)
    print_metric("PSNR", "psnr_scores", unit="dB")
    print_metric("SSIM", "ssim_scores")
    print_metric("LPIPS (AlexNet)", "lpips_scores")
    print_metric("Delta E (CIE76)", "deltae_scores")

    print('-' * 65)
    print("="*80)

def main():
    args = get_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    setattr(args, 'config_path', os.path.join(root_path, 'configs', 'checkpoint.yaml'))
    same_seeds(42)
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataloader = get_dataset(args)
    print("Loading models...")
    models, metrix = get_models(args, device=device)

    print("Setting up attack...")
    attack_func, ens_models = setup_attack(args, models)

    print("Attaching defenses (if any)...")
    eval_models = attach_defense(models, args)

    print("Starting evaluation loop...")
    imperceptibility_results = run_evaluation_loop(args, dataloader, attack_func, ens_models, eval_models, metrix, device)

    print("Evaluation finished. Printing results...")
    print_results(eval_models, metrix, imperceptibility_results)

if __name__ == '__main__':
    main()
