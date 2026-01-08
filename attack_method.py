import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def diverse_input(image, prob=0.7, resize_min=0.9, resize_max=1.1):
    if torch.rand(1).item() > prob:
        return image
    img_size = image.size(-1)
    new_size = max(1, int(torch.empty(1).uniform_(resize_min, resize_max).item() * img_size))
    if new_size < 4 and image.shape[-2] < 4:
         mode = 'nearest'
    else:
         mode = 'bilinear'
    resized_image = F.interpolate(image, size=new_size, mode=mode, align_corners=False if mode=='bilinear' else None)

    if new_size < img_size:
        pad_h = img_size - new_size
        pad_w = img_size - new_size
        pad_top = torch.randint(0, pad_h + 1, (1,)).item()
        pad_bottom = pad_h - pad_top
        pad_left = torch.randint(0, pad_w + 1, (1,)).item()
        pad_right = pad_w - pad_left
        return F.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)
    elif new_size > img_size:
        crop_h = new_size - img_size
        crop_w = new_size - img_size
        crop_top = torch.randint(0, crop_h + 1, (1,)).item()
        crop_left = torch.randint(0, crop_w + 1, (1,)).item()
        return resized_image[:, :, crop_top:crop_top + img_size, crop_left:crop_left + img_size]
    else:
        return resized_image


def get_ti_kernel(kernel_size=5, sigma=3.0):
    if kernel_size % 2 == 0: kernel_size +=1
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1.) / 2.
    variance = sigma**2.
    if variance < 1e-12: variance = 1e-12
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
    if torch.sum(gaussian_kernel) < 1e-12: gaussian_kernel.fill_(1.0 / (kernel_size**2))
    else: gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    return kernel

TI_KERNEL_BASE_CPU = get_ti_kernel()

def apply_ti(grad):
    if TI_KERNEL_BASE_CPU is None: return grad
    channels = grad.shape[1]
    kernel = TI_KERNEL_BASE_CPU.repeat(channels, 1, 1, 1).to(grad.device)
    return F.conv2d(grad, kernel, padding='same', groups=channels)


def clip_by_tensor(t, t_min, t_max):
    return torch.clamp(t, min=t_min, max=t_max)

class Weight_Selection(nn.Module):
    def __init__(self, weight_len) -> None:
        super(Weight_Selection, self).__init__()
        self.weight = nn.parameter.Parameter(torch.ones(weight_len))

    def forward(self, x, index):
        return self.weight[index] * x

def get_smer_gradient(images, labels, surrogate_models, weight_selection, optimizer, args):
    m = len(surrogate_models)
    if m == 0: return torch.zeros_like(images)

    m_smer = m * 4
    beta = args.beta / 255.0
    image_min = clip_by_tensor(images - (args.eps / 255.0), 0.0, 1.0).detach()
    image_max = clip_by_tensor(images + (args.eps / 255.0), 0.0, 1.0).detach()

    x_inner = images.clone().detach().requires_grad_(False)
    grad_inner = torch.zeros_like(images)

    options = []
    num_full_cycles = m_smer // m
    remainder = m_smer % m
    for _ in range(num_full_cycles):
        options_single = list(range(m))
        np.random.shuffle(options_single)
        options.extend(options_single)
    if remainder > 0:
        options_single = list(range(m))
        np.random.shuffle(options_single)
        options.extend(options_single[:remainder])

    for j in range(m_smer):
        x_inner_grad = x_inner.clone().detach().requires_grad_(True)

        if args.use_di:
            x_for_grad = diverse_input(x_inner_grad, prob=args.di_prob)
        else:
            x_for_grad = x_inner_grad

        option_idx = options[j]
        model_wrapped = surrogate_models[option_idx]
        noise_im_inner = torch.zeros_like(x_for_grad)

        try:
            initial_model_mode = model_wrapped.training
            model_wrapped.eval()
            out_logits = model_wrapped(x_for_grad)
            model_wrapped.train(initial_model_mode)

            if isinstance(out_logits, tuple): out_logits = out_logits[0]
            if isinstance(out_logits, list):
                loss = F.cross_entropy(out_logits[0], labels) + F.cross_entropy(out_logits[1], labels)
            else:
                loss = F.cross_entropy(out_logits, labels)

            model_wrapped.zero_grad(set_to_none=True)
            grad_outputs = torch.autograd.grad(loss, x_inner_grad,
                                                grad_outputs=torch.ones_like(loss),
                                                retain_graph=False, create_graph=False)[0]
            if grad_outputs is not None:
                current_weight_grad = weight_selection.weight[option_idx]
                noise_im_inner = current_weight_grad * grad_outputs

            else:
                print(f"Warning: Grad calculation returned None for model {option_idx} at inner step {j}.")

        except Exception as e:
            print(f"Warning: Grad calculation failed for model {option_idx} at inner step {j}. Error: {e}. Grad set to zero.")
            if 'initial_model_mode' in locals(): model_wrapped.train(initial_model_mode)

        if not getattr(args, 'disable_rl_reweighing', False):
            x_inner_no_grad = x_inner.detach()
            group_logits, group_aux_logits = 0.0, 0.0
            valid_rl_models = 0
            has_aux = False

            for m_step, model_s_wrapped in enumerate(surrogate_models):
                try:
                    initial_mode_rl = model_s_wrapped.training
                    model_s_wrapped.eval()
                    output = model_s_wrapped(x_inner_no_grad)
                    model_s_wrapped.train(initial_mode_rl)

                    if isinstance(output, tuple): output = output[0]
                    current_weight_rl = weight_selection.weight[m_step]

                    if isinstance(output, list):
                        logits = current_weight_rl * output[0]
                        aux_logits = current_weight_rl * output[1]
                        if not isinstance(group_aux_logits, torch.Tensor): group_aux_logits = torch.zeros_like(aux_logits)
                        group_aux_logits += aux_logits
                        has_aux = True
                    else:
                        logits = current_weight_rl * output
                    if not isinstance(group_logits, torch.Tensor): group_logits = torch.zeros_like(logits)
                    group_logits += logits
                    valid_rl_models += 1
                except Exception as e_rl:
                    if 'initial_mode_rl' in locals(): model_s_wrapped.train(initial_mode_rl)
                    continue

            if valid_rl_models > 0 and isinstance(group_logits, torch.Tensor):
                avg_group_logits = group_logits / valid_rl_models
                loss_rl = F.cross_entropy(avg_group_logits, labels)
                if has_aux and isinstance(group_aux_logits, torch.Tensor):
                    avg_group_aux_logits = group_aux_logits / valid_rl_models
                    loss_rl += F.cross_entropy(avg_group_aux_logits, labels)

                if torch.isfinite(loss_rl):
                    optimizer.zero_grad()
                    outer_loss = -torch.log(loss_rl + 1e-12)
                    outer_loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                         weight_selection.weight.clamp_(min=0.0)

        noise_inner_detached = noise_im_inner.detach()
        if not torch.all(noise_inner_detached == 0):
            l1_norm = torch.mean(torch.abs(noise_inner_detached), dim=(1, 2, 3), keepdim=True)
            noise_normalized = noise_inner_detached / (l1_norm + 1e-12)
        else:
            noise_normalized = noise_inner_detached

        grad_inner = grad_inner + noise_normalized

        x_inner = x_inner + beta * torch.sign(grad_inner)
        x_inner = clip_by_tensor(x_inner, image_min, image_max).detach()

    return grad_inner


def get_ensemble_average_gradient(images, labels, surrogate_models, weight_selection, args):
    total_weighted_grad = torch.zeros_like(images)
    sum_weights = 0.0

    images_grad = images.clone().detach().requires_grad_(True)

    if args.use_di:
        x_for_grad = diverse_input(images_grad, prob=args.di_prob)
    else:
        x_for_grad = images_grad

    for i, model in enumerate(surrogate_models):
        current_grad = torch.zeros_like(x_for_grad)
        try:
            initial_mode = model.training
            model.eval()
            out_logits = model(x_for_grad)
            model.train(initial_mode)

            if isinstance(out_logits, tuple): out_logits = out_logits[0]

            if isinstance(out_logits, list):
                loss = F.cross_entropy(out_logits[0], labels) + F.cross_entropy(out_logits[1], labels)
            else:
                loss = F.cross_entropy(out_logits, labels)

            model.zero_grad(set_to_none=True)
            grad_outputs = torch.autograd.grad(loss, x_for_grad,
                                                grad_outputs=torch.ones_like(loss),
                                                retain_graph=False, create_graph=False)[0]

            if grad_outputs is not None:
                current_grad = grad_outputs
            else:
                print(f"Warning: Avg Grad calculation returned None for model {i}.")

        except Exception as e:
            print(f"Warning: Avg Grad calculation failed for model {i}. Error: {e}. Grad set to zero.")
            if 'initial_mode' in locals(): model.train(initial_mode)

        current_weight = weight_selection.weight[i].clamp(min=0.0).detach()
        total_weighted_grad += current_weight * current_grad
        sum_weights += current_weight.item()

    if sum_weights > 1e-12:
        avg_grad = total_weighted_grad / sum_weights
    else:
        avg_grad = torch.zeros_like(images)

    return avg_grad.detach()


def perform_rl_update(images, labels, surrogate_models, weight_selection, optimizer, args):
    if getattr(args, 'disable_rl_reweighing', False):
        return

    x_no_grad = images.detach()
    group_logits, group_aux_logits = 0.0, 0.0
    valid_rl_models = 0
    has_aux = False

    for m_step, model in enumerate(surrogate_models):
        try:
            initial_mode = model.training
            model.eval()
            output = model(x_no_grad)
            model.train(initial_mode)

            if isinstance(output, tuple): output = output[0]
            current_weight = weight_selection.weight[m_step]

            if isinstance(output, list):
                logits = current_weight * output[0]
                aux_logits = current_weight * output[1]
                if not isinstance(group_aux_logits, torch.Tensor): group_aux_logits = torch.zeros_like(aux_logits)
                group_aux_logits += aux_logits
                has_aux = True
            else:
                logits = current_weight * output
            if not isinstance(group_logits, torch.Tensor): group_logits = torch.zeros_like(logits)
            group_logits += logits
            valid_rl_models += 1
        except Exception:
            if 'initial_mode' in locals(): model.train(initial_mode)
            continue

    if valid_rl_models > 0 and isinstance(group_logits, torch.Tensor):
        avg_logits = group_logits / valid_rl_models
        loss_rl = F.cross_entropy(avg_logits, labels)
        if has_aux and isinstance(group_aux_logits, torch.Tensor):
            avg_aux_logits = group_aux_logits / valid_rl_models
            loss_rl += F.cross_entropy(avg_aux_logits, labels)

        if torch.isfinite(loss_rl):
            optimizer.zero_grad()
            outer_loss = -torch.log(loss_rl + 1e-12)
            outer_loss.backward()
            optimizer.step()
            with torch.no_grad():
                weight_selection.weight.clamp_(min=0.0)


def MI_FGSM_SMER(surrogate_models, images, labels, args):
    eps, alpha, num_iter, momentum = args.eps/255.0, args.alpha/255.0, args.iters, args.momentum
    image_min = clip_by_tensor(images - eps, 0.0, 1.0).detach()
    image_max = clip_by_tensor(images + eps, 0.0, 1.0).detach()
    m = len(surrogate_models)
    if m == 0: return images.clone().detach()

    weight_selection = Weight_Selection(m).to(images.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(), lr=2e-2, weight_decay=2e-3)
    grad = torch.zeros_like(images)
    adv_images = images.clone().detach()

    use_mb = not getattr(args, 'disable_mb', False)

    for i in range(num_iter):
        lookahead_images = adv_images + alpha * momentum * grad if args.nesterov else adv_images
        lookahead_images = lookahead_images.detach()

        if use_mb:
            noise_direction = get_smer_gradient(lookahead_images, labels, surrogate_models, weight_selection, optimizer, args)
        else:
            noise_direction = get_ensemble_average_gradient(lookahead_images, labels, surrogate_models, weight_selection, args)
            perform_rl_update(lookahead_images, labels, surrogate_models, weight_selection, optimizer, args)

        if args.use_ti:
            noise_direction = apply_ti(noise_direction)

        if not torch.all(noise_direction == 0):
             l1_norm_outer = torch.mean(torch.abs(noise_direction), dim=(1, 2, 3), keepdim=True)
             noise = noise_direction / (l1_norm_outer + 1e-12)
        else:
             noise = noise_direction

        grad = momentum * grad + noise
        adv_images = adv_images + alpha * torch.sign(grad)
        adv_images = clip_by_tensor(adv_images, image_min, image_max).detach()

    return adv_images


def I_FGSM_SMER(surrogate_models, images, labels, args):
    eps, alpha, num_iter = args.eps/255.0, args.alpha/255.0, args.iters
    image_min = clip_by_tensor(images - eps, 0.0, 1.0).detach()
    image_max = clip_by_tensor(images + eps, 0.0, 1.0).detach()
    m = len(surrogate_models)
    if m == 0: return images.clone().detach()

    weight_selection = Weight_Selection(m).to(images.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(), lr=2e-2, weight_decay=2e-3)
    adv_images = images.clone().detach()
    use_mb = not getattr(args, 'disable_mb', False)

    for i in range(num_iter):
        adv_images_no_grad = adv_images.detach()

        if use_mb:
            grad_direction = get_smer_gradient(adv_images_no_grad, labels, surrogate_models, weight_selection, optimizer, args)
        else:
            grad_direction = get_ensemble_average_gradient(adv_images_no_grad, labels, surrogate_models, weight_selection, args)
            perform_rl_update(adv_images_no_grad, labels, surrogate_models, weight_selection, optimizer, args)

        adv_images = adv_images + alpha * torch.sign(grad_direction)
        adv_images = clip_by_tensor(adv_images, image_min, image_max).detach()

    return adv_images


def VMI_FGSM_SMER(surrogate_models, images, labels, args):
    eps, alpha, num_iter, momentum, vmi_beta = args.eps/255.0, args.alpha/255.0, args.iters, args.momentum, args.vmi_beta
    image_min = clip_by_tensor(images - eps, 0.0, 1.0).detach()
    image_max = clip_by_tensor(images + eps, 0.0, 1.0).detach()
    m = len(surrogate_models)
    if m == 0: return images.clone().detach()

    weight_selection = Weight_Selection(m).to(images.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(), lr=2e-2, weight_decay=2e-3)
    grad, variance = torch.zeros_like(images), torch.zeros_like(images)
    adv_images = images.clone().detach()
    use_mb = not getattr(args, 'disable_mb', False)

    for i in range(num_iter):
        lookahead_images = adv_images + alpha * momentum * grad if args.nesterov else adv_images
        lookahead_images = lookahead_images.detach()

        if use_mb:
            current_grad_direction = get_smer_gradient(lookahead_images, labels, surrogate_models, weight_selection, optimizer, args)
        else:
            current_grad_direction = get_ensemble_average_gradient(lookahead_images, labels, surrogate_models, weight_selection, args)
            perform_rl_update(lookahead_images, labels, surrogate_models, weight_selection, optimizer, args)

        if not torch.all(current_grad_direction == 0):
            l1_norm_current = torch.mean(torch.abs(current_grad_direction), dim=(1, 2, 3), keepdim=True)
            normalized_current_grad = current_grad_direction / (l1_norm_current + 1e-12)
            variance = momentum * variance + normalized_current_grad
        else:
            variance = momentum * variance

        vmi_grad = current_grad_direction + vmi_beta * variance

        if args.use_ti:
            vmi_grad = apply_ti(vmi_grad)

        if not torch.all(vmi_grad == 0):
            l1_norm_vmi = torch.mean(torch.abs(vmi_grad), dim=(1, 2, 3), keepdim=True)
            noise = vmi_grad / (l1_norm_vmi + 1e-12)
        else:
            noise = vmi_grad

        grad = momentum * grad + noise
        adv_images = adv_images + alpha * torch.sign(grad)
        adv_images = clip_by_tensor(adv_images, image_min, image_max).detach()

    return adv_images

def SI_MI_FGSM_SMER(surrogate_models, images, labels, args):
    eps, alpha, num_iter, momentum, si_scales = args.eps/255.0, args.alpha/255.0, args.iters, args.momentum, args.si_scales
    image_min = clip_by_tensor(images - eps, 0.0, 1.0).detach()
    image_max = clip_by_tensor(images + eps, 0.0, 1.0).detach()
    m = len(surrogate_models)
    if m == 0: return images.clone().detach()

    weight_selection = Weight_Selection(m).to(images.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(), lr=2e-2, weight_decay=2e-3)
    grad = torch.zeros_like(images)
    adv_images = images.clone().detach()
    original_size = images.shape[-2:]
    use_mb = not getattr(args, 'disable_mb', False)

    for i in range(num_iter):
        lookahead_images = adv_images + alpha * momentum * grad if args.nesterov else adv_images
        lookahead_images = lookahead_images.detach()
        total_scaled_grad = torch.zeros_like(images)

        for scale_factor_pow in range(si_scales):
            scale_factor = 1 / (2**scale_factor_pow)
            if scale_factor == 1.0:
                scaled_images = lookahead_images
            else:
                new_h = max(1, int(original_size[0] * scale_factor))
                new_w = max(1, int(original_size[1] * scale_factor))
                scaled_images = F.interpolate(lookahead_images, size=(new_h, new_w), mode='bilinear', align_corners=False)

            if use_mb:
                current_scaled_grad = get_smer_gradient(scaled_images, labels, surrogate_models, weight_selection, optimizer, args)
            else:
                current_scaled_grad = get_ensemble_average_gradient(scaled_images, labels, surrogate_models, weight_selection, args)
                if scale_factor_pow == 0:
                     perform_rl_update(scaled_images, labels, surrogate_models, weight_selection, optimizer, args)

            if current_scaled_grad.shape[-2:] != original_size:
                current_scaled_grad = F.interpolate(current_scaled_grad, size=original_size, mode='bilinear', align_corners=False)
            total_scaled_grad += current_scaled_grad

        avg_grad = total_scaled_grad / si_scales

        if args.use_ti:
            avg_grad = apply_ti(avg_grad)

        if not torch.all(avg_grad == 0):
             l1_norm_si = torch.mean(torch.abs(avg_grad), dim=(1, 2, 3), keepdim=True)
             noise = avg_grad / (l1_norm_si + 1e-12)
        else:
             noise = avg_grad

        grad = momentum * grad + noise
        adv_images = adv_images + alpha * torch.sign(grad)
        adv_images = clip_by_tensor(adv_images, image_min, image_max).detach()

    return adv_images
