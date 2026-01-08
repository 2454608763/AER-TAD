import os
import timm
import yaml
import torch
from torchvision import transforms
from utils.AverageMeter import AccuracyMeter
from .tf_model_wrapper import TensorFlowModelWrapper

def get_models(args, device):
    metrix = {}
    with open(os.path.join(args.root_path, 'configs', 'checkpoint.yaml'), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    print('🌟\tBuilding models...')
    models = {}
    if args.dataset == 'imagenet_compatible':
        local_weights_dir = os.path.join(args.root_path, 'ImageNet_pretrained')

        tf_models_info = {
            'inc_v3_ens3': {
                'path': 'ens3_adv_inception_v3/ens3_adv_inception_v3.ckpt',
                'input_tensor': 'Placeholder:0',
                'output_tensor': 'InceptionV3/Logits/SpatialSqueeze:0'
            },
            'inc_v3_ens4': {
                'path': 'ens4_adv_inception_v3/ens4_adv_inception_v3.ckpt',
                'input_tensor': 'Placeholder:0',
                'output_tensor': 'InceptionV3/Logits/SpatialSqueeze:0'
            },
            'incres_v2_ens': {
                'path': 'ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt',
                'input_tensor': 'Placeholder:0',
                'output_tensor': 'InceptionResnetV2/Logits/Logits/BiasAdd:0'
            }
        }
        for key, value in yaml_data.items():
            model_to_load = None

            if key in tf_models_info:
                print(f'🧠\tLoading TensorFlow model: {key}')
                tf_info = tf_models_info[key]
                model_path = os.path.join(local_weights_dir, tf_info['path'])
                
                if os.path.exists(model_path + '.meta'):
                    try:
                        model_to_load = TensorFlowModelWrapper(
                            model_path=model_path,
                            input_tensor_name=tf_info['input_tensor'],
                            output_tensor_name=tf_info['output_tensor']
                        )
                        models[key] = model_to_load 
                        models[key].eval()
                        metrix[key] = AccuracyMeter()
                        print(f'⭐\tload {key} (TensorFlow) successfully')
                        continue
                    except Exception as e:
                        print(f'❌\tFailed to load TensorFlow model {key}: {e}')
                        continue
                else:
                    print(f'⚠️\tTensorFlow checkpoint (.meta) not found for {key} at {model_path}.meta')
                    continue
            
            local_weight_path_suffix = value.get('ckp_path')
            local_weight_path = None
            if local_weight_path_suffix:
                local_weight_path = os.path.join(local_weights_dir, local_weight_path_suffix)
            
            if key in ['inc_v3_ens3_pytorch', 'inc_v3_ens4_pytorch', 'incres_v2_ens_pytorch']:
                print(f'🔄\t[{key}] Using reliable pretrained weights (ignoring local ckp to avoid mismatch).')
                model_to_load = timm.create_model(value['full_name'], pretrained=True, num_classes=1000)
            else:
                if local_weight_path_suffix and os.path.exists(local_weight_path):
                    print(f'🔍\tLoading {key} from local weights: {local_weight_path}')
                    try:
                        model_to_load = timm.create_model(value['full_name'], pretrained=False, num_classes=1000)
                        loaded_obj = torch.load(local_weight_path, map_location='cpu', weights_only=False)
                        state_dict = loaded_obj if isinstance(loaded_obj, dict) else loaded_obj.state_dict()
                        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
                        if 'model' in state_dict: state_dict = state_dict['model']
                        model_to_load.load_state_dict(state_dict, strict=False)
                        print(f'✅\tSuccessfully loaded {key} from local weights')
                    except Exception as e:
                        print(f'❌\tFailed to load {key} from local weights: {e}')
                        model_to_load = None
                else:
                    if local_weight_path_suffix:
                        print(f'⚠️\tLocal weights not found for {key} at {local_weight_path}')
            
            if model_to_load is None:
                print(f'🔄\tFalling back to online download for {key}')
                model_to_load = timm.create_model(value['full_name'], pretrained=True, num_classes=1000)

            models[key] = model_to_load.to(device)
            models[key].eval()
            
            if 'inception' in value.get('full_name', ''):
                print(f'✨\tAdapting {key} for {args.image_size}x{args.image_size} input by adding a Resize layer.')
                models[key] = torch.nn.Sequential(
                    transforms.Resize((299, 299), antialias=True),
                    models[key]
                )
            
            if 'inc' in key or 'vit' in key or 'deit' in key or 'bit' in key or 'swin' in key:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), models[key])
            else:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), models[key])
            
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    else:
        raise NotImplementedError
    return models, metrix
