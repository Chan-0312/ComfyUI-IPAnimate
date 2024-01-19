import torch
import contextlib
import os
import math

import comfy.utils
import comfy.model_management
from comfy.clip_vision import clip_preprocess
from comfy.ldm.modules.attention import optimized_attention
import folder_paths

from torch import nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as TT

# set the models directory backward compatible
GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
MODELS_DIR = GLOBAL_MODELS_DIR if os.path.isdir(GLOBAL_MODELS_DIR) else os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
if "ipadapter" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ipadapter"] = ([MODELS_DIR], folder_paths.supported_pt_extensions)
else:
    folder_paths.folder_names_and_paths["ipadapter"][1].update(folder_paths.supported_pt_extensions)

class MLPProjModelImport(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class ImageProjModelImport(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KVImport(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatchImport(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

def image_add_noise(image, noise):
    image = image.permute([0,3,1,2])
    torch.manual_seed(0) # use a fixed random for reproducible results
    transforms = TT.Compose([
        TT.CenterCrop(min(image.shape[2], image.shape[3])),
        TT.Resize((224, 224), interpolation=TT.InterpolationMode.BICUBIC, antialias=True),
        TT.ElasticTransform(alpha=75.0, sigma=noise*3.5), # shuffle the image
        TT.RandomVerticalFlip(p=1.0), # flip the image to change the geometry even more
        TT.RandomHorizontalFlip(p=1.0),
    ])
    image = transforms(image.cpu())
    image = image.permute([0,2,3,1])
    image = image + ((0.25*(1-noise)+0.05) * torch.randn_like(image) )   # add further random noise
    return image

def zeroed_hidden_states(clip_vision, batch_size):
    image = torch.zeros([batch_size, 224, 224, 3])
    comfy.model_management.load_model_gpu(clip_vision.patcher)
    pixel_values = clip_preprocess(image.to(clip_vision.load_device))

    if clip_vision.dtype != torch.float32:
        precision_scope = torch.autocast
    else:
        precision_scope = lambda a, b: contextlib.nullcontext(a)

    with precision_scope(comfy.model_management.get_autocast_device(clip_vision.load_device), torch.float32):
        outputs = clip_vision.model(pixel_values, intermediate_output=-2)

    # we only need the penultimate hidden states
    outputs = outputs[1].to(comfy.model_management.intermediate_device())

    return outputs

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)
    
def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
def contrast_adaptive_sharpening(image, amount):
    img = F.pad(image, pad=(1, 1, 1, 1)).cpu()

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]
    
    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)
    
    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2
    
    # Computing local weight
    inv_mx = torch.reciprocal(mx)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.sqrt(amp)
    w = - amp * (amount * (1/5 - 1/8) + 1/8)
    div = torch.reciprocal(1 + 4*w)

    output = ((b + d + f + h)*w + e) * div
    output = output.clamp(0, 1)
    output = torch.nan_to_num(output)

    return (output)

class IPAdapterImport(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4, is_sdxl=False, is_plus=False, is_full=False):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full

        self.image_proj_model = self.init_proj() if not is_plus else self.init_proj_plus()
        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KVImport(ipadapter_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = ImageProjModelImport(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    def init_proj_plus(self):
        if self.is_full:
            image_proj_model = MLPProjModelImport(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=self.clip_embeddings_dim
            )
        else:
            image_proj_model = ResamplerImport(
                dim=self.cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if self.is_sdxl else 12,
                num_queries=self.clip_extra_context_tokens,
                embedding_dim=self.clip_embeddings_dim,
                output_dim=self.output_cross_attention_dim,
                ff_mult=4
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

class CrossAttentionPatchImport:
    # forward for patching
    def __init__(self, weight, ipadapter, device, dtype, number, cond, uncond, weight_type, mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.device = 'cuda' if 'cuda' in device.type else 'cpu'
        self.dtype = dtype if 'cuda' in self.device else torch.bfloat16
        self.number = number
        self.weight_type = [weight_type]
        self.masks = [mask]
        self.sigma_start = [sigma_start]
        self.sigma_end = [sigma_end]
        self.unfold_batch = [unfold_batch]

        self.k_key = str(self.number*2+1) + "_to_k_ip"
        self.v_key = str(self.number*2+1) + "_to_v_ip"
    
    def set_new_condition(self, weight, ipadapter, device, dtype, number, cond, uncond, weight_type, mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.masks.append(mask)
        self.device = 'cuda' if 'cuda' in device.type else 'cpu'
        self.dtype = dtype if 'cuda' in self.device else torch.bfloat16
        self.weight_type.append(weight_type)
        self.sigma_start.append(sigma_start)
        self.sigma_end.append(sigma_end)
        self.unfold_batch.append(unfold_batch)

    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]
        sigma = extra_options["sigmas"][0].item() if 'sigmas' in extra_options else 999999999.9

        # extra options for AnimateDiff
        ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            q = n
            k = context_attn2
            v = value_attn2
            b = q.shape[0]
            qs = q.shape[1]
            batch_prompt = b // len(cond_or_uncond)
            out = optimized_attention(q, k, v, extra_options["n_heads"])
            _, _, lh, lw = extra_options["original_shape"]
            
            for weight, cond, uncond, ipadapter, mask, weight_type, sigma_start, sigma_end, unfold_batch in zip(self.weights, self.conds, self.unconds, self.ipadapters, self.masks, self.weight_type, self.sigma_start, self.sigma_end, self.unfold_batch):
                if sigma > sigma_start or sigma < sigma_end:
                    continue

                if unfold_batch and cond.shape[0] > 1:
                    # Check AnimateDiff context window
                    if ad_params is not None and ad_params["sub_idxs"] is not None:
                        # if images length matches or exceeds full_length get sub_idx images
                        if cond.shape[0] >= ad_params["full_length"]:
                            cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                            uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
                        # otherwise, need to do more to get proper sub_idxs masks
                        else:
                            # check if images length matches full_length - if not, make it match
                            if cond.shape[0] < ad_params["full_length"]:
                                cond = torch.cat((cond, cond[-1:].repeat((ad_params["full_length"]-cond.shape[0], 1, 1))), dim=0)
                                uncond = torch.cat((uncond, uncond[-1:].repeat((ad_params["full_length"]-uncond.shape[0], 1, 1))), dim=0)
                            # if we have too many remove the excess (should not happen, but just in case)
                            if cond.shape[0] > ad_params["full_length"]:
                                cond = cond[:ad_params["full_length"]]
                                uncond = uncond[:ad_params["full_length"]]
                            cond = cond[ad_params["sub_idxs"]]
                            uncond = uncond[ad_params["sub_idxs"]]

                    # if we don't have enough reference images repeat the last one until we reach the right size
                    if cond.shape[0] < batch_prompt:
                        cond = torch.cat((cond, cond[-1:].repeat((batch_prompt-cond.shape[0], 1, 1))), dim=0)
                        uncond = torch.cat((uncond, uncond[-1:].repeat((batch_prompt-uncond.shape[0], 1, 1))), dim=0)
                    # if we have too many remove the exceeding
                    elif cond.shape[0] > batch_prompt:
                        cond = cond[:batch_prompt]
                        uncond = uncond[:batch_prompt]

                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond)
                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond)
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond)
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond)
                else:
                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond).repeat(batch_prompt, 1, 1)
                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond).repeat(batch_prompt, 1, 1)
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond).repeat(batch_prompt, 1, 1)
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond).repeat(batch_prompt, 1, 1)

                if weight_type.startswith("linear"):
                    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0) * weight
                    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0) * weight
                else:
                    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
                    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

                    if weight_type.startswith("channel"):
                        # code by Lvmin Zhang at Stanford University as also seen on Fooocus IPAdapter implementation
                        # please read licensing notes https://github.com/lllyasviel/Fooocus/blob/main/fooocus_extras/ip_adapter.py#L225
                        ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                        ip_v_offset = ip_v - ip_v_mean
                        _, _, C = ip_k.shape
                        channel_penalty = float(C) / 1280.0
                        W = weight * channel_penalty
                        ip_k = ip_k * W
                        ip_v = ip_v_offset + ip_v_mean * W

                out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])           
                if weight_type.startswith("original"):
                    out_ip = out_ip * weight

                if mask is not None:
                    # TODO: needs checking
                    mask_h = max(1, round(lh / math.sqrt(lh * lw / qs)))
                    mask_w = qs // mask_h

                    # check if using AnimateDiff and sliding context window
                    if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
                        # if mask length matches or exceeds full_length, just get sub_idx masks, resize, and continue
                        if mask.shape[0] >= ad_params["full_length"]:
                            mask_downsample = torch.Tensor(mask[ad_params["sub_idxs"]])
                            mask_downsample = F.interpolate(mask_downsample.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)
                        # otherwise, need to do more to get proper sub_idxs masks
                        else:
                            # resize to needed attention size (to save on memory)
                            mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)
                            # check if mask length matches full_length - if not, make it match
                            if mask_downsample.shape[0] < ad_params["full_length"]:
                                mask_downsample = torch.cat((mask_downsample, mask_downsample[-1:].repeat((ad_params["full_length"]-mask_downsample.shape[0], 1, 1))), dim=0)
                            # if we have too many remove the excess (should not happen, but just in case)
                            if mask_downsample.shape[0] > ad_params["full_length"]:
                                mask_downsample = mask_downsample[:ad_params["full_length"]]
                            # now, select sub_idxs masks
                            mask_downsample = mask_downsample[ad_params["sub_idxs"]]
                    # otherwise, perform usual mask interpolation
                    else:
                        mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)

                    # if we don't have enough masks repeat the last one until we reach the right size
                    if mask_downsample.shape[0] < batch_prompt:
                        mask_downsample = torch.cat((mask_downsample, mask_downsample[-1:, :, :].repeat((batch_prompt-mask_downsample.shape[0], 1, 1))), dim=0)
                    # if we have too many remove the exceeding
                    elif mask_downsample.shape[0] > batch_prompt:
                        mask_downsample = mask_downsample[:batch_prompt, :, :]
                    
                    # repeat the masks
                    mask_downsample = mask_downsample.repeat(len(cond_or_uncond), 1, 1)
                    mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1, 1).repeat(1, 1, out.shape[2])

                    out_ip = out_ip * mask_downsample

                out = out + out_ip

        return out.to(dtype=org_dtype)



class IPAdapterApplyImport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "model": ("MODEL", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "unfold_batch": ("BOOLEAN", { "default": False }),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, ipadapter, model, weight, clip_vision=None, image=None, weight_type="original", noise=None, embeds=None, attn_mask=None, start_at=0.0, end_at=1.0, unfold_batch=False):
        self.dtype = model.model.diffusion_model.dtype
        self.device = comfy.model_management.get_torch_device()
        self.weight = weight
        self.is_full = "proj.0.weight" in ipadapter["image_proj"]
        self.is_plus = self.is_full or "latents" in ipadapter["image_proj"]

        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.is_sdxl = output_cross_attention_dim == 2048
        cross_attention_dim = 1280 if self.is_plus and self.is_sdxl else output_cross_attention_dim
        clip_extra_context_tokens = 16 if self.is_plus else 4

        if embeds is not None:
            embeds = torch.unbind(embeds)
            clip_embed = embeds[0].cpu()
            clip_embed_zeroed = embeds[1].cpu()
        else:
            if image.shape[1] != image.shape[2]:
                print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

            clip_embed = clip_vision.encode_image(image)
            neg_image = image_add_noise(image, noise) if noise > 0 else None
            
            if self.is_plus:
                clip_embed = clip_embed.penultimate_hidden_states
                if noise > 0:
                    clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
                else:
                    clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])
            else:
                clip_embed = clip_embed.image_embeds
                if noise > 0:
                    clip_embed_zeroed = clip_vision.encode_image(neg_image).image_embeds
                else:
                    clip_embed_zeroed = torch.zeros_like(clip_embed)

        clip_embeddings_dim = clip_embed.shape[-1]

        self.ipadapter = IPAdapterImport(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=self.is_sdxl,
            is_plus=self.is_plus,
            is_full=self.is_full,
        )
        
        self.ipadapter.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.ipadapter.get_image_embeds(clip_embed.to(self.device, self.dtype), clip_embed_zeroed.to(self.device, self.dtype))
        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        sigma_start = model.model.model_sampling.percent_to_sigma(start_at)
        sigma_end = model.model.model_sampling.percent_to_sigma(end_at)

        patch_kwargs = {
            "number": 0,
            "weight": self.weight,
            "ipadapter": self.ipadapter,
            "device": self.device,
            "dtype": self.dtype,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "weight_type": weight_type,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "unfold_batch": unfold_batch,
        }

        if not self.is_sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
                patch_kwargs["number"] += 1

        return (work_model, )

def prep_image(image, interpolation="LANCZOS", crop_position="center", sharpening=0.0):
    _, oh, ow, _ = image.shape
    output = image.permute([0,3,1,2])

    if "pad" in crop_position:
        target_length = max(oh, ow)
        pad_l = (target_length - ow) // 2
        pad_r = (target_length - ow) - pad_l
        pad_t = (target_length - oh) // 2
        pad_b = (target_length - oh) - pad_t
        output = F.pad(output, (pad_l, pad_r, pad_t, pad_b), value=0, mode="constant")
    else:
        crop_size = min(oh, ow)
        x = (ow-crop_size) // 2
        y = (oh-crop_size) // 2
        if "top" in crop_position:
            y = 0
        elif "bottom" in crop_position:
            y = oh-crop_size
        elif "left" in crop_position:
            x = 0
        elif "right" in crop_position:
            x = ow-crop_size
        
        x2 = x+crop_size
        y2 = y+crop_size

        # crop
        output = output[:, :, y:y2, x:x2]

    # resize (apparently PIL resize is better than tourchvision interpolate)
    imgs = []
    for i in range(output.shape[0]):
        img = TT.ToPILImage()(output[i])
        img = img.resize((224,224), resample=Image.Resampling[interpolation])
        imgs.append(TT.ToTensor()(img))
    output = torch.stack(imgs, dim=0)
    
    if sharpening > 0:
        output = contrast_adaptive_sharpening(output, sharpening)
    
    output = output.permute([0,2,3,1])

    return (output,)

class ResamplerImport(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class IPAdapterEncoderImport:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image_1": ("IMAGE",),
            "ipadapter_plus": ("BOOLEAN", { "default": False }),
            "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
            "weight_1": ("FLOAT", { "default": 1.0, "min": 0, "max": 1.0, "step": 0.01 }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "weight_2": ("FLOAT", { "default": 1.0, "min": 0, "max": 1.0, "step": 0.01 }),
                "weight_3": ("FLOAT", { "default": 1.0, "min": 0, "max": 1.0, "step": 0.01 }),
                "weight_4": ("FLOAT", { "default": 1.0, "min": 0, "max": 1.0, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("EMBEDS",)
    FUNCTION = "preprocess"
    CATEGORY = "ipadapter"

    def preprocess(self, clip_vision, image_1, ipadapter_plus, noise, weight_1, image_2=None, image_3=None, image_4=None, weight_2=1.0, weight_3=1.0, weight_4=1.0):
        weight_1 *= (0.1 + (weight_1 - 0.1))
        weight_1 = 1.19e-05 if weight_1 <= 1.19e-05 else weight_1
        weight_2 *= (0.1 + (weight_2 - 0.1))
        weight_2 = 1.19e-05 if weight_2 <= 1.19e-05 else weight_2
        weight_3 *= (0.1 + (weight_3 - 0.1))
        weight_3 = 1.19e-05 if weight_3 <= 1.19e-05 else weight_3
        weight_4 *= (0.1 + (weight_4 - 0.1))
        weight_5 = 1.19e-05 if weight_4 <= 1.19e-05 else weight_4

        image = image_1
        weight = [weight_1]*image_1.shape[0]
        
        if image_2 is not None:
            if image_1.shape[1:] != image_2.shape[1:]:
                image_2 = comfy.utils.common_upscale(image_2.movedim(-1,1), image.shape[2], image.shape[1], "bilinear", "center").movedim(1,-1)
            image = torch.cat((image, image_2), dim=0)
            weight += [weight_2]*image_2.shape[0]
        if image_3 is not None:
            if image.shape[1:] != image_3.shape[1:]:
                image_3 = comfy.utils.common_upscale(image_3.movedim(-1,1), image.shape[2], image.shape[1], "bilinear", "center").movedim(1,-1)
            image = torch.cat((image, image_3), dim=0)
            weight += [weight_3]*image_3.shape[0]
        if image_4 is not None:
            if image.shape[1:] != image_4.shape[1:]:
                image_4 = comfy.utils.common_upscale(image_4.movedim(-1,1), image.shape[2], image.shape[1], "bilinear", "center").movedim(1,-1)
            image = torch.cat((image, image_4), dim=0)
            weight += [weight_4]*image_4.shape[0]
        
        clip_embed = clip_vision.encode_image(image)
        neg_image = image_add_noise(image, noise) if noise > 0 else None
        
        if ipadapter_plus:
            clip_embed = clip_embed.penultimate_hidden_states
            if noise > 0:
                clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
            else:
                clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])
        else:
            clip_embed = clip_embed.image_embeds
            if noise > 0:
                clip_embed_zeroed = clip_vision.encode_image(neg_image).image_embeds
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)

        if any(e != 1.0 for e in weight):
            weight = torch.tensor(weight).unsqueeze(-1) if not ipadapter_plus else torch.tensor(weight).unsqueeze(-1).unsqueeze(-1)
            clip_embed = clip_embed * weight
        
        output = torch.stack((clip_embed, clip_embed_zeroed))

        return( output, )




class IPAdapterBatchEmbedsImport:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embed1": ("EMBEDS",),
            "embed2": ("EMBEDS",),
        }}

    RETURN_TYPES = ("EMBEDS",)
    FUNCTION = "batch"
    CATEGORY = "ipadapter"

    def batch(self, embed1, embed2):
        output = torch.cat((embed1, embed2), dim=1)
        return (output, )