
import numpy as np
from .imports.IPAdapterPlus import IPAdapterApplyImport, prep_image, IPAdapterEncoderImport
from .imports.AdvancedControlNet.nodes import AdvancedControlNetApplyImport


class IPAdapterAnimateNode:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "type_of_frame_distribution": (["linear", "dynamic"],),
                "linear_frame_distribution_value": ("INT", {"default": 8, "min": 4, "max": 64, "step": 1}),     
                "dynamic_frame_distribution_values": ("STRING", {"multiline": True, "default": "16,8,8,16"}),   
                "type_of_strength_distribution": (["linear", "dynamic"],),
                "linear_strength_value": ("STRING", {"multiline": False, "default": "(0.1,0.9)"}),
                "dynamic_strength_values": ("STRING", {"multiline": True, "default": "(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)"}),
                "relative_ipadapter_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "relative_cn_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "control_net": ("CONTROL_NET", {"default": None}),    
                "control_images": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("MODEL","CONDITIONING", "CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", "NET_INDEX", "LOGS")
    FUNCTION = "apply"

    CATEGORY = "IPAnimate"

    def apply(self, images, model, positive, negative, ipadapter, clip_vision, index, 
            type_of_frame_distribution, linear_frame_distribution_value, dynamic_frame_distribution_values,
            type_of_strength_distribution, linear_strength_value, dynamic_strength_values,
            relative_ipadapter_strength, relative_cn_strength, control_net=None, control_images=None
            ):

        def get_flow_image_index(images, type_of_frame_distribution, dynamic_frame_distribution_values, linear_frame_distribution_value):
            "获取处理流程图片列表索引"
            if type_of_frame_distribution == "dynamic":
                if isinstance(dynamic_frame_distribution_values, str):
                    flow_swap_nums =  [int(kf.strip()) for kf in dynamic_frame_distribution_values.split(',')]
                elif isinstance(dynamic_frame_distribution_values, list):
                    flow_swap_nums =  [int(kf) for kf in dynamic_frame_distribution_values]
                
            else:
                flow_swap_nums =  [linear_frame_distribution_value for i in range(len(images)-1)]

            flow_image_indexs = []
            for i, n in enumerate(flow_swap_nums):
                flow_image_indexs.extend([(i, i+1)]*n)
            return flow_image_indexs, flow_swap_nums

        def get_flow_strengths(flow_swap_nums, type_of_strength_distribution, dynamic_strength_values, linear_strength_value):
            "获取处理流程权重列表"
            if type_of_strength_distribution == "dynamic":
                if isinstance(dynamic_strength_values[0], str) and dynamic_strength_values[0] == "(":
                    string_representation = ''.join(dynamic_strength_values)
                    values = eval(f'[{string_representation}]')
                else:
                    values = dynamic_strength_values if isinstance(dynamic_strength_values, list) else [dynamic_strength_values]
            else:
                values = [eval(linear_strength_value) for _ in flow_swap_nums]

            flow_strengths = []
            for i, v in enumerate(values):
                v_min = min(v[0], v[1])
                v_max = max(v[0], v[1])
                v_min = v_min if v_min >= 0 else 0
                v_max = v_max if v_max <=1.0 else 1.0
                swap_num = flow_swap_nums[i]
                x = np.pi*(np.linspace(0, swap_num, swap_num)) / swap_num / 2
                strengths_2 = v_min + (v_max-v_min)*np.sin(x) 
                strengths_1 = v_min + (v_max-v_min)*np.cos(x)

                flow_strengths.extend(list(zip(strengths_1, strengths_2)))
 
            return flow_strengths


        # 是否使用controlnet
        use_cn = True
        if control_net is None or control_images is None:
            use_cn = False
            relative_cn_strength = 0

        assert len(images) > 1
        if use_cn:
            assert len(images) == len(control_images)

        # 获取处理流程图片列表索引
        flow_image_indexs, flow_swap_nums = get_flow_image_index(images, type_of_frame_distribution, dynamic_frame_distribution_values, linear_frame_distribution_value)
        # 获取处理流程权重列表
        flow_strengths = get_flow_strengths(flow_swap_nums, type_of_strength_distribution, dynamic_strength_values, linear_strength_value)
        
        assert len(flow_image_indexs) == len(flow_strengths)
        assert index < len(flow_image_indexs)

        if use_cn:
            apply_advanced_control_net = AdvancedControlNetApplyImport()
        ipadapter_application = IPAdapterApplyImport()
        ipadapter_encoder = IPAdapterEncoderImport()

        logs = {
            "index": index,
            "use_controlnet": use_cn,
            "image_index": flow_image_indexs[index],
            "ipadapter_strength": [i*relative_ipadapter_strength for i in flow_strengths[index]],
            "contronet_strength": [i*relative_cn_strength for i in flow_strengths[index]]
        }

        for i in range(2):
            if relative_ipadapter_strength > 0:
                # IP处理
                image = images[flow_image_indexs[index][i]]
                # 裁剪中间区域
                prepped_image = prep_image(image=image.unsqueeze(0), interpolation="LANCZOS", crop_position="pad", sharpening=0.0)[0]                        
                # 应用IPadapter
                embed, = ipadapter_encoder.preprocess(clip_vision, prepped_image, True, 0.0, 1.0)        
                model, = ipadapter_application.apply_ipadapter(ipadapter=ipadapter, model=model, weight=flow_strengths[index][i]*relative_ipadapter_strength, image=None, weight_type="linear", 
                                                        noise=0.0, embeds=embed, attn_mask=None, start_at=0.0, end_at=1.0, unfold_batch=False)    

            if relative_cn_strength > 0 and use_cn:
                # Contronet处理
                control_image = control_images[flow_image_indexs[index][i]]
                positive, negative = apply_advanced_control_net.apply_controlnet(positive, negative, control_net, control_image.unsqueeze(0), flow_strengths[index][i]*relative_cn_strength, 0.0, 1.0)
                

        return model, positive, negative, index+1, str(logs)


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "IPAdapterAnimate": IPAdapterAnimateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {    
    "IPAdapterAnimate": "IPAdapterAnimate by Chan"
}
