import os.path as osp
from PIL import Image
import numpy as np
import time

import torch
import torchvision.transforms as tvtrans
from extern.pfd.lib.cfg_helper import model_cfg_bank
from extern.pfd.lib.model_zoo import get_model

from collections import OrderedDict
from extern.pfd.lib.model_zoo.ddim import DDIMSampler

def load_sd_from_file(target):
    if osp.splitext(target)[-1] == '.ckpt':
        sd = torch.load(target, map_location='cpu')['state_dict']
    elif osp.splitext(target)[-1] == '.pth':
        sd = torch.load(target, map_location='cpu')
    elif osp.splitext(target)[-1] == '.safetensors':
        from safetensors.torch import load_file as stload
        sd = OrderedDict(stload(target, device='cpu'))
    else:
        assert False, "File type must be .ckpt or .pth or .safetensors"
    return sd

class PromptFreeDiffusionPipeline(object):
    def __init__(self, 
                 fp16=False, 
                 ctx_path=None,
                 diffuser_path=None,
                 ctl_path=None,
                 tag_ctx=None,
                 tag_diffuser=None,
                 tag_ctl=None,
                 device=None,
                 base_path=None,):

        self.ctx_path = ctx_path
        self.diffuser_path = diffuser_path
        self.ctl_path = ctl_path
        self.tag_ctx = tag_ctx
        self.tag_diffuser = tag_diffuser
        self.tag_ctl = tag_ctl
        self.strict_sd = True

        cfgm = model_cfg_bank()('pfd_seecoder_with_controlnet')
        if fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        self.net = get_model(base_path)(cfgm)
        
        self.action_load_ctx(ctx_path, tag_ctx)
        self.action_load_diffuser(diffuser_path, tag_diffuser)
        self.action_load_ctl(ctl_path, tag_ctl)

        if fp16:
            self.net.ctx['image'].fp16 = True
            self.net = self.net.half()
        
        self.device = device
        self.net.to(self.device)
            
        self.net.eval()
        self.sampler = DDIMSampler(self.net)

        self.n_sample_image = 1
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.image_latent_dim = 4

    def load_ctx(self, pretrained):
        sd = load_sd_from_file(pretrained)
        sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() \
            if ki.find('ctx.')!=0]
        sd.update(OrderedDict(sd_extra))

        self.net.load_state_dict(sd, strict=True)
        print('Load context encoder from [{}] strict [{}].'.format(pretrained, True))

    def load_diffuser(self, pretrained):
        sd = load_sd_from_file(pretrained)
        if len([ki for ki in sd.keys() if ki.find('diffuser.image.context_blocks.')==0]) == 0:
            sd = [(
                ki.replace('diffuser.text.context_blocks.', 'diffuser.image.context_blocks.'), vi) 
                    for ki, vi in sd.items()]
            sd = OrderedDict(sd)
        sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() \
            if ki.find('diffuser.')!=0]
        sd.update(OrderedDict(sd_extra))
        self.net.load_state_dict(sd, strict=True)
        print('Load diffuser from [{}] strict [{}].'.format(pretrained, True))

    def load_ctl(self, pretrained):
        sd = load_sd_from_file(pretrained)
        self.net.ctl.load_state_dict(sd, strict=True)
        print('Load controlnet from [{}] strict [{}].'.format(pretrained, True))

    def action_load_ctx(self, ctx_path, tag):
        pretrained = ctx_path
        if tag == 'SeeCoder-PA':
            from lib.model_zoo.seecoder import PPE_MLP
            pe_layer = \
                PPE_MLP(freq_num=20, freq_max=None, out_channel=768, mlp_layer=3)
            if self.dtype == torch.float16:
                pe_layer = pe_layer.half()
            if self.use_cuda:
                pe_layer.to('cuda')
            pe_layer.eval()
            self.net.ctx['image'].qtransformer.pe_layer = pe_layer
        else:
            self.net.ctx['image'].qtransformer.pe_layer = None
        if pretrained is not None:
            self.load_ctx(pretrained)
        self.tag_ctx = tag
        return tag

    def action_load_diffuser(self, diffuser_path, tag):
        pretrained = diffuser_path
        if pretrained is not None:
            self.load_diffuser(pretrained)
        self.tag_diffuser = tag
        return tag

    def action_load_ctl(self, ctl_path, tag):
        pretrained = ctl_path
        if pretrained is not None and pretrained != 'none':
            print(pretrained)
            self.load_ctl(pretrained)
        self.tag_ctl = tag
        return tag

    def action_autoset_hw(self, imctl):
        if imctl is None:
            return 512, 512
        w, h = imctl.size
        w = w//64 * 64
        h = h//64 * 64
        w = w if w >=512 else 512
        w = w if w <=1536 else 1536
        h = h if h >=512 else 512
        h = h if h <=1536 else 1536
        return h, w

    def action_inference(
            self, im, imctl, ctl_method, do_preprocess, 
            h, w, ugscale, seed):

        n_samples = self.n_sample_image

        sampler = self.sampler
        device = self.net.device

        w = w//64 * 64
        h = h//64 * 64
        if imctl is not None:
            imctl = imctl.resize([w, h], Image.Resampling.BICUBIC)

        craw = tvtrans.ToTensor()(im)[None].to(device).to(self.dtype)
        c = self.net.ctx_encode(craw, which='image').repeat(n_samples, 1, 1)
        u = torch.zeros_like(c)

        if self.tag_ctx in ["SeeCoder-Anime"]:
            u = torch.load('assets/anime_ug.pth')[None].to(device).to(self.dtype)
            pad = c.size(1) - u.size(1)
            u = torch.cat([u, torch.zeros_like(u[:, 0:1].repeat(1, pad, 1))], axis=1)

        if self.tag_ctl != 'none':
            ccraw = tvtrans.ToTensor()(imctl)[None].to(device).to(self.dtype)
            if do_preprocess:
                if(ctl_method in ['canny', 'canny_v11p']):
                    cc = self.net.ctl.preprocess(ccraw, type=ctl_method, size=[h, w], low_threshold=150, high_threshold=250)
                    cc = cc.to(self.dtype)
                else:
                    cc = self.net.ctl.preprocess(ccraw, type=ctl_method, size=[h, w])
                    cc = cc.to(self.dtype)
            else:
                cc = ccraw
        else:
            cc = None

        shape = [n_samples, self.image_latent_dim, h//8, w//8]

        if seed < 0:
            np.random.seed(int(time.time()))
            torch.manual_seed(-seed + 100)
        else:
            np.random.seed(seed + 100)
            torch.manual_seed(seed)

        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'image',},
            c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':ugscale,
                    'control':cc,},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)

        ccout = [tvtrans.ToPILImage()(i) for i in cc] if cc is not None else []
        imout = self.net.vae_decode(x, which='image')
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout + ccout