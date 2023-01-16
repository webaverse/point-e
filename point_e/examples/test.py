#!/usr/bin/env python
# coding: utf-8

# ## Point-E and StableDiffusion
# Possible improvements:
# - Increasing the resolution of the point cloud (8192 points?)
# - Move to stereo view: similar to stereo radiance fields, compute multiple prompts and use the best stereo pair as conditioning image input.
# https://github.com/jchibane/srf
# - Consistent multiple views with stable diffusion: 
# 1. using the same input noise?
# 2. Using textual inversion? https://huggingface.co/docs/diffusers/training/text_inversion

# convert girl.png -resize 512x512 -background white -gravity center -extent 512x512 girl2.png

# In[1]:


from PIL import Image
import torch
from tqdm.auto import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### text-to-img

# In[3]:


# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# load girl.png
generated_image = Image.open('girl2.png')


# In[4]:


# plt.imshow(generated_image)


# In[5]:


# PILtoTensor = trans#forms.ToTensor()
#generated_image = PILtoTensor(generated_image)

# del pipe


# In[6]:


# torch.cuda.empty_cache()


# ### img-to-pointcloud

# In[7]:


base_name = 'base1B' # base40M, use base300M or base1B for better results
n_points = 1024
final_n_points = 4096
MODEL_CONFIGS[base_name]["n_ctx"] = n_points
MODEL_CONFIGS["upsample"]["n_ctx"] = final_n_points - n_points

print('creating base model...')
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))


# In[8]:


sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[n_points, final_n_points - n_points], # points in cloud and missing ones for upsampling
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
)


# In[9]:


# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[generated_image]))):
    samples = x

print(samples.shape)

pc = sampler.output_to_point_clouds(samples)[0]

print(pc)
print(pc.coords)
print(pc.coords.shape)

coords = pc.coords
coords = coords.cpu()
coords = coords.numpy()
# convert to float32
coords = coords.astype('float32')
print(coords)
# convery tp bytes
coords_bytes = coords.tobytes()
print(len(coords_bytes))
# write to bytes.dat
with open('./pointcloud.f32', 'wb') as f:
    f.write(coords_bytes)
