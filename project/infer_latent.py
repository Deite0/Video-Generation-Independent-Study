from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid3
import torch
import datetime


pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
pretrained_model_path = "./checkpoints/mr-potato-head"

my_model_path = "./outputs/rabbit-watermelon_296_24"

unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

#### mallard

prompt = "a rabbit is eating pizza"

ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)

# Get current timestamp
timestamp = datetime.datetime.now().strftime("%m%d%H%M")

# Save video with timestamp in the filename
output_path = f"./{my_model_path}/video/{prompt}_{timestamp}.mp4"

video = pipe(prompt, latents=ddim_inv_latent, video_length=24, height=296, width=296, num_inference_steps=100, guidance_scale=12.5).videos

save_videos_grid3(video, output_path)