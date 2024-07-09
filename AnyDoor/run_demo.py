import cv2
import einops
import numpy as np
import torch
import random
import os
import albumentations as A

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import base64

from PIL import Image
import torchvision.transforms as T
from datasets.data_utils import * 
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from omegaconf import OmegaConf
from cldm.hack import disable_verbosity, enable_sliced_attention


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/demo.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file
use_interactive_seg = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

if use_interactive_seg:
    from iseg.coarse_mask_refine_util import BaselineModel
    model_path = './iseg/coarse_mask_refine.pth'
    iseg_model = BaselineModel().eval()
    weights = torch.load(model_path , map_location='cpu')['state_dict']
    iseg_model.load_state_dict(weights, strict= True)


def process_image_mask(image_np, mask_np):
    img = torch.from_numpy(image_np.transpose((2, 0, 1)))
    img = img.float().div(255).unsqueeze(0)
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    pred = iseg_model(img, mask)['instances'][0,0].detach().numpy() > 0.5 
    return pred.astype(np.uint8)

def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 3 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]
    tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return tar_image

def inference_single_image(ref_image, 
                           ref_mask, 
                           tar_image, 
                           tar_mask, 
                           strength, 
                           ddim_steps, 
                           scale, 
                           seed,
                           enable_shape_control,
                           ):
    raw_background = tar_image.copy()
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control = enable_shape_control)

    ref = item['ref']
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": [control], 
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = ([strength] * 13)
    samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                     shape, cond, verbose=False, eta=0,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=un_cond)

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    tar_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 

    # keep background unchanged
    y1,y2,x1,x2 = item['tar_box_yyxx']
    raw_background[y1:y2, x1:x2, :] = tar_image[y1:y2, x1:x2, :]
    return raw_background


def process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8, enable_shape_control = False):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]

    ratio = np.random.randint(11, 15) / 10 #11,13
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # collage aug 
    masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
    tar_box_yyxx_full = tar_box_yyxx
    
    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]

    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0
    if enable_shape_control:
        collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]

    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    collage_mask[collage_mask == 2] = -1

    masked_ref_image = masked_ref_image  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
    
    item = dict(ref=masked_ref_image.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ),
                tar_box_yyxx=np.array(tar_box_yyxx_full),
                 ) 
    return item


ref_dir='./examples/Gradio/FG'
image_dir='./examples/Gradio/BG'
ref_list=[os.path.join(ref_dir,file) for file in os.listdir(ref_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file ]
ref_list.sort()
image_list=[os.path.join(image_dir,file) for file in os.listdir(image_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file]
image_list.sort()

def mask_image(image, mask):
    blanc = np.ones_like(image) * 255
    mask = np.stack([mask,mask,mask],-1) / 255
    masked_image = mask * ( 0.5 * blanc + 0.5 * image) + (1-mask) * image
    return masked_image.astype(np.uint8)

def refine_mask(ref):
    ref_image = ref["image"].convert("RGB")
    ref_mask = ref["mask"].convert("L")
    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    refined_ref_mask = process_image_mask(ref_image, ref_mask)
    refined_ref_mask_pil = Image.fromarray(refined_ref_mask * 255).convert("L")

    return ref_image, refined_ref_mask_pil


def run_local(base, ref, strength, ddim_steps, scale, seed, enable_shape_control):
    image = base["image"].convert("RGB")
    mask = base["mask"].convert("L")
    ref_image = ref["image"].convert("RGB")
    ref_mask = ref["mask"].convert("L")
    image = np.asarray(image)
    mask = np.asarray(mask)
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    synthesis = inference_single_image(ref_image.copy(), ref_mask.copy(), image.copy(), mask.copy(), 
                                       strength, ddim_steps, scale, seed, enable_shape_control)
    synthesis = torch.from_numpy(synthesis).permute(2, 0, 1)
    synthesis = synthesis.permute(1, 2, 0).numpy()
    return [synthesis]

def duplicate_ref_mask_to_background(ref_mask, base_mask):
    ref_mask_array = process_mask(ref_mask.image_data)
    base_mask_array = process_mask(base_mask.image_data)
    
    # Combine masks
    combined_mask = np.maximum(base_mask_array, ref_mask_array)
    
    # Convert back to PIL Image
    return Image.fromarray(combined_mask).convert("RGBA")

app = FastAPI()

# FastAPI endpoints
@app.post("/run_local")
async def api_run_local(base_image: UploadFile, base_mask: UploadFile, ref_image: UploadFile, ref_mask: UploadFile, strength: float, ddim_steps: int, scale: float, seed: int, enable_shape_control: bool):
    base_image = Image.open(io.BytesIO(await base_image.read())).convert("RGB")
    base_mask = Image.open(io.BytesIO(await base_mask.read())).convert("L")
    ref_image = Image.open(io.BytesIO(await ref_image.read())).convert("RGB")
    ref_mask = Image.open(io.BytesIO(await ref_mask.read())).convert("L")

    result = run_local({"image": base_image, "mask": base_mask},
                       {"image": ref_image, "mask": ref_mask},
                       strength, ddim_steps, scale, seed, enable_shape_control)

    buffered = io.BytesIO()
    Image.fromarray(result[0]).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image": img_str}

@app.post("/refine_mask")
async def api_refine_mask(ref_image: UploadFile, ref_mask: UploadFile):
    ref_image = Image.open(io.BytesIO(await ref_image.read())).convert("RGB")
    ref_mask = Image.open(io.BytesIO(await ref_mask.read())).convert("L")

    refined_image, refined_mask = refine_mask({"image": ref_image, "mask": ref_mask})

    buffered_mask = io.BytesIO()
    refined_mask.save(buffered_mask, format="PNG")
    mask_str = base64.b64encode(buffered_mask.getvalue()).decode()

    return {"mask": mask_str}

# Streamlit UI
def main():
    st.title("AnyDoor: Teleport your Target Objects!")

    # Output gallery
    output_image = st.empty()

    # Advanced options
    with st.expander("Advanced Options"):
        strength = st.slider("Control Strength", 0.0, 2.0, 1.0, 0.01)
        ddim_steps = st.slider("Steps", 1, 100, 30, 1)
        scale = st.slider("Guidance Scale", 0.1, 30.0, 4.5, 0.1)
        seed = st.slider("Seed", -1, 999999999, -1, 1)
        enable_shape_control = st.checkbox("Enable Shape Control", False)

    # Image upload and mask drawing
    col1, col2 = st.columns(2)
    
    # Background image and mask
    with col1:
        st.subheader("Background")
        base_image = st.file_uploader("Upload background image", type=["png", "jpg", "jpeg"])
        if base_image:
            base_image_pil = Image.open(base_image).convert("RGB")
            st.image(base_image_pil, use_column_width=True)
            
            drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"), key="base_drawing_mode")
            stroke_width = st.slider("Stroke width:", 1, 25, 3, key="base_stroke_width")
            stroke_color = st.color_picker("Stroke color:", "#FF0000", key="base_stroke_color")
            
            base_mask = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=f"{stroke_color}50",  # 50 is the alpha value for translucency
                background_image=base_image_pil,
                height=base_image_pil.height,
                width=base_image_pil.width,
                drawing_mode=drawing_mode,
                key="base_canvas",
            )

    # Reference image and mask
    with col2:
        st.subheader("Reference")
        ref_image = st.file_uploader("Upload reference image", type=["png", "jpg", "jpeg"])
        if ref_image:
            ref_image_pil = Image.open(ref_image).convert("RGB")
            st.image(ref_image_pil, use_column_width=True)
            
            drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"), key="ref_drawing_mode")
            stroke_width = st.slider("Stroke width:", 1, 25, 3, key="ref_stroke_width")
            stroke_color = st.color_picker("Stroke color:", "#FF0000", key="ref_stroke_color")
            
            ref_mask = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color=f"{stroke_color}50",  # 50 is the alpha value for translucency
                background_image=ref_image_pil,
                height=ref_image_pil.height,
                width=ref_image_pil.width,
                drawing_mode=drawing_mode,
                key="ref_canvas",
            )

    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate"):
            if base_image and base_mask and ref_image and ref_mask:
                base_mask_array = process_mask(base_mask.image_data)
                ref_mask_array = process_mask(ref_mask.image_data)
                
                response = requests.post(
                    "http://localhost:8000/run_local",
                    files={
                        "base_image": ("base_image.png", base_image.getvalue()),
                        "base_mask": ("base_mask.png", Image.fromarray(base_mask_array).tobytes()),
                        "ref_image": ("ref_image.png", ref_image.getvalue()),
                        "ref_mask": ("ref_mask.png", Image.fromarray(ref_mask_array).tobytes())
                    },
                    data={
                        "strength": strength,
                        "ddim_steps": ddim_steps,
                        "scale": scale,
                        "seed": seed,
                        "enable_shape_control": enable_shape_control
                    }
                )
                result = response.json()
                output_image.image(base64.b64decode(result["image"]))
            else:
                st.warning("Please upload all required images and draw masks.")

    with col2:
        if st.button("Refine Mask"):
            if ref_image and ref_mask:
                ref_mask_array = process_mask(ref_mask.image_data)
                response = requests.post(
                    "http://localhost:8000/refine_mask",
                    files={
                        "ref_image": ("ref_image.png", ref_image.getvalue()),
                        "ref_mask": ("ref_mask.png", Image.fromarray(ref_mask_array).tobytes())
                    }
                )
                result = response.json()
                st.image(base64.b64decode(result["mask"]), caption="Refined Mask")
            else:
                st.warning("Please upload reference image and draw a mask.")

    with col3:
        if st.button("Duplicate Ref Mask to Background"):
            if ref_mask is not None and base_mask is not None:
                new_base_mask = duplicate_ref_mask_to_background(ref_mask, base_mask)
            
                # Update the base_canvas with the new mask
                base_mask = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=stroke_width,
                    stroke_color=f"{stroke_color}50",
                    background_image=base_image_pil,
                    initial_drawing=new_base_mask,
                    height=base_image_pil.height,
                    width=base_image_pil.width,
                    drawing_mode=drawing_mode,
                    key="base_canvas_updated",
                )
            else:
                st.warning("Please draw both reference and background masks.")

def process_mask(mask_data):
    if isinstance(mask_data, np.ndarray):
        # If it's already a numpy array, convert to PIL Image
        mask_image = Image.fromarray(mask_data)
    elif isinstance(mask_data, Image.Image):
        # If it's already a PIL Image, use it directly
        mask_image = mask_data
    else:
        # If it's neither, try to create a PIL Image from it
        try:
            mask_image = Image.fromarray(np.array(mask_data))
        except:
            raise ValueError("Invalid mask data type. Expected numpy array or PIL Image.")
    
    
    mask_array = np.array(mask_image.convert("L"))
    
    binary_mask = (mask_array > 128).astype(np.uint8) * 255
    
    return binary_mask

def combine_masks(base_mask, ref_mask):
    return np.maximum(base_mask, ref_mask)

import uvicorn
import threading
import subprocess
import sys
import os

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8504)

def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--server.port", "8505"])

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Run Streamlit in the main thread
    run_streamlit()