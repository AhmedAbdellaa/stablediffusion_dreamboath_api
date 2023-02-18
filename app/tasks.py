import time
import os 
import json
import shutil

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler


def sle(delay,usr):
    print(f"sleep for {delay} seconds")
    time.sleep(delay)
    print("finsih")
    return 1






def create_json(images_path,photo_path,json_path):
    #"concepts_list.json"
    print(images_path)
    if os.path.isdir(images_path) :
        print(photo_path)
        if os.path.isdir(photo_path):

            concepts_list = [
            {
                "instance_prompt":      "secourses",
                "class_prompt":         "portrait photo of a person",
                "instance_data_dir":    images_path,
                "class_data_dir":       photo_path
            }
            ]
            with open(json_path, "w") as f:
                json.dump(concepts_list, f, indent=4)
                print("**********************************jsondon**********************************")
            return True
        
    return False
def acc_model(Model_name,output_dir,vae_name,learning_rate,lr_scheduler,num_class,train_step,lr_warmup_steps,save_sample_prompt,
                instance_prompt,class_prompt,images_path,photo_path):
    
    command_str = f"""accelerate launch ./app/train_dreambooth.py \
  --pretrained_model_name_or_path="{Model_name}" \
  --pretrained_vae_name_or_path="{vae_name}" \
  --output_dir="{output_dir}" \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate={learning_rate} \
  --lr_scheduler="{lr_scheduler}" \
  --lr_warmup_steps={lr_warmup_steps}\
  --num_class_images={num_class} \
  --sample_batch_size=4 \
  --max_train_steps={train_step} \
  --save_interval=10000 \
  --save_sample_prompt="{save_sample_prompt}" \
  --instance_prompt="{instance_prompt}" \
  --class_prompt="{class_prompt}" \
  --instance_data_dir="{images_path}" \
  --class_data_dir="{photo_path}" 2>&1 | tee ./app/command.txt"""
    # print(command_str)
    with open('./app/command.txt','a') as f :
        f.write('**********************************************************************************\n')
    # time.sleep(10)
    result =  os.system(command_str)
    try :
        shutil.rmtree(images_path, ignore_errors=True)
    except : 
        pass
    try :
        shutil.rmtree(photo_path, ignore_errors=True)
    except :
        pass
    return result


def inferance_model(weights_dir_path,save_dir,
        prompt ="face portrait of secourses, symmetrical, detailed face, insanely detailed, concept art,\
         trending on artstation, daily deviations Highly Realistic sharp, elegant. by Rockstar Games, digital painting, looking into the camera",
        negative_prompt = "bad anatomy, bw, black and white, ugly, tiling, poorly drawn hands,\
             poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry bad anatomy, blurred"
             ,num_samples=5, guidance_scale=7.5,
              num_inference_steps=50, height=512, width=512,usr="",method="" ):

    if os.path.isdir(weights_dir_path):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(weights_dir_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

        g_cuda = None

        with autocast("cuda"), torch.inference_mode():
            images = pipe(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

        for index,img in enumerate(images):
            img.save(os.path.join(save_dir,str(index)+".jpg"))

def queue_turn(Model_name,output_dir,vae_name,learning_rate,lr_scheduler,num_class,train_step,lr_warmup_steps,save_sample_prompt,
                instance_prompt,class_prompt,images_path,photo_path, weights_dir_path,
save_dir,prompt, negative_prompt, num_samples, guidance_scale, num_inference_steps, height, width,usr,method):

    mo =  acc_model(Model_name,output_dir,vae_name,learning_rate,lr_scheduler,num_class,train_step,lr_warmup_steps,save_sample_prompt,
                instance_prompt,class_prompt,images_path,photo_path)
    # if "Error" not in mo :
    #     print("inf")
    inf = inferance_model(weights_dir_path,save_dir,prompt ,
        negative_prompt,num_samples, guidance_scale,num_inference_steps,
            height, width,usr,method)
        # command_str = f"""python ./app/inferance.py \
        #         --weights_dir_path=weights_dir_path
        #         --save_dir=save_dir \
        #         --prompt=prompt \
        #         --negative_prompt=negative_prompt \
        #         --num_samples=num_samples \
        #         --guidance_scale=guidance_scale \
        #         --save_infer_steps=save_infer_steps \
        #         --num_inference_steps=num_inference_steps \
        #         --height=height \
        #         --width=width """
        # inf= os.popen(command_str).read()
# else:
#     print("Error")
    
    # os.mkdir(secrets.token_urlsafe(6))
    
