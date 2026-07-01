from diffusers import AutoPipelineForImage2Image
from PIL import Image
import torch
from cfg import Config
import os
from diffusers import DiffusionPipeline
from dataloader import load_zero_shot_prompts, batchify, load_task_datas


def load_pipe(model_name):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "/home/lyj/models/{}".format(model_name),
        # "/home/lyj/models/Kolors-diffusers/",
        # "/home/lyj/models/FLUX.1-dev/",
        torch_dtype=torch.bfloat16,
        device_map="balanced"
    )
    return pipe

def VICL(pipe, idx, data, task, model_name):
    # 加载模型
    prompt = data['text'].replace("<image>", "")
    if type(data['image']) == list:
        input_image = Image.open(data['image'][0]).convert("RGB")
    else:
        input_image = Image.open(data['image']).convert("RGB")
    # 推理
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=25,
            guidance_scale=5.0,
            strength=0.7,    # 越接近1就越接近重绘
            width=1024,
            height=1024
        ).images[0]
    low_res_image = image.resize((128, 128), Image.LANCZOS)
    # 保存
    dir = "results/{}/{}".format(task, model_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    low_res_image.save("{}/{}.png".format(dir, idx))


if __name__ == '__main__':
    cfg = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    for model_name in ["Kolors-diffusers"]:
        pipe = load_pipe(model_name)
        for task in ['部件生成_0']:
            cfg.task = task
            datas = load_task_datas(cfg)
            for i, data in enumerate(datas):
                VICL(pipe, i, data, task, model_name)