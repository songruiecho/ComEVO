from dataloader import load_zero_shot_prompts, batchify, load_task_datas, prepare_batch_for_DeepSeek
from cfg import Config
import torch
from tqdm import tqdm
import os
import re
from transformers import BitsAndBytesConfig
from deepseek_vl2.utils.io import load_pil_images
import traceback
from PIL import Image
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from transformers import AutoModelForCausalLM
from utils import cal_acc

def load_VLM(cfg, model_path=''):
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 开启 8-bit 权重量化
        llm_int8_threshold=6.0,  # 默认阈值（可调）
        llm_int8_has_fp16_weight=False,  # 是否保留 FP16 权重（一般 False）
        llm_int8_enable_fp32_cpu_offload=True  # 避免不支持的 GPU kernel
    )
    if model_path == '':   # 空的时候指定path
        model_path = cfg.LLM_path+cfg.VLM
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = vl_gpt.to(torch.float16).cuda().eval()
    return model, processor

def VICL(cfg, batches, model, processor, task, model_name):
    results = []
    max_new_tokens = 20
    for batch in tqdm(batches, desc="VICL", total=len(batches), disable=len(batches) <= 1):
        try:
            messages = [b[0] for b in batch]
            answers = [b[1] for b in batch]
            all_answers = []
            conversations = prepare_batch_for_DeepSeek(cfg, messages)
            for c in conversations:
                images = [Image.open(path).convert("RGB") for path in c[0]['images']]
                prepare_inputs = processor(
                    conversations=c,
                    images=images,
                    force_batchify=True,
                    system_prompt="").to(model.device)
                vision_dtype = model.vision.patch_embed.proj.weight.dtype
                prepare_inputs.images = prepare_inputs.images.to(
                    device=model.device,
                    dtype=vision_dtype
                )
                with torch.no_grad():
                    inputs_embeds = model.prepare_inputs_embeds(
                        input_ids=prepare_inputs.input_ids,
                        images=prepare_inputs.images,
                        images_seq_mask=prepare_inputs.images_seq_mask,
                        images_spatial_crop=prepare_inputs.images_spatial_crop,
                        attention_mask=prepare_inputs.attention_mask,
                    )
                    outputs = model.language.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        bos_token_id=processor.tokenizer.bos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    all_answers.append(answer)
            assert len(all_answers) == len(answers)
            print(all_answers, answers)
            for out, answer in zip(all_answers, answers):
                out = re.sub(r'[\s\u3000\xa0]+', '', out).strip()
                if isinstance(answer, list):
                    results.append(out + '\t' + answer[0])
                else:
                    results.append(out + '\t' + answer)

        except Exception as e:
            traceback.print_exc()
            continue

    # 保存结果
    with open('results/{}_{}.txt'.format(task, model_name), 'w', encoding='utf-8-sig') as f:
        for line in results:
            f.write(line + '\n')

if __name__ == '__main__':
    cfg = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cfg.batch_size = 16
    for model_name in ['DeepSeek-VL2-tiny', 'DeepSeek-VL2-small']:
        assert 'deepseek-vl2' in model_name.lower()
        cfg.VLM = model_name
        model, processor = load_VLM(cfg)
        # for task in ['部件识别_1', '部件识别_2', '部件识别_3', '部件识别_4', '部件演化_1', '部件演化_2', '部件演化_3', '部件演化_4']:
        for task in ['部件识别_4']:
            cfg.task = task
            datas = load_task_datas(cfg)
            prompts = load_zero_shot_prompts(cfg, datas)
            batches = batchify(prompts, cfg.batch_size)
            VICL(cfg, batches, model, processor, task, model_name)
            cal_acc(task, model_name)