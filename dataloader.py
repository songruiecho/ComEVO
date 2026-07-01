# 根据不同的任务加载、评估VLM
import json
import re
from traceback import print_tb

import torch
from PIL import Image
from tqdm import tqdm
from cfg import Config
try:
    from train_retriever import load_datas
except:
    pass

def load_task_datas(cfg, output_file=''):
    if output_file == '':
        output_file = f"BaseDatas/split_tasks/{cfg.task}_test.json"
    with open(output_file, "r", encoding="utf-8") as f:
        raw_datas = json.load(f)
    # 将raw_datas 处理成LLMs所需的数据格式
    return raw_datas

def load_train_datas(cfg, output_file=''):
    if output_file == '':
        output_file = f"BaseDatas/split_tasks/{cfg.task}_train.json"
    with open(output_file, "r", encoding="utf-8") as f:
        raw_datas = json.load(f)
    # 将raw_datas 处理成LLMs所需的数据格式
    return raw_datas


def load_zero_shot_prompts(cfg, datas):
    prompts = []
    for i, sample in tqdm(list(enumerate(datas)), desc="Loading Prompts"):
        content = []
        answer = sample['answer']
        if type(answer) == str:
            answer = [answer]
        else:
            if type(answer) != list:
                answer = answer.tolist()
        if type(sample['image']) == str:
            images = [sample['image']]
        else:
            images = sample['image']
        text = sample['text'].split('<image>')
        if len(text) == 1:
            content.append({"type": "text", "text": text[0]})
            for image in images:
                content.append({"type": "image", "image": image})
        else:   # 多个图要穿插着来
            for txt, img in zip(text, images):
                content.append({"type": "text", "text": txt})
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": text[-1]})
        messages = [{ "role": "user", "content": content}]
        prompts.append([messages, answer])
    return prompts

def load_few_shot_prompts(cfg, datas, shot=1):
    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "r", encoding="utf-8") as f:
        all_demos = json.load(f)
    prompts = []
    for i, sample in tqdm(list(enumerate(datas)), desc="Loading Prompts"):
        context = all_demos[i][:shot]
        context_text = '\n'.join([each['text'] for each in context])
        sample['text'] = context_text + '\n' + sample['text']
        if type(context[0]['image']) == str:
            context_img = [each['image'] for each in context]
        else:
            context_img = []
            for each in context:
                context_img.extend(each['image'])

        content = []
        answer = sample['answer']
        if type(answer) == str:
            answer = [answer]
        else:
            if type(answer) != list:
                answer = answer.tolist()
        if type(sample['image']) == str:
            images = [sample['image']]
        else:
            images = sample['image']
        images = context_img + images
        text = sample['text'].split('<image>')
        if len(text) == 1:
            content.append({"type": "text", "text": text[0]})
            for image in images:
                content.append({"type": "image", "image": image})
        else:   # 多个图要穿插着来
            for txt, img in zip(text, images):
                content.append({"type": "text", "text": txt})
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": text[-1]})
        messages = [{ "role": "user", "content": content}]
        prompts.append([messages, answer])

    return prompts

def batchify(iterable, batch_size):
    """将可迭代对象分成大小为 batch_size 的批次，并一次性存入内存"""
    return [iterable[i:i + batch_size] for i in range(0, len(iterable), batch_size)]

def safe_resize(img, min_size=30):
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    new_w = max(w, min_size)
    new_h = max(h, min_size)
    # 白色背景，不破坏图像内容
    new_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    new_img.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
    return new_img

from functools import lru_cache
@lru_cache(maxsize=4096)
def load_image(path):
    return Image.open(path)


def prepare_batch_for_InternVL(cfg, messages):
    """
    messages: List[List[Dict]]
    每条消息可以包含多段文本和多张图片：
    [
        [
            {"role": "user", "content": [
                {"type": "text", "text": "描述图像"},
                {"type": "image", "image": "img1.jpg"},p
                {"type": "text", "text": "这是另一段说明"},
                {"type": "image", "image": "img2.jpg"}
            ]}
        ],
        ...
    ]
    """
    batch_texts = []
    batch_images = []
    for chat in messages:
        text_parts = []
        imgs = []
        for turn in chat:
            for item in turn["content"]:
                if item["type"] == "text":
                    text_parts.append(item["text"])
                elif item["type"] == "image":
                    img = load_image(item["image"])
                    if 'glm' in cfg.VLM.lower():
                        img = safe_resize(img)
                    imgs.append(img)

        # 拼接文本，每段文本之间换行
        text_joined = "\n".join(text_parts)
        # 对应的 prompt 中图片占位符
        # 插入与图片数量相同的 <image> 占位符
        text_with_placeholders = "<IMG_CONTEXT>" * len(imgs) + "\n" + text_joined
        batch_texts.append(text_with_placeholders)
        batch_images.append(imgs)
    return batch_texts, batch_images


def prepare_batch_for_DeepSeek(cfg, batch):
    """
    将原始多模态数据转换为 DeepSeek-VL2 所需格式
    """
    conversations = []
    # 提取文本与图片
    for sample in batch:
        user_msg = sample[0]["content"]
        text_parts = []
        image_paths = []
        for item in user_msg:
            if item["type"] == "text":
                text_parts.append(item["text"])
            elif item["type"] == "image":
                image_paths.append(item["image"])
        # 构造 DeepSeek conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": (
                    "<image>".join(text_parts)
                ),
                "images": image_paths,
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            }
        ]
        conversations.append(conversation)
    return conversations


def prepare_batch_for_MiniCPM(cfg, messages):
    msgs = []
    for message_list in messages:
        for msg in message_list:
            text_parts = []
            images = []
            for content in msg["content"]:
                if content["type"] == "text":
                    text_parts.append(content["text"])
                elif content["type"] == "image":
                    # 打开图片并转为 RGB
                    img = Image.open(content["image"]).convert("RGB")
                    images.append(img)
            msgs.append([{
                'role': 'user',
                'content': images + [' '.join(text_parts)],
            }])
    return msgs


def find_demos_task1_1(cfg, train_datas, test_datas):
    ''' 仅以1为shots进行查找，因为山下文数量太多了
    :param cfg:
    :param datas:
    :param shot:
    :return:
    '''
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = qwen3vl_similarity["glyphs"]
    components = qwen3vl_similarity["components"]
    print(len(glyphs), len(components))
    all_demos = []
    for data in test_datas:
        if type(data['image']) == list:
            img = data['image'][0]
        else:
            img = data['image']
        try:
            idx = components.index(img)
        except:
            demos.append([])
        similarity = qwen3vl_similarity["com_similarity"][idx]
        top5_values, top5_indices = torch.topk(similarity, k=6)
        top5_indices = top5_indices.cpu().numpy().tolist()
        cand_com = [components[i] for i in top5_indices if components[i] != img]
        demos = []
        for each_cand in cand_com:
            cand_answer = re.split('[_ .]', each_cand)[-2]
            text = '给定部件图像<image>，其对应部件为{}'.format(cand_answer)
            demos.append({
                "text": text,
                "image": each_cand
            })
        all_demos.append(demos)

    output_file = "BaseDatas/split_tasks/部件识别_1_demos.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)


def find_demos_task1_2(cfg, train_datas, test_datas):
    ''' 仅以1为shots进行查找，因为山下文数量太多了
    :param cfg:
    :param datas:
    :param shot:
    :return:
    '''
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = qwen3vl_similarity["glyphs"]
    components = qwen3vl_similarity["components"]
    print(len(glyphs), len(components))
    glyph2coms = load_datas()

    all_demos = []
    for data in test_datas:
        if type(data['image']) == list:
            img = data['image'][0]
        else:
            img = data['image']
        try:
            idx = glyphs.index(img)
        except:
            demos.append([])
        similarity = glyph_jaccard["similarity"][idx]
        top5_values, top5_indices = torch.topk(similarity, k=6)
        top5_indices = top5_indices.cpu().numpy().tolist()
        cand_com = [glyphs[i] for i in top5_indices if glyphs[i] != img]
        demos = []
        for each_cand in cand_com:
            # 答案是直接给出对应的部件图片
            answer = glyph2coms[each_cand]
            text = '给定古文字图像<image>，其包含所有可能的部件为：{}'.format('<image>'*len(answer))
            demos.append({
                "text": text,
                "image": [each_cand] + answer
            })
        all_demos.append(demos)

    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)


def find_demos_task1_4(cfg, train_datas, test_datas):
    ''' 仅以1为shots进行查找，因为山下文数量太多了
    :param cfg:
    :param datas:
    :param shot:
    :return:
    '''
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = qwen3vl_similarity["glyphs"]
    components = qwen3vl_similarity["components"]
    print(len(glyphs), len(components))
    glyph2coms = load_datas()

    all_demos = []
    for data in test_datas:
        if type(data['image']) == list:
            img = data['image'][0]
        else:
            img = data['image']
        try:
            idx = glyphs.index(img)
        except:
            demos.append([])
        similarity = glyph_jaccard["similarity"][idx]
        top5_values, top5_indices = torch.topk(similarity, k=6)
        top5_indices = top5_indices.cpu().numpy().tolist()
        cand_com = [glyphs[i] for i in top5_indices if glyphs[i] != img]
        demos = []
        for each_cand in cand_com:
            # 答案是直接给出对应的部件图片
            answer = glyph2coms[each_cand]
            answer = [re.split('[_ .]', each_cand)[-2] for each_cand in answer]
            text = '给定古文字图像<image>，所有可能的部件对应的汉字为：{}'.format(';'.join(answer))
            demos.append({
                "text": text,
                "image": each_cand
            })
        print(demos)
        all_demos.append(demos)

    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)


def find_demos_task2_1(cfg, train_datas, test_datas):
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = qwen3vl_similarity["glyphs"]
    components = qwen3vl_similarity["components"]
    print(len(glyphs), len(components))
    glyph2coms = load_datas()

    all_demos = []
    for data in test_datas:
        if type(data['image']) == list:
            img = data['image'][0]
        else:
            img = data['image']
        try:
            idx = components.index(img)
        except:
            demos.append([])
        similarity = qwen3vl_similarity["com_similarity"][idx]
        top5_values, top5_indices = torch.topk(similarity, k=6)
        top5_indices = top5_indices.cpu().numpy().tolist()
        cand_com = [components[i] for i in top5_indices if components[i] != img]
        demos = []
        for each_cand in cand_com:
            answer = re.split('[/_ ]', each_cand)[-3]
            text = '给定古文字部件图像<image>，其对应的书体风格为：{}'.format(answer)
            demos.append({
                "text": text,
                "image": each_cand
            })
        all_demos.append(demos)

    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)


def find_demos_task2_2(cfg, train_datas, test_datas):
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = qwen3vl_similarity["glyphs"]
    components = qwen3vl_similarity["components"]
    print(len(glyphs), len(components))
    glyph2coms = load_datas()
    STYLES = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    all_demos = []
    for data in test_datas:
        try:
            if type(data['image']) == list:
                img = data['image'][0]    # 还是查找部件
            else:
                img = data['image']
            idx = components.index(img)
        except:
            demos.append([])
        similarity = qwen3vl_similarity["com_similarity"][idx]
        top5_values, top5_indices = torch.topk(similarity, k=6)
        top5_indices = top5_indices.cpu().numpy().tolist()
        cand_com = [components[i] for i in top5_indices if components[i] != img]
        demos = []
        for each_cand in cand_com:
            each_cand_styles, answer = [], []
            style = re.split('[/_ ]', each_cand)[-3]   # 找到所有具有不同风格的部件
            for sss in STYLES:
                each_cand_style = each_cand.replace(style, sss)
                if each_cand_style in components:
                    each_cand_styles.append(each_cand_style)
                    answer.append(sss)
            radical = re.split('[/_ .]', each_cand)[-2]
            text = '给定部件{}不同书体风格的图像{}，相应的书体风格为：{}'.format(radical, '<image>' * len(answer), ';'.join(answer))
            demos.append({
                "text": text,
                "image": each_cand_styles
            })
        all_demos.append(demos)

    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)


def find_demos_task3_0(cfg, train_datas, test_datas):
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = glyph_jaccard["glyphs"]
    components = qwen3vl_similarity["components"]
    glyph2coms = load_datas()
    STYLES = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    all_demos = []
    for data in test_datas:
        try:
            img, tgt = data['image'][0], data['answer']
            source_style = re.split('[/_ ]', img)[-2]
            target_style = re.split('[/_ ]', tgt)[-2]
            # 寻找top5的字形
            idx = glyphs.index(img)
            # similarity = glyph_jaccard["similarity"][idx]
            similarity = qwen3vl_similarity["glyph_similarity"][idx]
            top_values, top_indices = torch.topk(similarity, k=100)
            top_indices = top_indices.cpu().numpy().tolist()
            cand_samples = []
            for top_idx in top_indices[1:]:
                top_glyph = glyphs[top_idx]
                if source_style in top_glyph and top_glyph!=img:
                    top_target_glyph = top_glyph.replace(source_style, target_style)
                    if top_target_glyph in glyphs:
                        cand_samples.append([top_glyph, top_target_glyph, source_style, target_style])
                        if len(cand_samples) == 5:
                            break

            demos = []
            for sample in cand_samples:
                char = sample[0].split('/')[-2]
                text = '给定文字{}的{}风格图像{}，其{}风格图像为{}。'.format(char, sample[2],'<image>', sample[3], '<image>')
                demos.append({
                    "text": text,
                    "image": sample[:2]
                })
        except:
            demos = []
        all_demos.append(demos)
    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)


def find_demos_task3_1(cfg, train_datas, test_datas):
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = glyph_jaccard["glyphs"]
    components = qwen3vl_similarity["components"]
    glyph2coms = load_datas()
    STYLES = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    all_demos = []
    for data in test_datas:
        try:
            img, tgt = data['image'][0], data['answer']
            source_style = re.split('[/_ ]', img)[-2]
            match = re.search(r'该文字包含部件"([^"]+)"', data['text'])
            component = match.group(1)
            cand_coms = glyph2coms[data['image'][0]]
            cand_com = [com for com in cand_coms if component in com][0]
            idx = components.index(cand_com)
            similarity = qwen3vl_similarity["com_similarity"][idx]
            top_values, top_indices = torch.topk(similarity, k=100)
            top_indices = top_indices.cpu().numpy().tolist()
            cand_samples = []
            for top_idx in top_indices[1:]:
                top_com = components[top_idx]
                if source_style in top_com:
                    cand_samples.append([top_com, source_style, component])
                    if len(cand_samples) == 5:
                        break
            demos = []
            for sample in cand_samples:
                text = '部件{}的{}风格图像为{}'.format(sample[2], sample[1], '<image>')
                demos.append({
                    "text": text,
                    "image": [sample[0]]
                })
        except:
            demos = []
            continue
        all_demos.append(demos)
    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)

def find_demos_task3_3(cfg, train_datas, test_datas):
    qwen3vl_similarity = torch.load("retrieve_results/qwen3vl_similarity.pt", weights_only=False)
    glyph_jaccard = torch.load("retrieve_results/glyph_jaccard.pt", weights_only=False)
    glyph_semantic_similarity = torch.load("retrieve_results/glyph_semantic_similarity.pt", weights_only=False)
    glyphs = glyph_jaccard["glyphs"]
    components = qwen3vl_similarity["components"]
    glyph2coms = load_datas()
    STYLES = ['甲骨文', '金文', '篆文', '隶书', '楷书']
    all_demos = []
    for data in test_datas:
        # try:
            img, tgt = data['image'][0], data['answer']
            source_style = re.split('[/_ ]', img)[-2]
            target_style = re.split('[/_ ]', tgt)[-3]
            match = re.search(r'其包含部件“([^"]+)”', data['text'])
            component = match.group(1)
            cand_coms = glyph2coms[data['image'][0]]
            cand_com = [com for com in cand_coms if component in com][0]
            idx = components.index(cand_com)
            similarity = qwen3vl_similarity["com_similarity"][idx]
            top_values, top_indices = torch.topk(similarity, k=100)
            top_indices = top_indices.cpu().numpy().tolist()
            cand_samples = []
            for top_idx in top_indices[1:]:
                top_com = components[top_idx]
                # target_com = top_com.replace(source_style, target_style)
                if source_style in top_com and top_com.replace(source_style, target_style) in components:
                    cand_samples.append([top_com.replace(source_style, target_style), target_style, component])
                    if len(cand_samples) == 5:
                        break
            demos = []
            for sample in cand_samples:
                text = '部件{}的{}风格图像为{}'.format(sample[2], sample[1], '<image>')
                demos.append({
                    "text": text,
                    "image": [sample[0]]
                })
        # except:
        #     demos = []
        #     continue
            all_demos.append(demos)
    print(len(all_demos))
    output_file = "BaseDatas/split_tasks/{}_demos.json".format(cfg.task)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_demos, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    cfg = Config()
    cfg.task = '部件生成_3'
    test_datas = load_task_datas(cfg)
    train_datas = load_train_datas(cfg)
    find_demos_task3_3(cfg, train_datas, test_datas)
