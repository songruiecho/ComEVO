import os
import re
import random
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from cfg import Config
from collections import defaultdict
from itertools import combinations

STYLES = ['甲骨文', '金文', '篆文', '隶书', '楷书']
STYLE_MAP = {
    '甲骨文': 'Oracle Bone Script',
    '金文': 'Bronze Script',
    '篆文': 'Seal Script',
    '隶书': 'Clerical Script',
    '楷书': 'Regular Script'
}


def statics():
    # 统计部件、字体的数量，以及每个字体、部件按照风格的分布，再给出部件对应的数量分布
    chars = {style: [] for style in STYLES}
    radicals = {style: [] for style in STYLES}
    for dir in os.listdir('BaseDatas/基础部件/'):
        for file in os.listdir('BaseDatas/基础部件/' + dir):
            path = 'BaseDatas/基础部件/' + dir + '/' + file
            for style in STYLES:
                if style in path:
                    chars[style].append(path)
                    break  # 假设一个path只属于一个style

    for dir in os.listdir('BaseDatas/文字拆分/'):
        for file in os.listdir('BaseDatas/文字拆分/' + dir):
            path = 'BaseDatas/文字拆分/' + dir + '/' + file
            for style in STYLES:
                if style in path:
                    if len(re.split('[_ ]', file)) == 3:   # 说明是部件
                        radicals[style].append(path)
                    else:   # 说明是字
                        chars[style].append(path)

    char_counts = {style: len(paths) for style, paths in chars.items()}
    radical_counts = {style: len(paths) for style, paths in radicals.items()}

    print("字符数量分布：")
    for style, count in char_counts.items():
        print(f"{style}: {count}")

    print("\n部件数量分布：")
    for style, count in radical_counts.items():
        print(f"{style}: {count}")

    total_chars = sum(char_counts.values())
    total_radicals = sum(radical_counts.values())

    print(f"字符总数: {total_chars}")
    print(f"部件总数: {total_radicals}")

    # 再统计不同radical的数量以及风格分布，为了加强对少数部件的理解，构建数据集的时候需要根据部件以及分布对其进行采样
    radical_freqs = {}
    for style, paths in radicals.items():
        for path in paths:
            target = path.split('/')[-1]
            target = target.replace('_', ' ').split(' ')[-1]
            target = re.sub(r'\d+', '', target).strip()[:-4]
            # if len(target) !=5:
            #     print(repr(path))
            if target not in radical_freqs:
                radical_freqs[target] = 1
            else:
                radical_freqs[target] += 1
    # 按频次排序（从高到低）
    sorted_freqs = sorted(radical_freqs.items(), key=lambda x: x[1], reverse=True)
    # 输出
    # for radical, freq in sorted_freqs[:100]:
    #     print(radical, freq)
    return sorted_freqs, char_counts, radical_counts, chars, radicals

def generate_instruction1_1(radical_counts, radicals):
    # 根据数据采样生成指令
    # ===========Radical Reg ===========================
    query = '请仔细观察给定的古文字部件图像<image>，识别其所对应的部件，并输出你判断最可能对应的5个现代汉字。请按照可能性从高到低排序，仅输出汉字本身，不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。'
    # 混合采样，通过古文字的类型以及部件的数量进行采样(10000，然后自动划分8000训练2000测试)
    # 直接随机采样10000条
    sample_size = 10000
    all_radicals = []
    for style, paths in radicals.items():
        for path in paths:
            all_radicals.append({
                "path": path,
                "style": style
            })
    if len(all_radicals) >= sample_size:
        sampled_data = random.sample(all_radicals, sample_size)
    else:
        sampled_data = random.choices(all_radicals, k=sample_size)
    data_for_model = []
    for each in sampled_data:
        path, time = each['path'], each['style']
        class_name = re.split('[_ ]', path)[-1][0]
        if time in STYLES:  # 只考虑5个time的
            data_for_model.append({
                "text": query,
                "image": path,  # 或者可以改成 base64 编码
                "answer": class_name,
                "time": time
            })
    # 4. 保存为 JSON 文件
    output_file = "BaseDatas/tasks/部件识别_1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def generate_instruction1_2(radical_counts, radicals):
    # 部件匹配：给定一个字以及字图，以及ABCD四个部件图候选，让模型自己判断哪个才是真正的部件图。
    data_for_model = []
    query = '请仔细观察给定的古文字"<char>"以及对应的图像<image>，从候选项A<image>；B<image>；C<image>；D<image>选出最有可能的部件图，仅输出A、B、C或D中的单个字母，不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。'
    # 直接随机采样10000条
    radical2feas = torch.load('radical_embeddings.pt')
    radical_paths = list(radical2feas.keys())
    radical_feas = [radical2feas[path] for path in radical_paths]
    radical_feas = torch.stack(radical_feas, dim=0)
    sample_size = 10000
    all_radicals = []
    for style, paths in radicals.items():
        for path in paths:
            all_radicals.append({
                "path": path,
                "style": style
            })
    if len(all_radicals) >= sample_size:
        sampled_data = random.sample(all_radicals, sample_size)
    else:
        sampled_data = random.choices(all_radicals, k=sample_size)
    # 生成采样部件的原始char，并选择相似的部件img作为候选答案
    char_paths, sampled_feas = [], []
    for data in sampled_data:
        path, time = data['path'], data['style']
        splits = re.split('[_ ]', path)
        char_path = '_'.join(splits[:2]) + '.jpg'
        char_paths.append(char_path)
        sampled_feas.append(radical2feas[path])
    sampled_feas = torch.stack(sampled_feas, dim=0)
    # 计算相似性
    sims = torch.matmul(sampled_feas, radical_feas.T)
    # 取最相似的 Top-k
    top_k = 1000
    top_values, top_indices = torch.topk(sims, k=top_k, dim=1)
    for i in tqdm(range(len(sampled_data))):
        data, indices, char_path = sampled_data[i], top_indices[i], char_paths[i]
        char = char_path.split('/')[2]
        top_paths = [radical_paths[id.item()] for id in indices][1:]  # 排除本身
        # 随后从中选取最相似的三个作为候选
        cand = []
        for path in top_paths:
            # 要确保一定不是该字本身的变体
            radical = re.split('[_ ]', path)[-1][0]
            if radical in data['path']:   # 说明是本部件
                continue
            cand.append(path)
            if len(cand) == 3:
                break
        cand = cand+[data['path']]
        random.shuffle(cand)
        # 找到正确答案的位置索引
        gt_index = cand.index(data["path"])
        # 分配 ABCD（如果是 4 个候选）
        labels = ["A", "B", "C", "D"]
        answer = labels[gt_index]
        data_for_model.append({
            'text': query.replace('<char>', char),
            'image': [char_path]+cand,
            'answer': answer,
        })
    output_file = "BaseDatas/tasks/部件识别_2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def generate_instruction1_3(radical_counts, radicals):
    data_for_model = []
    query = '请仔细观察给定的古文字"<char>"以及对应的图像<image>，从候选项A<image>；B<image>；C<image>；D<image>选出部件"<radical>"最有可能图片，仅输出A、B、C或D中的单个字母，不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。'
    # 直接随机采样10000条
    radical2feas = torch.load('radical_embeddings.pt')
    radical_paths = list(radical2feas.keys())
    radical_feas = [radical2feas[path] for path in radical_paths]
    radical_feas = torch.stack(radical_feas, dim=0)
    sample_size = 10000
    all_radicals = []
    for style, paths in radicals.items():
        for path in paths:
            all_radicals.append({
                "path": path,
                "style": style
            })
    if len(all_radicals) >= sample_size:
        sampled_data = random.sample(all_radicals, sample_size)
    else:
        sampled_data = random.choices(all_radicals, k=sample_size)
    char_paths, sampled_feas = [], []
    for data in sampled_data:
        path, time = data['path'], data['style']
        splits = re.split('[_ ]', path)
        char_path = '_'.join(splits[:2]) + '.jpg'
        char_paths.append(char_path)
        sampled_feas.append(radical2feas[path])
    sampled_feas = torch.stack(sampled_feas, dim=0)
    # 计算相似性
    sims = torch.matmul(sampled_feas, radical_feas.T)
    # 取最相似的 Top-k
    top_k = 1000
    top_values, top_indices = torch.topk(sims, k=top_k, dim=1)
    for i in tqdm(range(len(sampled_data))):
        data, indices, char_path = sampled_data[i], top_indices[i], char_paths[i]
        radical = re.split('[_ ]', data['path'])[-1][0]
        char = char_path.split('/')[2]
        top_paths = [radical_paths[id.item()] for id in indices][1:]  # 排除本身
        # 随后从中选取最相似的三个作为候选
        cand = []
        for path in top_paths:
            # 和上一个不同的是本部件也可以选，但是必须和图中的最吻合
            cand.append(path)
            if len(cand) == 3:
                break
        cand = cand+[data['path']]
        random.shuffle(cand)
        # 找到正确答案的位置索引
        gt_index = cand.index(data["path"])
        # 分配 ABCD（如果是 4 个候选）
        labels = ["A", "B", "C", "D"]
        answer = labels[gt_index]
        data_for_model.append({
            'text': query.replace('<char>', char).replace('<radical>', radical),
            'image': [char_path]+cand,
            'answer': answer,
        })
    output_file = "BaseDatas/tasks/部件识别_3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def generate_instruction1_4(radical_counts, radicals):
    data_for_model = []
    query = '请仔细观察给定的古文字"<char>"以及对应的图像<image>，给出所有可能的部件对应的汉字，不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。'
    sample_size = 10000
    all_chars = []
    for style, paths in radicals.items():
        for path in paths:
            if '文字拆分' in path:    # 注意只处理有部件的字
                all_chars.append({
                    "path": path,
                    "style": style
                })
    if len(all_chars) >= sample_size:
        sampled_data = random.sample(all_chars, sample_size)
    else:
        sampled_data = random.choices(all_chars, k=sample_size)
    for data in tqdm(sampled_data):
        char = data['path'].split('/')[2]
        style = data['path'].split('/')[-1].split('.')[0]
        style = re.split('[_ ]', style)[:2]
        style = '_'.join(style)
        char_dir = os.path.join('BaseDatas/文字拆分', char)
        char_path = os.path.join('BaseDatas/文字拆分/{}/'.format(char), style+'.jpg')
        radicals = os.listdir(char_dir)
        answers = []
        for radical in radicals:
            if style in radical and len(re.split('[_ ]', radical)) == 3:
                answers.append(re.split('[_ ]', radical)[-1][0])
        data_for_model.append({
            'text': query.replace('<char>', char),
            'image': char_path,
            'answer': ';'.join(answers),
        })
    output_file = "BaseDatas/tasks/部件识别_4.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)

def generate_instruction2_1(radicals):
    # 给定任一部件，判断该部件的书体风格
    query = '请仔细观察给定的古文字部件图像<image>，并从甲骨文、金文、篆文、隶书、楷书中选择你认为最可能的书体风格。不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。'
    # 混合采样，通过古文字的类型以及部件的数量进行采样(10000，然后自动划分8000训练2000测试)
    # 直接随机采样10000条
    sample_size = 10000
    all_radicals = []
    for style, paths in radicals.items():
        for path in paths:
            all_radicals.append({
                "path": path,
                "style": style
            })
    if len(all_radicals) >= sample_size:
        sampled_data = random.sample(all_radicals, sample_size)
    else:
        sampled_data = random.choices(all_radicals, k=sample_size)
    data_for_model = []
    for each in sampled_data:
        path, time = each['path'], each['style']
        class_name = re.split('[_ ]', path)[-1][0]
        if time in STYLES:  # 只考虑5个time的
            data_for_model.append({
                "text": query,
                "image": path,  # 或者可以改成 base64 编码
                "answer": time,
                "time": time
            })
    # 4. 保存为 JSON 文件
    output_file = "BaseDatas/tasks/部件演化_1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)



def generate_instruction2_3(radicals):
    # 跟2_1类似，只不过要对部件的演化进行排序
    # 首先选出没有发生数量讹变的部件以及发生过讹的部件，没有讹变的是规律演化，反之是不规律演化
    char2path = {}
    for style, paths in radicals.items():
        for path in paths:
            if '文字拆分' in path:  # 注意只处理有部件的字
                char = path.split('/')[2]
                if char not in char2path:
                    char2path[char] = {}
                if len(re.split('[_ ]', path)) == 3:
                    # 提取style
                    style = path.split('/')[-1].split('.')[0]
                    style = re.split('[_ ]', style)[:2]
                    style = '_'.join(style)
                    if style not in char2path[char].keys():
                        char2path[char][style] = [path]
                    else:
                        char2path[char][style].append(path)
    Corrupted_chars, UnCorrupted_chars = {}, {}
    for char in char2path.keys():
        # 如果甲骨文、金文、篆文、隶书、楷书有任意两个在keys中即要保留
        base_styles = set()
        for s in char2path[char].keys():
            base_style = s.split('_')[0]  # 例如 篆文_1 → 篆文
            base_styles.add(base_style)
        if len(base_styles) > 1:
            style2radical = {}
            # print(char2path[char])   # {'隶书_0': ['BaseDatas/文字拆分/嵚/隶书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/隶书_0 山.jpg'], '楷书_0': ['BaseDatas/文字拆分/嵚/楷书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/楷书_0 山.jpg']}
            for style in char2path[char].keys():
                radicals = []
                for path in char2path[char][style]:
                    radical = re.split('[_ ]', path)[-1][0]
                    radicals.append(radical)
                style2radical[style] = sorted(radicals)
            # 分别判断：长度变化 / 内容变化
            length_changed = False
            content_changed = False
            # 取第一个 style 作为基准
            first_radicals = list(style2radical.values())[0]
            first_len = len(first_radicals)
            for radicals in style2radical.values():
                # 先判断长度
                if len(radicals) != first_len:
                    length_changed = True
                # 再判断内容（只有长度相同才有意义）
                if len(radicals) == first_len and radicals != first_radicals:
                    content_changed = True
            if not length_changed and content_changed:
                Corrupted_chars[char] = char2path[char]
            if not length_changed and not content_changed:
                UnCorrupted_chars[char] = char2path[char]
    print(len(UnCorrupted_chars), len(Corrupted_chars))
    query = '给定文字"<char>"，其包含部件"<radical>"，请根据该部件在不同书体中的形态变化，从甲骨文、金文、篆文、隶书、楷书中选择每个部件的书体风格。不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。\n 形态变化如下：'
    data_for_model = []
    for char in tqdm(UnCorrupted_chars.keys()):
        # print(UnCorrupted_chars[char])
        cand_radicals = []
        grouped = defaultdict(list)
        for style in UnCorrupted_chars[char].keys():
            # char_path = os.path.join('BaseDatas/文字拆分', char, style+'.jpg')
            cand_radicals.extend(UnCorrupted_chars[char][style])
        # print(cand_radicals)
        for path in cand_radicals:
            filename = path.split('/')[-1].replace('.jpg', '')
            # 提取 "1 糸" 这种结构
            match = re.search(r'_(\d+)\s*(\S+)', filename)
            if match:
                key = f"{match.group(1)} {match.group(2)}"
                grouped[key].append(path)
        for g in grouped:
            group = grouped[g]
            random.shuffle(group)
            radical = re.split('[_ ]', group[0])[-1][0]
            prompt = query.replace("<char>", char).replace('<radical>', radical)
            prompt = prompt + ';'.join(['<image>']*len(group)) + '\n' + '书体风格为：'
            answers = []
            for path in group:
                style = path.split('_')[0].split('/')[-1]
                answers.append(style)
            data_for_model.append({
                'text': prompt,
                'image': group,
                'answer': ';'.join(answers),
            })
    if len(data_for_model) > 10000:
        data_for_model = random.sample(data_for_model, 10000)
    print(len(data_for_model))
    output_file = "BaseDatas/tasks/部件演化_3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def generate_instruction2_2(radicals):
    char2path = {}
    for style, paths in radicals.items():
        for path in paths:
            if '文字拆分' in path:  # 注意只处理有部件的字
                char = path.split('/')[2]
                if char not in char2path:
                    char2path[char] = {}
                if len(re.split('[_ ]', path)) == 3:
                    # 提取style
                    style = path.split('/')[-1].split('.')[0]
                    style = re.split('[_ ]', style)[:2]
                    style = '_'.join(style)
                    if style not in char2path[char].keys():
                        char2path[char][style] = [path]
                    else:
                        char2path[char][style].append(path)
    Corrupted_chars, UnCorrupted_chars = {}, {}
    for char in char2path.keys():
        # 如果甲骨文、金文、篆文、隶书、楷书有任意两个在keys中即要保留
        base_styles = set()
        for s in char2path[char].keys():
            base_style = s.split('_')[0]  # 例如 篆文_1 → 篆文
            base_styles.add(base_style)
        if len(base_styles) > 1:
            style2radical = {}
            # print(char2path[char])   # {'隶书_0': ['BaseDatas/文字拆分/嵚/隶书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/隶书_0 山.jpg'], '楷书_0': ['BaseDatas/文字拆分/嵚/楷书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/楷书_0 山.jpg']}
            for style in char2path[char].keys():
                radicals = []
                for path in char2path[char][style]:
                    radical = re.split('[_ ]', path)[-1]
                    radicals.append(radical)
                style2radical[style] = sorted(radicals)
            # 分别判断：长度变化 / 内容变化
            length_changed = False
            content_changed = False
            # 取第一个 style 作为基准
            first_radicals = list(style2radical.values())[0]
            first_len = len(first_radicals)
            for radicals in style2radical.values():
                # 先判断长度
                if len(radicals) != first_len:
                    length_changed = True
                # 再判断内容（只有长度相同才有意义）
                if len(radicals) == first_len and radicals != first_radicals:
                    content_changed = True
            if not length_changed and content_changed:
                Corrupted_chars[char] = char2path[char]
            if not length_changed and not content_changed:
                UnCorrupted_chars[char] = char2path[char]
    print(len(UnCorrupted_chars), len(Corrupted_chars))
    # 识别出突变的部件 首先加载部件表征用于生成相似候选
    radical2feas = torch.load('radical_embeddings.pt')
    radical_paths = list(radical2feas.keys())
    radical_feas = [radical2feas[path] for path in radical_paths]
    radical_feas = torch.stack(radical_feas, dim=0)
    print(radical_feas.shape)
    Evolution_dict = {}
    for char in tqdm(Corrupted_chars.keys()):
        data = Corrupted_chars[char]
        # print(char, data)
        part_sets = {}
        for stage, paths in data.items():
            parts = set()
            for path in paths:
                filename = os.path.basename(path)
                # 提取最后的部件名
                # "甲骨文_0 止1.jpg" -> 止1
                part = re.split('[_ ]', path)[-1].replace('.jpg', '')
                parts.add(part)
            part_sets[stage] = parts
        stages = list(part_sets.keys())
        evolution_pairs = []
        # 两两比较不同阶段
        for i in range(len(stages)):
            for j in range(i + 1, len(stages)):
                s1 = stages[i]
                s2 = stages[j]
                p1 = part_sets[s1]
                p2 = part_sets[s2]
                # 固定部件（交集）
                fixed_parts = p1 & p2
                # 演化部件（差集）
                diff1 = p1 - fixed_parts
                diff2 = p2 - fixed_parts
                # 若两边都只剩一个部件且style不一致，则认为发生演化
                if len(diff1) == 1 and len(diff2) == 1 and s1.split('_')[0] != s2.split('_')[0]:
                    old_part = list(diff1)[0]
                    new_part = list(diff2)[0]
                    # path还原
                    path1 = 'BaseDatas/文字拆分/{}/{} {}.jpg'.format(char, s1, old_part)
                    path2 = 'BaseDatas/文字拆分/{}/{} {}.jpg'.format(char, s2, new_part)
                    if path1 not in radical2feas.keys():
                        continue
                    if path2 not in radical2feas.keys():
                        continue
                    if old_part != new_part:   # 注意二者不能相同，要不然就是同等演化了
                        evolution_pairs.append({
                            'stage_pair': f'{s1}-{s2}',
                            'evolution': f'{old_part}-{new_part}',
                            'old_path': path1,
                            'new_path': path2
                        })
        Evolution_dict[char] = evolution_pairs

    # 构建prompt
    data_for_model = []
    query = '请仔细观察给定的古文字“<char>”以及对应的图像<image>，其包含部件<image>“<radical>”，在候选项 A<image>;B<image>;C<image>;D<image>中，选择该部件在“<style>”书体下最可能对应的形态。仅输出A、B、C或D中的单个字母，不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。'
    for char in tqdm(Evolution_dict.keys()):
        for pair in Evolution_dict[char]:
            stage_pair = pair['stage_pair']
            evolution = pair['evolution']
            stage1, stage2 = stage_pair.split('-')
            style = stage2.split('_')[0]
            radical1, radical2 = evolution.split('-')
            path1, path2 = pair['old_path'], pair['new_path']
            path2_feas = radical2feas[path2]
            sims = torch.matmul(path2_feas, radical_feas.T).squeeze(0)
            top_values, top_indices = torch.topk(sims, k=100)
            cand_radicals = []
            for idx in top_indices.tolist():
                if radical2.strip() not in radical_paths[idx]:   # 排除相同的部件，选择最相似的top3
                    cand_radicals.append(radical_paths[idx])
            cand_radicals = cand_radicals[:3]
            # print(radical2, cand_radicals)
            cand_radicals = cand_radicals + [path2]
            random.shuffle(cand_radicals)
            gt_index = cand_radicals.index(path2)
            # 分配 ABCD（如果是 4 个候选）
            labels = ["A", "B", "C", "D"]
            answer = labels[gt_index]
            char_path = 'BaseDatas/文字拆分/{}/{}.jpg'.format(char, stage1)
            data_for_model.append({
                'text': query.replace("<char>", char).replace("<radical>", radical1).replace("<style>", style.split('_')[0]),
                'image': [char_path, path1] + cand_radicals,
                'answer': answer,
            })
    print(len(data_for_model))
    output_file = "BaseDatas/tasks/部件演化_2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def radical_embd(radicals):
    config = Config()
    all_radicals = []
    for style, paths in radicals.items():
        for path in paths:
            all_radicals.append(path)
    # 计算所有的embeddings
    model = CLIPModel.from_pretrained(config.CLIP_path).cuda()
    processor = CLIPProcessor.from_pretrained(config.CLIP_path)
    radical_embeddings = {}
    batch_size = 128
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_radicals), batch_size)):
            batch_paths = all_radicals[i:i + batch_size]
            batch_images = []
            valid_paths = []
            # 批量读取图片
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            if len(batch_images) == 0:
                continue
            # CLIP batch 编码
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {
                k: v.cuda()
                for k, v in inputs.items()
            }
            image_features = model.get_image_features(**inputs)
            # L2 normalize（便于相似度计算）
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu()
            # 保存结果
            for path, feat in zip(valid_paths, image_features):
                radical_embeddings[path] = feat
    save_path = "radical_embeddings.pt"
    torch.save(radical_embeddings, save_path)
    print(f"Saved to {save_path}")


def generate_instruction2_4(radicals):
    char2path = {}
    for style, paths in radicals.items():
        for path in paths:
            if '文字拆分' in path:  # 注意只处理有部件的字
                char = path.split('/')[2]
                if char not in char2path:
                    char2path[char] = {}
                if len(re.split('[_ ]', path)) == 3:
                    # 提取style
                    style = path.split('/')[-1].split('.')[0]
                    style = re.split('[_ ]', style)[:2]
                    style = '_'.join(style)
                    if style not in char2path[char].keys():
                        char2path[char][style] = [path]
                    else:
                        char2path[char][style].append(path)
    Corrupted_chars, UnCorrupted_chars = {}, {}
    for char in char2path.keys():
        # 如果甲骨文、金文、篆文、隶书、楷书有任意两个在keys中即要保留
        base_styles = set()
        for s in char2path[char].keys():
            base_style = s.split('_')[0]  # 例如 篆文_1 → 篆文
            base_styles.add(base_style)
        if len(base_styles) > 1:
            style2radical = {}
            # print(char2path[char])   # {'隶书_0': ['BaseDatas/文字拆分/嵚/隶书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/隶书_0 山.jpg'], '楷书_0': ['BaseDatas/文字拆分/嵚/楷书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/楷书_0 山.jpg']}
            for style in char2path[char].keys():
                radicals = []
                for path in char2path[char][style]:
                    radical = re.split('[_ ]', path)[-1]
                    radicals.append(radical)
                style2radical[style] = sorted(radicals)
            # 分别判断：长度变化 / 内容变化
            length_changed = False
            content_changed = False
            # 取第一个 style 作为基准
            first_radicals = list(style2radical.values())[0]
            first_len = len(first_radicals)
            for radicals in style2radical.values():
                # 先判断长度
                if len(radicals) != first_len:
                    length_changed = True
                # 再判断内容（只有长度相同才有意义）
                if len(radicals) == first_len and radicals != first_radicals:
                    content_changed = True
            if not length_changed and content_changed:
                Corrupted_chars[char] = char2path[char]
            if not length_changed and not content_changed:
                UnCorrupted_chars[char] = char2path[char]
    print(len(UnCorrupted_chars), len(Corrupted_chars))
    # 识别出突变的部件 首先加载部件表征用于生成相似候选
    radical2feas = torch.load('radical_embeddings.pt')
    radical_paths = list(radical2feas.keys())
    radical_feas = [radical2feas[path] for path in radical_paths]
    radical_feas = torch.stack(radical_feas, dim=0)
    # 根据Corrupted_chars生成突变的演化路径
    Evolution_dict = {}
    for char in tqdm(Corrupted_chars.keys()):
        data = Corrupted_chars[char]
        # print(char, data)
        part_sets = {}
        for stage, paths in data.items():
            parts = set()
            for path in paths:
                filename = os.path.basename(path)
                # 提取最后的部件名
                # "甲骨文_0 止1.jpg" -> 止1
                part = re.split('[_ ]', path)[-1].replace('.jpg', '')
                parts.add(part)
            part_sets[stage] = parts
        stages = list(part_sets.keys())
        evolution_pairs = []
        # 两两比较不同阶段
        for i in range(len(stages)):
            for j in range(i + 1, len(stages)):
                s1 = stages[i]
                s2 = stages[j]
                p1 = part_sets[s1]
                p2 = part_sets[s2]
                # 固定部件（交集）
                fixed_parts = p1 & p2
                # 演化部件（差集）
                diff1 = p1 - fixed_parts
                diff2 = p2 - fixed_parts
                # 若两边都只剩一个部件且style不一致，则认为发生演化
                if len(diff1) == 1 and len(diff2) == 1 and s1.split('_')[0] != s2.split('_')[0]:
                    old_part = list(diff1)[0]
                    new_part = list(diff2)[0]
                    # path还原
                    path1 = 'BaseDatas/文字拆分/{}/{} {}.jpg'.format(char, s1, old_part)
                    path2 = 'BaseDatas/文字拆分/{}/{} {}.jpg'.format(char, s2, new_part)
                    if path1 not in radical2feas.keys():
                        continue
                    if path2 not in radical2feas.keys():
                        continue
                    if old_part != new_part:  # 注意二者不能相同，要不然就是同等演化了
                        evolution_pairs.append({
                            'stage_pair': f'{s1}-{s2}',
                            'evolution': f'{old_part}-{new_part}',
                            'old_path': path1,
                            'new_path': path2
                        })
        Evolution_dict[char] = evolution_pairs
    # 提取演化路径
    data_for_model = []
    query = '给定文字"<char>"，其包含部件"<radical>"，请根据该部件在不同书体中的形态变化，从甲骨文、金文、篆文、隶书、楷书中选择每个部件的书体风格。不得包含解释、分析、推理过程、标点符号、序号、示例或任何其他额外文字。\n 形态变化如下：'
    for char in tqdm(Evolution_dict.keys()):
        pairs = Evolution_dict[char]
        # 保存：
        # stage -> part
        stage2part = {}
        for pair in pairs:
            stage1, stage2 = pair['stage_pair'].split('-')
            part1, part2 = pair['evolution'].split('-')
            style1 = stage1.split('_')[0]
            style2 = stage2.split('_')[0]
            path1, path2 = pair['old_path'], pair['new_path']
            stage2part[style1] = path1
            stage2part[style2] = path2
        evolution_path = []
        for style in STYLES:
            if style in stage2part:
                evolution_path.append(
                    f'{style}:{stage2part[style]}'
                )
        # 拼接成字符串
        random.shuffle(evolution_path)
        try:
            center_radical = [each for each in evolution_path if '楷书' in each][0]    # 没有楷书的跳过吧
        except:
            continue
        center_radical = center_radical.split(' ')[-1][0]
        prompt = query.replace("<char>", char).replace('<radical>', center_radical)
        prompt = prompt + ';'.join(['<image>'] * len(evolution_path)) + '\n' + '书体风格为：'
        answers = ';'.join([each.split(':')[0] for each in evolution_path])
        imgs = [each.split(':')[1] for each in evolution_path]
        print(prompt, answers, imgs)
        data_for_model.append({
                'text': prompt,
                'image': imgs,
                'answer': answers,
            })
    output_file = "BaseDatas/tasks/部件演化_4.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)

def split_tasks():
    # 将task下的所有json按照随机9：1划分成训练和测试集
    print(os.listdir("BaseDatas"))
    for json_path in os.listdir("BaseDatas/tasks"):
        if json_path not in ['部件生成_1.json']:
            continue
        with open("BaseDatas/tasks/"+json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 条目级随机打乱
            random.shuffle(data)
            train_data = []
            test_data = []
            split_idx = int(len(data) * 0.9)
            train_data.extend(data[:split_idx])
            test_data.extend(data[split_idx:])
            # 保存为统一的数据集文件
            with open("BaseDatas/split_tasks/{}_train.json".format(json_path.split(".")[0]), "w", encoding="utf-8") as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            with open("BaseDatas/split_tasks/{}_test.json".format(json_path.split(".")[0]), "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)


def generate_instruction3_0(radicas):
    # 添加一个文字生成的任务：对于某个书体风格下的字，去生成另一个风格下的字
    query = '请仔细观察给定的“<char>”字图像<image>，在保持原文字结构、笔画组成与语义内容一致的前提下，生成其对应的<style>书体风格字形图像。生成结果中不得包含其他部件或额外内容。'
    # 混合采样，通过古文字的类型以及部件的数量进行采样(10000，然后自动划分8000训练2000测试)
    # 直接随机采样10000条
    sample_size = 10000
    all_radicals = []
    for style, paths in radicals.items():
        for path in paths:
            all_radicals.append({
                "path": path,
                "style": style
            })
    # 按照字对所有的字进行分组
    char2paths = {}
    for radical in all_radicals:
        char_path = re.split('[_ ]', radical["path"])[:-1]
        char_path = '_'.join(char_path)+ '.jpg'
        char = char_path.split('/')[-2]
        if char not in char2paths.keys():
            char2paths[char] = [char_path]
        elif char_path not in char2paths[char]:
            char2paths[char].append(char_path)
    pairs = []
    for char, paths in char2paths.items():
        if len(paths) > 1:
            pairs.extend(list(combinations(paths, 2)))
    samples = []
    for pair in pairs:
        s1 = re.split('[_ /]', pair[0])[-2]
        s2 = re.split('[_ /]', pair[1])[-2]
        if s1 != s2:
            samples.append(pair)
    samples = random.sample(samples, sample_size)
    data_for_model = []
    for sample in samples:
        char = sample[0].split('/')[-2]
        s2 = re.split('[_ /]', sample[1])[-2]
        data_for_model.append({
            'text': query.replace('<char>', char).replace('<style>', s2),
            'image': [sample[0]],
            'answer': sample[1]
        })
    output_file = "BaseDatas/tasks/部件生成_0.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)



def generate_instruction3_1(radicals):
    query = '请观察给定古文字图像<image>。该文字包含部件"<radical>"。请仅生成该部件在原图中的对应形态，并保持与原图一致的书体风格、结构与笔画特征。生成结果中不得包含其他部件或额外内容。'
    # 混合采样，通过古文字的类型以及部件的数量进行采样(10000，然后自动划分8000训练2000测试)
    # 直接随机采样10000条
    sample_size = 10000
    all_radicals = []
    for style, paths in radicals.items():
        for path in paths:
            all_radicals.append({
                "path": path,
                "style": style
            })
    if len(all_radicals) >= sample_size:
        sampled_data = random.sample(all_radicals, sample_size)
    else:
        sampled_data = random.choices(all_radicals, k=sample_size)
    data_for_model = []
    for each in tqdm(sampled_data):
        path = each['path']
        match = re.match(r'^(.*?_\d+)', path)
        if match:
            time = match[0]
        class_name = re.split('[_ ]', path)[-1][0]
        char = path.split('/')[-2]
        char_path = '{}.jpg'.format(time)
        data_for_model.append({
            'text': query.replace('<radical>', class_name),
            'image': [char_path],
            'answer': path,
        })
    output_file = "BaseDatas/tasks/部件生成_1.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def generate_instruction3_2(chars):
    query = '请观察给定古文字图像<image>，请生成其中包含的所有部件，并保持其与原图一致的书体风格、结构与笔画特征。生成结果中不得包含其他部件或额外内容。'
    all_chars = []
    sample_size = 10000
    for char in chars.keys():
        all_chars.extend(chars[char])
    if len(all_chars) >= sample_size:
        sampled_data = random.sample(all_chars, sample_size)
    else:
        sampled_data = random.choices(all_chars, k=sample_size)
    data_for_model = []
    for data in tqdm(sampled_data):
        char_dir = '/'.join(data.split('/')[:-1])
        cand_paths = os.listdir(char_dir)
        cand_paths = [char_dir+'/'+each for each in cand_paths if each.endswith('.jpg')]
        char_name = data.replace('.jpg', '')
        radicals = [cand_path for cand_path in cand_paths if cand_path.startswith(char_name) and cand_path != data]
        data_for_model.append({
            'text': query,
            'image': data,
            'answer': radicals,
        })
    output_file = "BaseDatas/tasks/部件生成_2.json"
    def clean_surrogates(obj):
        if isinstance(obj, str):
            return obj.encode('utf-8', 'ignore').decode('utf-8')
        elif isinstance(obj, list):
            return [clean_surrogates(i) for i in obj]
        elif isinstance(obj, dict):
            return {clean_surrogates(k): clean_surrogates(v) for k, v in obj.items()}
        return obj

    data_for_model = clean_surrogates(data_for_model)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


def generate_instruction3_3(radicals):
    query = '请仔细观察给定的古文字字图像<image>，其包含部件“<radical>”，生成其对应的<style>书体风格的部件。生成结果中不得包含其他部件或额外内容。'
    char2path = {}
    for style, paths in radicals.items():
        for path in paths:
            if '文字拆分' in path:  # 注意只处理有部件的字
                char = path.split('/')[2]
                if char not in char2path:
                    char2path[char] = {}
                if len(re.split('[_ ]', path)) == 3:
                    # 提取style
                    style = path.split('/')[-1].split('.')[0]
                    style = re.split('[_ ]', style)[:2]
                    style = '_'.join(style)
                    if style not in char2path[char].keys():
                        char2path[char][style] = [path]
                    else:
                        char2path[char][style].append(path)
    Corrupted_chars, UnCorrupted_chars = {}, {}
    for char in tqdm(char2path.keys()):
        # 如果甲骨文、金文、篆文、隶书、楷书有任意两个在keys中即要保留
        base_styles = set()
        for s in char2path[char].keys():
            base_style = s.split('_')[0]  # 例如 篆文_1 → 篆文
            base_styles.add(base_style)
        if len(base_styles) > 1:
            style2radical = {}
            # print(char2path[char])   # {'隶书_0': ['BaseDatas/文字拆分/嵚/隶书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/隶书_0 山.jpg'], '楷书_0': ['BaseDatas/文字拆分/嵚/楷书_0 欽.jpg', 'BaseDatas/文字拆分/嵚/楷书_0 山.jpg']}
            for style in char2path[char].keys():
                radicals = []
                for path in char2path[char][style]:
                    radical = re.split('[_ ]', path)[-1][0]
                    radicals.append(radical)
                style2radical[style] = sorted(radicals)
            # 分别判断：长度变化 / 内容变化
            length_changed = False
            content_changed = False
            # 取第一个 style 作为基准
            first_radicals = list(style2radical.values())[0]
            first_len = len(first_radicals)
            for radicals in style2radical.values():
                # 先判断长度
                if len(radicals) != first_len:
                    length_changed = True
                # 再判断内容（只有长度相同才有意义）
                if len(radicals) == first_len and radicals != first_radicals:
                    content_changed = True
            if not length_changed and content_changed:
                Corrupted_chars[char] = char2path[char]
            if not length_changed and not content_changed:
                UnCorrupted_chars[char] = char2path[char]
    # print(len(UnCorrupted_chars), len(Corrupted_chars))
    # 提取无变异的部件作为标准答案
    data_for_model, pairs = [], []
    for char in tqdm(UnCorrupted_chars.keys()):
        # 提取：部件 -> [(时期, 路径)]
        data = UnCorrupted_chars[char]
        radical2paths = defaultdict(list)
        for period_key, paths in data.items():
            period = period_key.split('_')[0]
            for path in paths:
                # 文件名形如：甲骨文_0 日.jpg
                radical = path.split(' ')[-1].replace('.jpg', '')
                radical2paths[radical].append((period, path))
        for radical, items in radical2paths.items():
            # 两两组合
            for (p1, path1), (p2, path2) in combinations(items, 2):
                # 只保留“不同时期”
                if p1 != p2:
                    pairs.append({
                        'radical': radical,
                        'period1': p1,
                        'path1': path1,
                        'period2': p2,
                        'path2': path2
                    })
    pairs = random.sample(pairs, 10000)
    for pair in pairs:
        splits = re.split('[_ ]', pair['path1'])
        char_path = '_'.join(splits[:2]) + '.jpg'
        data_for_model.append({
            'text': query.replace('<radical>', pair['radical']).replace('<style>', pair['period2']),
            'image': [char_path],
            'answer': pair['path2'],
        })

    output_file = "BaseDatas/tasks/部件生成_3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_for_model, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    sorted_freqs, char_counts, radical_counts, chars, radicals = statics()
    # vis_pie(char_counts, radical_counts)
    # vis_radical_dis(sorted_freqs)
    # generate_instruction1_1(radical_counts, radicals)
    generate_instruction1_2(radical_counts, radicals)
    # generate_instruction1_3(radical_counts, radicals)
    # generate_instruction1_4(radical_counts, radicals)
    # radical_embd(radicals)
    # generate_instruction2_1(radicals)
    # generate_instruction2_3(radicals)
    # generate_instruction2_2(radicals)
    # generate_instruction2_4(radicals)
    # generate_instruction3_0(radicals)
    # generate_instruction3_1(radicals)
    # generate_instruction3_2(chars)
    # generate_instruction3_3(radicals)
    split_tasks()

# 任务定义：
# 部件识别
# 1. 基础部件识别：给定一张部件图片识别图片的内容对应的top5现代汉字。
# 2. 部件匹配：给定一个字以及字图，以及ABCD四个部件图候选，让模型自己判断哪个才是真正的部件图。
# 3. 部件详细匹配：给定一个字以及字图，以及其中的某个部件，让模型从ABCD四个候选中提取最正确的部件图。
# 3. 字形部件理解：给一个字以及其字图，不论什么风格，问模型其中包含的全部可能部件对应的现代汉字。

# 部件演化
# 1. 部件风格识别，给出部件图识别其style。
# 2. 规律演化路径排序，输入同一个部件的打乱顺序，让模型总结正确时间顺序，也是GEVO的字形演化的拓展。
# 3. 变异演化部件匹配：给一个字对应的部件的某个时期，从ABCD中选出对应的其它演化时期的对应部件[注意这个部件同原字的应该不是一个]。
# 4. 变异演化路径排序，输入同一个部件的打乱顺序，让模型总结正确时间顺序，同样这个部件同原字的应该不是一个。

# 部件提取（基于图生成模型的）
# 部件提取（这个需要图生成模型）
# 1. 条件部件生成。对于某个字图，告诉他其中的部件（宿，宀）让其生成标准化部件。然后将生成的结果与基准中的所有部件字形做matching，以topK-matching（也就是topk中到底有几个与“宀”匹配了）的程度进行得分计算。
# 2. 无条件部件生成。对于某个字图，就让他生成所有的可能部件，然后同样对每个结果进行总体的matching。【需要多次采样并生成结果】
# 3. 部件风格迁移。给定一个字图，生成其某个部件在某个书体风格下的演化结果，然后对结果进行matching（这个超级难）。


# RAG方法：[可以视为多模态的上下文学习]
# 形意驱动的多模态检索增强。从字形字义两方面进行检索增强，然后利用ICL对预测结果进行促进。
