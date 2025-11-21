# -*- coding: utf-8 -*-
"""

硬件要求：支持 Flash Attention 2
依赖库：pip install torch transformers numpy scipy pot tqdm flash-attn
"""

import os
import re
import json
import math
import random
import argparse
import itertools
import heapq
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# 引入先进架构 (RoPE) 和 传统架构 (APE)
from transformers import (
    GPTNeoXConfig, GPTNeoXForCausalLM,
    GPT2Config, GPT2LMHeadModel,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
from scipy.spatial.distance import cdist
import ot  # Python Optimal Transport (POT)

# ==============================================================================
# 1.  (Experiment Matrix)
# ==============================================================================
# 严格控制参数量，保证 Scale 内部公平对比
# Formula: Params ≈ 12 * n_layer * n_embd^2

MODEL_CONFIGS = {
    # --- Scale 1: Small (~10M) - 验证 "脑容量不足" ---
    'small_deep': {'n_layer': 12, 'n_embd': 192, 'n_head': 4},
    'small_wide': {'n_layer': 3, 'n_embd': 384, 'n_head': 4},

    # --- Scale 2: Medium (~85M) - 验证 "涌现临界点" ---
    'medium_deep': {'n_layer': 24, 'n_embd': 512, 'n_head': 8},
    'medium_wide': {'n_layer': 6, 'n_embd': 1024, 'n_head': 16},
    'medium_balanced': {'n_layer': 12, 'n_embd': 768, 'n_head': 12},

    # --- Scale 3: Large (~350M) - 验证 "性能上限/SoS复现" ---
    'large_deep': {'n_layer': 48, 'n_embd': 768, 'n_head': 12},
    'large_wide': {'n_layer': 12, 'n_embd': 1536, 'n_head': 24},
}


# ==============================================================================
# 2. 核心搜索逻辑 (Core Search Logic) - 无简化
# ==============================================================================

class CountdownNode:
    __slots__ = ['idx', 'parent', 'nums', 'operations', 'heuristic']

    def __init__(self, idx, parent, nums, operations, heuristic):
        self.idx = idx
        self.parent = parent
        self.nums = nums
        self.operations = operations
        self.heuristic = heuristic

    def __lt__(self, other):
        return self.heuristic < other.heuristic


def combine_nums(a, b):
    """生成所有合法的算术操作（加、乘、非负减、整除）"""
    a, b = int(a), int(b)
    possible = [[a + b, f"{a}+{b}={a + b}"], [a * b, f"{a}*{b}={a * b}"]]

    if a <= b:
        if b - a >= 0: possible.append([b - a, f"{b}-{a}={b - a}"])
    else:
        if a - b >= 0: possible.append([a - b, f"{a}-{b}={a - b}"])

    if a <= b:
        if a != 0 and b % a == 0: possible.append([b // a, f"{b}/{a}={b // a}"])
    else:
        if b != 0 and a % b == 0: possible.append([a // b, f"{a}/{b}={a // b}"])

    return possible


def heuristic_fn(nums, target, type='sum'):
    """
    启发式函数：
    sum: 引导模型通过加减逼近
    multiply: 引导模型通过因数分解逼近
    """
    if not nums: return abs(target)

    if type == 'sum':
        return abs(target - sum(nums))
    elif type == 'multiply':
        if target == 0: return 1000
        factors = {i for i in range(1, int(abs(target) ** 0.5) + 1) if target % i == 0}
        factors.update({abs(target) // f for f in factors})
        return min(abs(sum(nums) - f) for f in factors)

    return abs(target - sum(nums))


def bfs_searcher(target, nums, beam_size=3, h_type='sum'):
    """
    BFS：结构化、短路径
    """
    search_trace = []
    open_set = []
    initial_h = heuristic_fn(nums, target, h_type)
    heapq.heappush(open_set, (initial_h, CountdownNode("0", None, nums, [], initial_h)))
    visited = set()

    max_nodes = 3000  # 防止死循环的安全阀，但在有效范围内不做简化
    nodes_gen = 0

    while open_set and nodes_gen < max_nodes:
        current_batch = [heapq.heappop(open_set) for _ in range(min(beam_size, len(open_set)))]

        for _, curr in current_batch:
            state = tuple(sorted(curr.nums))
            if state in visited: continue
            visited.add(state)

            search_trace.append(f"Current State: {target}:{curr.nums} Operations: {curr.operations}")

            if len(curr.nums) == 1:
                if curr.nums[0] == target:
                    return "\n".join(search_trace) + "\nGoal Reached"
                continue

            candidates = []
            for i, j in itertools.combinations(range(len(curr.nums)), 2):
                rem = [curr.nums[k] for k in range(len(curr.nums)) if k != i and k != j]
                for res, op in combine_nums(curr.nums[i], curr.nums[j]):
                    new_nums = rem + [res]
                    candidates.append((heuristic_fn(new_nums, target, h_type), op, new_nums))

            candidates.sort()
            for idx, (h, op, nn) in enumerate(candidates[:beam_size]):
                node_id = f"{curr.idx},{idx}"
                search_trace.append(f"Exploring: {op} Result: {nn}")

                if len(nn) == 1 and nn[0] == target:
                    return "\n".join(search_trace) + "\nGoal Reached"

                search_trace.append(f"Gen Node #{node_id}")
                heapq.heappush(open_set, (h, CountdownNode(node_id, curr, nn, curr.operations + [op], h)))
                nodes_gen += 1

    return "\n".join(search_trace) + "\nNo Solution"


def dfs_searcher(target, nums, threshold, h_type='sum'):
    """
    DFS：试错、长路径
    """
    search_trace = []
    stack = [CountdownNode("0", None, nums, [], heuristic_fn(nums, target, h_type))]
    visited = set()

    max_nodes = 3000
    nodes_gen = 0

    while stack and nodes_gen < max_nodes:
        curr = stack.pop()
        state = tuple(sorted(curr.nums))
        if state in visited: continue
        visited.add(state)

        search_trace.append(f"Current State: {target}:{curr.nums} Operations: {curr.operations}")

        if len(curr.nums) == 1:
            if curr.nums[0] == target:
                return "\n".join(search_trace) + "\nGoal Reached"
            search_trace.append("Backtracking...")
            continue

        candidates = []
        for i, j in itertools.combinations(range(len(curr.nums)), 2):
            rem = [curr.nums[k] for k in range(len(curr.nums)) if k != i and k != j]
            for res, op in combine_nums(curr.nums[i], curr.nums[j]):
                new_nums = rem + [res]
                h = heuristic_fn(new_nums, target, h_type)
                if h <= threshold:  # Soft Pruning
                    candidates.append((h, op, new_nums))

        candidates.sort(key=lambda x: x[0], reverse=True)  # DFS Stack Order
        for idx, (h, op, nn) in enumerate(candidates):
            node_id = f"{curr.idx},{idx}"
            search_trace.append(f"Exploring: {op} Result: {nn}")

            if len(nn) == 1 and nn[0] == target:
                return "\n".join(search_trace) + "\nGoal Reached"

            search_trace.append(f"Gen Node #{node_id}")
            stack.append(CountdownNode(node_id, curr, nn, curr.operations + [op], h))
            nodes_gen += 1

    return "\n".join(search_trace) + "\nNo Solution"


# ==============================================================================
# 3. 分布度量 ( Metrics & Global Scaler)
# ==============================================================================

def extract_symbolic_features(trajectory_text):
    """提取 9 个关键符号特征"""
    features = np.zeros(9, dtype=np.float32)
    # 1. 成功标志
    features[0] = 1.0 if "Goal Reached" in trajectory_text else 0.0
    # 2. 搜索空间大小
    all_nodes = re.findall(r"Generated Node #([\d,]+)", trajectory_text)  # Robust Regex
    features[1] = len(all_nodes)
    # 3. 最大深度
    if all_nodes:
        depths = [s.count(',') for s in all_nodes]
        features[2] = max(depths)
    # 4. 失败次数
    features[3] = len(re.findall(r"unequal: No Solution", trajectory_text))
    # 5. 回溯/移动步数
    features[4] = len(re.findall(r"Moving to Node", trajectory_text))
    # 6-9. 算术密度
    features[5] = trajectory_text.count('+')
    features[6] = trajectory_text.count('-')
    features[7] = trajectory_text.count('*')
    features[8] = trajectory_text.count('/')
    return features


class GlobalScaler:
    """全局特征缩放器，确保不同实验批次间的距离可比性"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, vectors):
        self.mean = np.mean(vectors, axis=0)
        self.std = np.std(vectors, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, vectors):
        if self.mean is None: raise ValueError("Scaler not fitted!")
        return (vectors - self.mean) / self.std


def calculate_wasserstein_rigorous(texts_a, texts_b, scaler, sample_size=2000):
    """计算精确 Wasserstein 距离 (EMD)，不使用 Sinkhorn 近似"""
    if len(texts_a) == 0 or len(texts_b) == 0: return 0.0

    n = min(len(texts_a), sample_size)
    m = min(len(texts_b), sample_size)

    t1 = np.random.choice(texts_a, n, replace=False)
    t2 = np.random.choice(texts_b, m, replace=False)

    v1 = np.array([extract_symbolic_features(t) for t in t1])
    v2 = np.array([extract_symbolic_features(t) for t in t2])

    # 标准化
    v1_norm = scaler.transform(v1)
    v2_norm = scaler.transform(v2)

    # Cost Matrix & EMD
    M = cdist(v1_norm, v2_norm, metric='euclidean')
    a, b = np.ones(n) / n, np.ones(m) / m
    return ot.emd2(a, b, M, numItermax=1000000)


# ==============================================================================
# 4. Data Pipeline with OOD & Metrics
# ==============================================================================

class SoSTokenizer:
    def __init__(self):
        self.special = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.keywords = ["Current", "State:", "Operations:", "Exploring:", "Result:",
                         "Gen", "Node", "#", "Goal", "Reached", "No", "Solution",
                         "Backtracking...", "None"]
        self.symbols = ["+", "-", "*", "/", "=", ":", ",", "[", "]", "'"]
        self.nums = [str(i) for i in range(1001)]
        self.vocab = self.special + self.keywords + self.symbols + self.nums
        self.t2i = {t: i for i, t in enumerate(self.vocab)}
        self.i2t = {i: t for i, t in enumerate(self.vocab)}
        self.pad_id, self.bos_id, self.eos_id = self.t2i["<pad>"], self.t2i["<bos>"], self.t2i["<eos>"]

    def tokenize(self, text):
        tokens = re.findall(r"[A-Za-z]+:?|\d+|[+\-*/=:,\[\]#']|<[^>]+>", text)
        return [self.t2i.get(t, self.t2i["<unk>"]) for t in tokens]


class SoSDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=1024):
        self.data = data_list;
        self.tokenizer = tokenizer;
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = [self.tokenizer.bos_id] + self.tokenizer.tokenize(self.data[idx]) + [self.tokenizer.eos_id]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [self.tokenizer.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def prepare_experiment_data(args):
    """
    实验数据准备：
    1. 生成训练数据 (ID) 并根据 ratio 混合。
    2. 拟合 Global Scaler。
    3. 测量 Wasserstein 距离。
    4. 生成测试数据 (OOD)。
    """
    print(f"\n=== Phase 1: Data Generation (Ratio: {args.mixing_ratio}) ===")

    # --- 1. 生成训练数据 (ID: 10-79) ---
    # 我们需要分别生成 Part A (BFS) 和 Part B (DFS/BFS2) 以便计算距离
    train_group_a = []
    train_group_b = []

    n_dfs = int(args.num_samples * args.mixing_ratio)
    n_bfs = args.num_samples - n_dfs

    id_range = (10, 79)

    # Group A: 始终是 BFS (Base Logic)
    rng_a = random.Random(42)
    # 注意：如果 ratio=1.0 (纯DFS)，这里 n_bfs=0，Group A 为空
    # 为了度量一致性，如果 ratio=1.0，我们将 Group A 设为 BFS(ref)，Group B 设为 DFS
    # 如果 ratio=0.0，Group A 为 BFS(seed1)，Group B 为 BFS(seed2)

    # 逻辑分支：
    if args.mixing_ratio == 0.0:
        # Mode: Pure BFS (Noise Baseline)
        print("Mode: Pure BFS (Measuring Noise)")
        half = args.num_samples // 2
        for _ in tqdm(range(half), desc="Gen BFS-A"):
            t = rng_a.randint(*id_range)
            n = [rng_a.randint(1, 25) for _ in range(4)]
            train_group_a.append(bfs_searcher(t, n, h_type='sum'))

        rng_b = random.Random(1337)
        for _ in tqdm(range(args.num_samples - half), desc="Gen BFS-B"):
            t = rng_b.randint(*id_range)
            n = [rng_b.randint(1, 25) for _ in range(4)]
            train_group_b.append(bfs_searcher(t, n, h_type='sum'))

    else:
        # Mode: Mixed or Pure DFS
        print(f"Mode: Conflict (BFS={n_bfs}, DFS={n_dfs})")

        # Group A: BFS (Context)
        if n_bfs > 0:
            for _ in tqdm(range(n_bfs), desc="Gen BFS"):
                t = rng_a.randint(*id_range)
                n = [rng_a.randint(1, 25) for _ in range(4)]
                train_group_a.append(bfs_searcher(t, n, h_type='sum'))
        else:
            # 如果是纯 DFS，为了计算相对于 BFS 的距离，我们需要生成一组临时的 BFS 作为参考系
            # 这组参考 BFS 不会进入训练集，仅用于度量
            print("  -> Generating Reference BFS for Metric Calculation...")
            for _ in range(2000):
                t = rng_a.randint(*id_range)
                n = [rng_a.randint(1, 25) for _ in range(4)]
                train_group_a.append(bfs_searcher(t, n, h_type='sum'))

        # Group B: DFS (Conflict)
        rng_b = random.Random(42)
        for _ in tqdm(range(n_dfs), desc="Gen DFS"):
            t = rng_b.randint(*id_range)
            n = [rng_b.randint(1, 25) for _ in range(4)]
            train_group_b.append(dfs_searcher(t, n, threshold=t + 30, h_type='multiply'))

    # --- 2. 拟合 Scaler & 计算距离 ---
    print("=== Phase 2: Metric Measurement ===")
    # 采样用于 Fit
    fit_sample = []
    if len(train_group_a) > 0: fit_sample.extend(np.random.choice(train_group_a, min(2000, len(train_group_a))))
    if len(train_group_b) > 0: fit_sample.extend(np.random.choice(train_group_b, min(2000, len(train_group_b))))

    scaler = GlobalScaler()
    scaler.fit(np.array([extract_symbolic_features(t) for t in fit_sample]))

    # 计算 W 距离
    w_dist = calculate_wasserstein_rigorous(train_group_a, train_group_b, scaler)
    print(f">>> Measured Wasserstein Distance: {w_dist:.6f}")

    # --- 3. 组装训练数据 ---
    # 如果是纯 DFS 模式，train_group_a 只是参考系，不能放进训练集
    if args.mixing_ratio == 1.0:
        full_train = train_group_b
    else:
        full_train = train_group_a + train_group_b

    random.shuffle(full_train)

    # --- 4. 生成 OOD 数据 (Targets 80-100) ---
    print("=== Phase 3: OOD Data Generation ===")
    ood_range = (80, 100)
    rng_ood = random.Random(999)
    ood_data = []
    # OOD Set 包含 50% BFS 和 50% DFS，以测试综合泛化能力
    for _ in range(2500):
        t = rng_ood.randint(*ood_range)
        n = [rng_ood.randint(1, 25) for _ in range(4)]
        ood_data.append(bfs_searcher(t, n, h_type='sum'))
    for _ in range(2500):
        t = rng_ood.randint(*ood_range)
        n = [rng_ood.randint(1, 25) for _ in range(4)]
        ood_data.append(dfs_searcher(t, n, threshold=t + 30, h_type='multiply'))

    return full_train, ood_data, w_dist


# ==============================================================================
# 5. 训练主入口 (Main)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, required=True, choices=['small', 'medium', 'large'])
    parser.add_argument("--structure", type=str, required=True, choices=['deep', 'wide', 'balanced'])
    parser.add_argument("--pe_type", type=str, default='rope', choices=['rope', 'ape'])
    parser.add_argument("--mixing_ratio", type=float, default=0.0, help="0.0=Pure BFS, 1.0=Pure DFS")

    parser.add_argument("--num_samples", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()

    # 全局随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 路径管理
    run_id = f"{args.scale}_{args.structure}_{args.pe_type}_ratio{args.mixing_ratio}_N{args.num_samples}"
    args.output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # Logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    print(f"Starting Experiment: {run_id}")

    # 1. 准备数据
    tokenizer = SoSTokenizer()
    train_raw, ood_raw, w_dist = prepare_experiment_data(args)

    train_ds = SoSDataset(train_raw, tokenizer, args.ctx_len)
    ood_ds = SoSDataset(ood_raw, tokenizer, args.ctx_len)

    # 拆分 ID Validation
    train_len = int(0.95 * len(train_ds))
    val_len = len(train_ds) - train_len
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_len, val_len])

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 初始化模型
    config_key = f"{args.scale}_{args.structure}"
    model_spec = MODEL_CONFIGS.get(config_key)
    if not model_spec:
        # Fallback logic for balanced/custom
        if 'balanced' in args.structure:
            # Simplified fallback logic for example, normally define in dict
            model_spec = {'n_layer': 12, 'n_embd': 768, 'n_head': 12}
        else:
            raise ValueError(f"Config {config_key} invalid.")

    device = torch.device("cuda")

    # 模型构建：RoPE vs APE
    if args.pe_type == 'rope':
        config = GPTNeoXConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=model_spec['n_embd'],
            num_hidden_layers=model_spec['n_layer'],
            num_attention_heads=model_spec['n_head'],
            intermediate_size=model_spec['n_embd'] * 4,
            max_position_embeddings=args.ctx_len,
            use_cache=False,
            attn_implementation="flash_attention_2"  # H100 Speedup!
        )
        model = GPTNeoXForCausalLM(config).to(device)
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer.vocab),
            n_embd=model_spec['n_embd'],
            n_layer=model_spec['n_layer'],
            n_head=model_spec['n_head'],
            n_positions=args.ctx_len,
            use_cache=False
        )
        model = GPT2LMHeadModel(config).to(device)

    # 3. 记录数据
    meta_info = {
        "args": vars(args),
        "w_dist": w_dist,
        "params": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tokenizer.vocab)
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    # 4. 训练循环
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps),
                                                num_training_steps=total_steps)
    scaler = GradScaler()

    metrics_history = []

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            # BF16 Context for H100
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 50 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- Evaluation ---
        model.eval()

        # ID Eval
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with autocast(dtype=torch.bfloat16):
                    val_loss += model(input_ids=batch, labels=batch).loss.item()
        avg_val = val_loss / len(val_loader)

        # OOD Eval
        ood_loss = 0
        with torch.no_grad():
            for batch in ood_loader:
                batch = batch.to(device)
                with autocast(dtype=torch.bfloat16):
                    ood_loss += model(input_ids=batch, labels=batch).loss.item()
        avg_ood = ood_loss / len(ood_loader)

        # Log & Print
        res_str = f"Epoch {epoch + 1} | ID Loss: {avg_val:.4f} | OOD Loss: {avg_ood:.4f} | W-Dist: {w_dist:.4f}"
        print(res_str)
        logging.info(res_str)

        metrics_history.append({
            "epoch": epoch + 1,
            "id_loss": avg_val,
            "ood_loss": avg_ood
        })

        # Checkpoint
        model.save_pretrained(os.path.join(args.output_dir, f"ckpt_ep{epoch + 1}"))

    # Final Save
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)


if __name__ == "__main__":
    main()