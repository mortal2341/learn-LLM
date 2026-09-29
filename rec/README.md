**用 PyTorch 实现双塔召回模型，**
在 MovieLens-100k 上完成「数据预处理 → 模型训练 → 候选召回 → 指标评估」的完整闭环。
用于补齐简历中「搜索召回」环节的项目缺口。

## 目录结构

```
soutui/
├── data.py        # 数据层：下载/缓存、隐式反馈构建、留一法划分、负采样
├── model.py       # 双塔召回模型（用户塔/物品塔，点积 + 物品偏置打分）
├── train.py       # 训练：BPR pairwise 排序损失 + 均匀负采样
├── evaluate.py    # 召回生成（全库打分 top-K）+ Recall@K/HitRate@K/NDCG@K
├── main.py        # 入口：串联全流程，命令行参数
├── requirements.txt
├── data/ml-100k/  # 数据集缓存（首次运行自动下载）
└── output/        # 运行产物：recall_candidates.csv / two_tower_recall.pt
```

## 环境要求

- `pytorch_clean`（本机：`D:\Anaconda\envs\pytorch_clean`，torch 2.13 + CUDA）
- 依赖仅 `torch / numpy / pandas`（环境中已具备）

## 运行方式

```bash
# 在 pytorch_clean 环境中，进入项目目录后一条命令跑通：
python main.py

# 常用可选参数（均有默认值）：
python main.py --epochs 30 --embed_dim 64 --num_neg 4 --topk 50 \
    --min_rating 4.0 --device cuda
```

首次运行会自动下载 MovieLens-100k 到 `data/ml-100k/`（约 5MB），之后复用缓存。

## 方法说明

| 环节  | 做法                                                       |
| --- | -------------------------------------------------------- |
| 数据  | MovieLens-100k（943 用户 × 1682 电影 × 10 万评分）；评分 ≥ 4 记为隐式正反馈 |
| 划分  | 按时间对每个用户「留一法」：最近一次交互进测试集，其余进训练集                          |
| 模型  | 双塔：用户 id / 物品 id 各自 Embedding + MLP 塔，打分 = 点积 + 物品偏置     |
| 训练  | BPR 损失 `-log σ(pos−neg)`，每个正样本均匀负采样 `num_neg` 个          |
| 召回  | 全库物品打分（屏蔽训练集已交互物品），取 top-K 作为候选召回列表                      |
| 评估  | Recall@K / HitRate@K / NDCG@K（留一法下 Recall@K = HitRate@K） |

## 参考结果（默认参数，CPU/GPU 均约 20 秒）

```
users=942  items=1447  train=54433  test=942  density=3.99%
Recall@10 = 0.0987   HitRate@10 = 0.0987   NDCG@10 = 0.0493
Recall@20 = 0.1592   HitRate@20 = 0.1592   NDCG@20 = 0.0646
Recall@50 = 0.2898   HitRate@50 = 0.2898   NDCG@50 = 0.0904
```

## 输出产物

- `output/recall_candidates.csv`：每个用户的 top-K 候选召回列表
  （列：`user_id, rank, item_id, score`，已映射回原始 id）
- `output/two_tower_recall.pt`：训练好的模型权重


