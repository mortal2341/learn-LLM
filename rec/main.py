"""最小可行推荐系统召回（双塔 + BPR）。

完整流程：
    下载/加载 MovieLens-100k -> 构建隐式反馈 -> 留一法划分
    -> 双塔 BPR 训练 -> 全库打分生成 top-K 候选召回列表
    -> Recall@K / HitRate@K / NDCG@K 评估。

运行（在 pytorch_clean 环境中）：
    python main.py
常用参数见 parse_args()。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from data import prepare_dataset
from evaluate import evaluate_recall, generate_candidates
from model import TwoTowerRecall
from train import train_model


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="双塔召回 MVP（MovieLens-100k）")
    p.add_argument("--data_dir", type=Path, default=here / "data" / "ml-100k", help="数据缓存目录")
    p.add_argument("--out_dir", type=Path, default=here / "output", help="候选列表 / 模型输出目录")
    p.add_argument("--min_rating", type=float, default=4.0, help="隐式反馈正样本评分阈值")
    p.add_argument("--embed_dim", type=int, default=64, help="召回向量维度")
    p.add_argument("--hidden_dim", type=int, default=64, help="塔内 MLP 隐层维度")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_neg", type=int, default=4, help="每个正样本的负采样个数")
    p.add_argument("--topk", type=int, default=50, help="生成的候选召回列表长度")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"[1/4] 加载并预处理数据 ... (device={device})")
    bundle = prepare_dataset(args.data_dir, args.min_rating)
    n_users, n_items = bundle["n_users"], bundle["n_items"]
    n_train, n_test = len(bundle["train_u"]), len(bundle["test_u"])
    density = n_train / (n_users * n_items)
    print(f"      users={n_users}  items={n_items}  train={n_train}  test={n_test}  density={density:.4%}")

    print("[2/4] 构建双塔模型并训练 (BPR) ...")
    model = TwoTowerRecall(n_users, n_items, args.embed_dim, args.hidden_dim)
    model = train_model(
        model, bundle, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_neg=args.num_neg, seed=args.seed,
    )

    print(f"[3/4] 全库打分，生成 top-{args.topk} 候选召回列表 ...")
    topk_items, topk_scores = generate_candidates(model, bundle, device, topk=args.topk)

    print("[4/4] 评估召回效果 ...")
    metrics = evaluate_recall(topk_items, bundle["test_u"], bundle["test_i"], ks=(10, 20, 50))
    for k, m in metrics.items():
        print(f"      Recall@{k:<3d} = {m['recall']:.4f}   "
              f"HitRate@{k:<3d} = {m['hitrate']:.4f}   NDCG@{k:<3d} = {m['ndcg']:.4f}")

    # 保存候选召回列表（映射回原始 user/item id）
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = args.out_dir / "recall_candidates.csv"
    u_uniq, i_uniq = bundle["u_uniq"], bundle["i_uniq"]
    rows = [
        (int(u_uniq[u]), rank + 1, int(i_uniq[topk_items[u, rank]]), float(topk_scores[u, rank]))
        for u in range(n_users)
        for rank in range(args.topk)
    ]
    pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "score"]).to_csv(cand_path, index=False)
    torch.save(model.state_dict(), args.out_dir / "two_tower_recall.pt")
    print(f"\n候选召回列表已保存: {cand_path}")
    print(f"模型权重已保存:     {args.out_dir / 'two_tower_recall.pt'}")

    # 打印一个用户的召回示例，便于直观检查
    demo_u = int(bundle["test_u"][0])
    demo_items = [int(i_uniq[i]) for i in topk_items[demo_u, :10]]
    print(f"\n示例：用户 {int(u_uniq[demo_u])} 的 top-10 候选 item = {demo_items}")
    print(f"      该用户测试集真实 item = {int(i_uniq[bundle['test_i'][0]])}")


if __name__ == "__main__":
    main()
