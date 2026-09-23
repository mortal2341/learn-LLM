# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: DPO chosen和reject概率上升和下降对比
# --------------------------------------------


from math import log, exp

def dpo_loss(yw, yw_ref, yl, yl_ref, beta):
    w_ratio = beta * log(yw / yw_ref)
    l_ratio = beta * log(yl / yl_ref)
    return -log(1 / (1 + exp(-(w_ratio - l_ratio))))

# DPO loss (基线loss)
print(dpo_loss(yw=0.5, yw_ref=0.5, yl=0.5, yl_ref=0.5, beta=0.1))
print("="*50)

# 策略yw增加，策略yl减少
print(dpo_loss(yw=0.8, yw_ref=0.5, yl=0.2, yl_ref=0.5, beta=0.1))
print(dpo_loss(yw=0.99, yw_ref=0.5, yl=0.01, yl_ref=0.5, beta=0.1))
print("="*50)

# 策略yw减少，策略yl减少
print(dpo_loss(yw=0.3, yw_ref=0.5, yl=0.2, yl_ref=0.5, beta=0.1))
print(dpo_loss(yw=0.2, yw_ref=0.5, yl=0.02, yl_ref=0.5, beta=0.1))
print("="*50)

# 策略yw增加，策略yl增加
print(dpo_loss(yw=0.8, yw_ref=0.5, yl=0.6, yl_ref=0.5, beta=0.1))
print("="*50)

# 策略yw不变，策略 yl减少
print(dpo_loss(yw=0.5, yw_ref=0.5, yl=0.2, yl_ref=0.5, beta=0.1))
print(dpo_loss(yw=0.5, yw_ref=0.5, yl=0.02, yl_ref=0.5, beta=0.1))
print(dpo_loss(yw=0.5, yw_ref=0.5, yl=0.002, yl_ref=0.5, beta=0.1))
print("="*50)

# 策略yw增加，策略yl不变
print(dpo_loss(yw=0.8, yw_ref=0.5, yl=0.5, yl_ref=0.5, beta=0.1))
print(dpo_loss(yw=0.99, yw_ref=0.5, yl=0.5, yl_ref=0.5, beta=0.1))
print("="*50)

# 策略yw减少，策略yl增加
print(dpo_loss(yw=0.2, yw_ref=0.5, yl=0.8, yl_ref=0.5, beta=0.1))
print("="*50)
