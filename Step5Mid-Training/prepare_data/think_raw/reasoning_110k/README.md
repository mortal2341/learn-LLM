---
license: apache-2.0
language:
- zh
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- text2text-generation
- question-answering
---

# 中文基于满血DeepSeek-R1蒸馏数据集（Chinese-Data-Distill-From-R1）

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k">ModelScope</a> &nbsp&nbsp | &nbsp&nbsp🚀 <a href="https://github.com/YunwenTechnology/Chinese-Data-Distill-From-R1">Github</a> &nbsp&nbsp | &nbsp&nbsp📑 <a href="https://zhuanlan.zhihu.com/p/24430839729">Blog</a>
</p>

注意：提供了直接SFT使用的版本，[点击下载](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT)。将数据中的思考和答案整合成output字段，大部分SFT代码框架均可直接直接加载训练。

本数据集为中文开源蒸馏满血R1的数据集，数据集中不仅包含math数据，还包括大量的通用类型数据，总数量为110K。

为什么开源这个数据？

R1的效果十分强大，并且基于R1蒸馏数据SFT的小模型也展现出了强大的效果，但检索发现，大部分开源的R1蒸馏数据集均为英文数据集。 同时，R1的报告中展示，蒸馏模型中同时也使用了部分通用场景数据集。
为了帮助大家更好地复现R1蒸馏模型的效果，特此开源中文数据集。

该中文数据集中的数据分布如下：

- Math：共计36568个样本，
- Exam：共计2432个样本，
- STEM：共计12648个样本，
- General：共计58352，包含弱智吧、逻辑推理、小红书、知乎、Chat等。

字段说明：
- input: 输入
- reasoning_content: 思考
- content: 输出
- repo_name: 数据源
- score: 模型打分结果

## 数据集蒸馏细节

数据的prompt源来自：

- [Haijian/Advanced-Math](https://modelscope.cn/datasets/Haijian/Advanced-Math)
- [gavinluo/applied_math](https://modelscope.cn/datasets/gavinluo/applied_math)
- [meta-math/GSM8K_zh](https://huggingface.co/datasets/meta-math/GSM8K_zh)
- [EduChat-Math](https://github.com/ECNU-ICALK/EduChat-Math)
- [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA)
- [m-a-p/neo_sft_phase2](https://huggingface.co/datasets/m-a-p/neo_sft_phase2)
- [hfl/stem_zh_instruction](https://huggingface.co/datasets/hfl/stem_zh_instruction)

同时为了方便大家溯源，在每条数据的repo_name字段中都加入的原始数据源repo。

在蒸馏过程中，按照[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)官方提供的细节，进行数据蒸馏。

- 不增加额外的系统提示词
- 设置temperature为0.6
- 如果为数学类型数据，则增加提示词，“请一步步推理，并把最终答案放到 \boxed{}。”
- 防止跳出思维模式，强制在每个输出的开头增加"\n"，再开始生成数据

由于个人资源有限，所有数据的蒸馏均调用[无问芯穹](https://cloud.infini-ai.com/genstudio?source=knlpdis)的企业版满血R1 API生成，在此由衷的感谢无问芯穹。

任务期间，保持稳定地运行300并发，支持64k上下文，32k输出长度，持续运行近12个小时，性能始终保持一致，数据可用性100%。测试时首token延时基本在500ms以下，推理速度最快25 tokens/s（需根据实际运行任务进行测试实际稳定性指标比较合理）。

## 数据打分细节

数据生成结果进行了二次校验，并保留了评价分数。

针对Math和Exam数据，先利用[Math-Verify](https://github.com/huggingface/Math-Verify)进行校对，无法规则抽取结果的数据，再利用[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)模型进行打分，正确为10分，错误为0分。

针对其他数据，直接利用[Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)模型从无害性、有用性、正确性/完整性三个角度进行打分，分值范围为0-10分。

本数据集保留了最后打分结果，为后续的数据筛选提供帮助，但注意，所有打分均基于模型，因此评分可能并不准确，请斟酌使用。

数据的二次校验，使用8张A100 GPU 部署多节点Qwen72B模型进行推理打分，耗时接近24H，感谢我司云问科技提供的服务器支持。

## 局限性

由于数据是由蒸馏DeepSeek-R1生成的，未经严格验证，在事实性和其他方面还存在一些不足。因此，在使用此数据集时，请务必注意甄别。

本数据集不代表任何一方的立场、利益或想法，无关任何团体的任何类型的主张。因使用本数据集带来的任何损害、纠纷，本项目的开发者不承担任何责任。

## 引用

```text
@misc{Chinese-Data-Distill-From-R1,
  author = {Cong Liu, Zhong Wang, ShengYu Shen, Jialiang Peng, Xiaoli Zhang, ZhenDong Du, YaFang Wang},
  title = {The Chinese dataset distilled from DeepSeek-R1-671b},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k}},
}
```

## 联系作者

- email: cliu@iyunwen.com
- 知乎：[刘聪NLP](https://www.zhihu.com/people/LiuCongNLP)
- 公众号：[NLP工作站](images/image.png)