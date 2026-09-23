from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

student_name = "LLM-Research/Llama-3.2-1B-Instruct"
teacher_name = "Qwen/Qwen2.5-1.5B-Instruct"

# 加载学生模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(student_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载教师模型和学生模型
model = AutoModelForCausalLM.from_pretrained(student_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name)


# 加载数据集
train_dataset = load_dataset(
    "HuggingFaceTB/OpenR1-Math-220k-default-verified",
    "all",
    split="train[:1024]",
)

training_args = GOLDConfig(
    output_dir="saved/gold-model",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    teacher_tokenizer_name_or_path=teacher_name,
    save_strategy="epoch",
    use_uld_loss=True,
    uld_use_hybrid_loss=True,
)

trainer = GOLDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
