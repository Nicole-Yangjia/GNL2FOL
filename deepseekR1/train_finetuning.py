import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import transformers

warnings.filterwarnings("ignore")


class CoTGenerationTrainer:
    """思维链生成训练器"""

    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.tokenizer = None
        self.model = None
        self.max_length = 2048

    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print(f"加载数据: {self.data_path}")
        df = pd.read_excel(self.data_path)

        required_columns = ['natural_chinese_description', 'cot']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据集中缺少必要的列: {missing_columns}")

        if 'type' in df.columns:
            print(f"数据类型分布:")
            print(df['type'].value_counts())
        else:
            print("注意：数据中没有type列")

        print(f"数据形状: {df.shape}")

        return df

    def prepare_training_data(self):
        """准备训练数据"""
        df = self.load_and_preprocess_data()

        training_samples = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="准备训练数据"):
            try:
                sample = self._create_standard_example(row)
                training_samples.append(sample)

            except Exception as e:
                print(f"处理行 {idx} 时出错: {e}")
                continue

        print(f"生成的训练样本总数: {len(training_samples)} (每个原始样本生成1个示例)")
        return training_samples

    def _create_standard_example(self, row):
        """创建标准格式示例"""
        description = row['natural_chinese_description']
        cot = row['cot']

        game_type = row.get('type', 'unknown') if 'type' in row else 'unknown'

        input_text = f"""请为以下博弈游戏生成详细的思维链分析：

重要说明：
- 在分析过程中，请使用谓词 leq(i, x1,...,xn, y1,...,yn) 来表示参与者i对于两个结果(x1,...,xn)和(y1,...,yn)的偏好关系
- 其中i是参与者编号，n是参与者总数
- (x1,...,xn)和(y1,...,yn)表示策略组合，其中xk和yk分别表示第k个参与者的策略选择
- leq(i, x1,...,xn, y1,...,yn) 表示对于参与者i来说，他对结果(y1,...,yn)的偏好不低于(x1,...,xn)
- 也就是说，如果(x1,...,xn)的收益小于等于（包含严格小于）(y1,...,yn)的收益，那么 leq(i, x1,...,xn, y1,...,yn) 成立

游戏描述：
{description}

请按照以下6个步骤进行分析：

1. 识别参与者与策略
2. 分析偏好关系
3. 构建具体偏好排序
4. 将偏好转换为leq关系（请使用上述定义的leq谓词）
5. 抽象为一阶逻辑表达式（基于leq谓词）
6. 验证逻辑表达式与自然语言描述的一致性

请确保分析过程严谨、完整，并且在步骤4和5中明确使用 leq(i, x1,...,xn, y1,...,yn) 谓词。"""

        output_text = cot

        text = f"User\n{input_text}Assistant\n{output_text}"

        return {
            'input': input_text,
            'output': output_text,
            'text': text,
            'metadata': {
                'type': game_type,
                'source': 'standard'
            }
        }

    def initialize_model(self):
        """初始化DeepSeek-R1模型"""
        print("初始化DeepSeek-R1-Distill-LLaMA-8B模型...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"特殊token: pad_token={self.tokenizer.pad_token}, eos_token={self.tokenizer.eos_token}")
        print(f"特殊token ID: pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}")

        # 4bit量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False,
            trust_remote_code=True
        )

        # 准备模型用于k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)

        # 优化LoRA配置
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.model.gradient_checkpointing_enable()

        return self.model, self.tokenizer

    def tokenize_function(self, examples):
        """分词函数"""
        texts = examples['text']

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding='longest',
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokenized["labels"] = tokenized["input_ids"].clone()

        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].to(self.model.device)

        return tokenized

    def train(self, training_samples, output_dir="./cot_generation_model"):
        """训练模型 - 优化训练参数"""
        print("开始训练...")

        if training_samples and 'type' in training_samples[0]['metadata']:
            train_samples, val_samples = train_test_split(
                training_samples,
                test_size=0.15,
                random_state=42,
                stratify=[s['metadata']['type'] for s in training_samples]
            )
        else:
            train_samples, val_samples = train_test_split(
                training_samples,
                test_size=0.15,
                random_state=42
            )

        print(f"训练集: {len(train_samples)} 条")
        print(f"验证集: {len(val_samples)} 条")

        # 转换为Dataset
        train_dataset = Dataset.from_list(train_samples)
        val_dataset = Dataset.from_list(val_samples)

        # 分词
        print("\n分词处理...")
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="分词训练集"
        )

        tokenized_val = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="分词验证集"
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # 优化训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            weight_decay=0.001,
            dataloader_drop_last=True,
            report_to="none",
            optim="adamw_8bit",
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            remove_unused_columns=False,
            group_by_length=False,
            ddp_find_unused_parameters=False,
        )

        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01
                )
            ] if EarlyStoppingCallback else []
        )

        # 开始训练
        print("\n开始训练...")
        try:
            train_result = trainer.train()
            print("训练完成！")

            metrics = train_result.metrics
            print(f"训练指标: {metrics}")

            with open(f"{output_dir}/training_metrics.json", "w") as f:
                import json
                json.dump(metrics, f, indent=2)

        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"模型已保存到: {output_dir}")

        return trainer

    def generate_response(self, input_text, max_new_tokens=500):
        """生成响应"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未初始化")

        prompt = f"User\n{input_text}Assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_output[len(prompt):]

        return generated



def main():
    """主函数"""
    MODEL_PATH = "/root/sj-tmp/llama_finetuning"
    DATA_PATH = "game_data_two3.xlsx"
    OUTPUT_DIR = "/root/sj-tmp/llama_finetuning/deepseek_r1_cot_model_0108"

    # 1. 初始化训练器
    trainer = CoTGenerationTrainer(MODEL_PATH, DATA_PATH)

    # 2. 准备训练数据
    print("\n1. 准备训练数据...")
    training_samples = trainer.prepare_training_data()

    # 3. 初始化模型
    print("\n2. 初始化模型...")
    model, tokenizer = trainer.initialize_model()

    # 4. 训练
    print("\n3. 开始训练...")
    try:
        trainer_obj = trainer.train(training_samples, OUTPUT_DIR)
        if trainer_obj is None:
            print("训练失败，退出程序")
            return
        print("训练成功完成！")

        # 5. 简单的测试示例
        print("\n4. 简单测试示例...")
        if training_samples:
            test_sample = training_samples[0]
            print(f"输入示例: {test_sample['input'][:200]}...")

            # 使用新的生成方法
            generated = trainer.generate_response(test_sample['input'])
            print(f"生成结果: {generated[:800]}...")

            print("\n期望输出片段:")
            print(test_sample['output'][:500] + "...")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
