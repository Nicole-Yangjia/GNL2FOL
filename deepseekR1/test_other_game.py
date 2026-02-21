import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import warnings

warnings.filterwarnings("ignore")


class CoTModelTester:

    def __init__(self, base_model_path, fine_tuned_model_path):
        self.base_model_path = base_model_path
        self.fine_tuned_model_path = fine_tuned_model_path
        self.base_tokenizer = None
        self.base_model = None
        self.fine_tuned_tokenizer = None
        self.fine_tuned_model = None

    def load_models(self):
        print("加载基础模型...")

        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                use_cache=True,
                trust_remote_code=True
            )
            print("基础模型加载成功")
        except Exception as e:
            print(f"基础模型加载失败: {e}")
            return False

        print("加载微调模型...")
        try:
            self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_model_path)
            if self.fine_tuned_tokenizer.pad_token is None:
                self.fine_tuned_tokenizer.pad_token = self.fine_tuned_tokenizer.eos_token

            self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                self.fine_tuned_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                use_cache=True,
                trust_remote_code=True
            )
            print("微调模型加载成功（直接加载完整模型）")
        except:
            try:
                print("尝试以Peft格式加载微调模型...")
                self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
                if self.fine_tuned_tokenizer.pad_token is None:
                    self.fine_tuned_tokenizer.pad_token = self.fine_tuned_tokenizer.eos_token

                base_for_peft = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_cache=True,
                    trust_remote_code=True
                )

                self.fine_tuned_model = PeftModel.from_pretrained(
                    base_for_peft,
                    self.fine_tuned_model_path,
                    device_map="auto"
                )

                self.fine_tuned_model = self.fine_tuned_model.merge_and_unload()
                print("微调模型加载成功（Peft格式）")

            except Exception as e:
                print(f"微调模型加载失败: {e}")
                print("尝试加载配置信息...")
                try:
                    import os
                    config_path = os.path.join(self.fine_tuned_model_path, "adapter_config.json")
                    if os.path.exists(config_path):
                        print(f"找到Peft配置文件: {config_path}")
                        peft_config = PeftConfig.from_pretrained(self.fine_tuned_model_path)
                        self.fine_tuned_model = PeftModel.from_pretrained(
                            self.base_model,
                            self.fine_tuned_model_path,
                            device_map="auto"
                        )
                        self.fine_tuned_tokenizer = self.base_tokenizer
                        print("微调模型加载成功（使用基础模型+Peft适配器）")
                    else:
                        self.fine_tuned_model = None
                except Exception as e2:
                    print(f"配置文件加载失败: {e2}")
                    self.fine_tuned_model = None

        return self.fine_tuned_model is not None

    def create_translation_prompt(self, text, source_lang, target_lang):
        if source_lang == "en" and target_lang == "zh":
            return f"""请将以下英文游戏描述翻译成中文：

英文原文：
{text}

中文翻译："""
        elif source_lang == "zh" and target_lang == "en":
            return f"""Please translate the following Chinese text to English:

Chinese text:
{text}

English translation:"""
        else:
            return f"""请将以下文本从{source_lang}翻译成{target_lang}：

原文：
{text}

翻译："""

    def create_cot_prompt(self, game_description_chinese):
        prompt = f"""请为以下博弈游戏生成详细的思维链分析：

重要说明：
- 在分析过程中，请使用谓词 leq(i, x1,...,xn, y1,...,yn) 来表示参与者i对于两个结果(x1,...,xn)和(y1,...,yn)的偏好关系
- 其中i是参与者编号，n是参与者总数
- (x1,...,xn)和(y1,...,yn)表示策略组合，其中xk和yk分别表示第k个参与者的策略选择
- leq(i, x1,...,xn, y1,...,yn) 表示对于参与者i来说，他对结果(y1,...,yn)的偏好不低于(x1,...,xn)
- 也就是说，如果(x1,...,xn)的收益小于等于（包含严格小于）(y1,...,yn)的收益，那么 leq(i, x1,...,xn, y1,...,yn) 成立

游戏描述：
{game_description_chinese}

请按照以下6个步骤进行分析：

1. 识别参与者与策略
2. 分析偏好关系
3. 构建具体偏好排序
4. 将偏好转换为leq关系（请使用上述定义的leq谓词）
5. 抽象为一阶逻辑表达式（基于leq谓词）
6. 验证逻辑表达式与自然语言描述的一致性

请确保分析过程严谨、完整，并且在步骤4和5中明确使用 leq(i, x1,...,xn, y1,...,yn) 谓词。"""

        return prompt

    def format_for_deepseek(self, prompt, is_translation=False):
        if is_translation:
            return f"User\n{prompt}Assistant\n"
        else:
            return f"User\n{prompt}Assistant\n"

    def generate_response(self, model, tokenizer, prompt, max_new_tokens=2500):
        formatted_prompt = self.format_for_deepseek(prompt)

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True,
                num_return_sequences=1,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Assistant" in full_output:
            assistant_part = full_output.split("Assistant")[-1]
            if assistant_part.startswith("\n"):
                assistant_part = assistant_part[1:]
            elif assistant_part.startswith(":"):
                assistant_part = assistant_part[1:]

            assistant_part = assistant_part.strip()
            return assistant_part
        else:
            generated = full_output[len(formatted_prompt):].strip()
            return generated

    def process_english_input(self, model, tokenizer, game_description_english):
        print("步骤1: 将英文游戏描述翻译为中文...")

        translation_prompt_zh = self.create_translation_prompt(
            game_description_english, "en", "zh"
        )
        chinese_translation = self.generate_response(
            model, tokenizer, translation_prompt_zh, max_new_tokens=800
        )

        print(f"中文翻译结果: {chinese_translation}")

        print("\n步骤2: 进行中文思维链分析...")
        cot_prompt = self.create_cot_prompt(chinese_translation)
        chinese_cot = self.generate_response(
            model, tokenizer, cot_prompt, max_new_tokens=1600
        )

        print("\n步骤3: 将中文思维链分析翻译为英文...")
        translation_prompt_en = self.create_translation_prompt(
            chinese_cot, "zh", "en"
        )
        english_cot = self.generate_response(
            model, tokenizer, translation_prompt_en, max_new_tokens=2000
        )

        return {
            "chinese_translation": chinese_translation,
            "chinese_cot": chinese_cot,
            "english_cot": english_cot
        }

    def process_chinese_input(self, model, tokenizer, game_description_chinese):
        """处理中文输入：直接进行中文分析"""
        print("进行中文思维链分析...")
        cot_prompt = self.create_cot_prompt(game_description_chinese)
        chinese_cot = self.generate_response(
            model, tokenizer, cot_prompt, max_new_tokens=2000
        )

        return {
            "chinese_cot": chinese_cot
        }

    def test_all_cases(self):
        test_cases = [

            ("囚徒困境（三人）",
             "三人团队项目。若都努力，各得10；若两人努力一人偷懒，偷懒者得12，努力者各得6；若一人努力两人偷懒，努力者得3，偷懒者各得5；若都偷懒，各得0。对每人而言，偷懒总是更优，但集体都偷懒（0）远差于都努力（10）。",
             True)
        ]

        for case_name, game_desc, is_chinese in test_cases:
            print("\n" + "=" * 100)
            print(f"测试案例: {case_name}")
            print(f"输入语言: {'中文' if is_chinese else '英文'}")
            print("=" * 100)

            print(f"\n原始游戏描述:\n{game_desc}\n")

            print("-" * 50)
            print("基础模型分析:")
            print("-" * 50)

            if is_chinese:
                try:
                    base_result = self.process_chinese_input(
                        self.base_model, self.base_tokenizer, game_desc
                    )
                    print(f"\n中文思维链分析:\n{base_result['chinese_cot']}")
                except Exception as e:
                    print(f"基础模型处理中文输入时出错: {e}")
            else:
                try:
                    base_result = self.process_english_input(
                        self.base_model, self.base_tokenizer, game_desc
                    )
                    print(f"\n中文翻译结果:\n{base_result['chinese_translation']}")
                    print(f"\n中文思维链分析:\n{base_result['chinese_cot']}")
                    print(f"\n英文思维链分析:\n{base_result['english_cot']}")
                except Exception as e:
                    print(f"基础模型处理英文输入时出错: {e}")

            # 测试微调模型
            if self.fine_tuned_model:
                print("\n" + "-" * 50)
                print("微调模型分析:")
                print("-" * 50)

                if is_chinese:
                    try:
                        fine_tuned_result = self.process_chinese_input(
                            self.fine_tuned_model, self.fine_tuned_tokenizer, game_desc
                        )
                        print(f"\n中文思维链分析:\n{fine_tuned_result['chinese_cot']}")
                    except Exception as e:
                        print(f"微调模型处理中文输入时出错: {e}")
                else:
                    try:
                        fine_tuned_result = self.process_english_input(
                            self.fine_tuned_model, self.fine_tuned_tokenizer, game_desc
                        )
                        print(f"\n中文翻译结果:\n{fine_tuned_result['chinese_translation']}")
                        print(f"\n中文思维链分析:\n{fine_tuned_result['chinese_cot']}")
                        print(f"\n英文思维链分析:\n{fine_tuned_result['english_cot']}")
                    except Exception as e:
                        print(f"微调模型处理英文输入时出错: {e}")
            else:
                print("\n微调模型不可用，跳过微调模型输出")

            print("\n" + "=" * 100 + "\n\n")


def main():
    """主函数"""
    print("思维链生成模型对比测试 - DeepSeek-R1-Distill-LLaMA-8B")
    print("=" * 80)
    print("说明：对于英文输入，模型将执行以下步骤：")
    print("1. 将英文游戏描述翻译为中文")
    print("2. 进行中文思维链分析")
    print("3. 将分析结果翻译为英文")
    print("=" * 80)
    BASE_MODEL_PATH = "/root/sj-tmp/llama_finetuning"
    FINE_TUNED_MODEL_PATH = "/root/sj-tmp/llama_finetuning/deepseek_r1_cot_model_0108"  # 微调后的模型路径

    # 初始化测试器
    tester = CoTModelTester(BASE_MODEL_PATH, FINE_TUNED_MODEL_PATH)

    # 加载模型
    print("正在加载模型...")
    success = tester.load_models()

    if not success:
        print("微调模型加载失败，尝试其他方式...")
        # 尝试直接加载完整模型
        try:
            tester.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
            if tester.fine_tuned_tokenizer.pad_token is None:
                tester.fine_tuned_tokenizer.pad_token = tester.fine_tuned_tokenizer.eos_token

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            tester.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                FINE_TUNED_MODEL_PATH,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                use_cache=True,
                trust_remote_code=True
            )
            print("微调模型加载成功（直接加载完整模型）")
        except Exception as e:
            print(f"微调模型加载失败: {e}")
            print("将只测试基础模型")

    print("\n开始测试...")
    tester.test_all_cases()

    print("测试完成！")


if __name__ == "__main__":
    main()
