import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from sklearn.model_selection import train_test_split
import re
import json
from tqdm import tqdm
import warnings
from z3 import *

warnings.filterwarnings("ignore")

class FixedLogicParser:
    def __init__(self):
        self.Strat = DeclareSort('Strat')
        self.Player = DeclareSort('Player')
        self.c = Const('c', self.Strat)
        self.d = Const('d', self.Strat)
        self.player1 = Const('1', self.Player)
        self.player2 = Const('2', self.Player)
        self.leq = Function('leq', self.Player, self.Strat, self.Strat, self.Strat, self.Strat, BoolSort())
        self.var_map = {}
        self.player_var_map = {}

    def parse_fol(self, fol_str):
        fol_str = fol_str.strip()
        if fol_str.startswith('leq(') and '∧' in fol_str:
            return self._parse_conjunction_str(fol_str)
        self.current_pos = 0
        self.input_str = fol_str
        return self._parse_expression()

    def _parse_conjunction_str(self, fol_str):
        parts = fol_str.split('∧')
        constraints = []
        for part in parts:
            part = part.strip()
            if part:
                self.current_pos = 0
                self.input_str = part
                try:
                    constraint = self._parse_atomic()
                    constraints.append(constraint)
                except Exception as e:
                    print(f"解析合取项时出错: {part}, 错误: {e}")
                    continue
        if not constraints:
            raise ValueError("没有有效的合取项")
        if len(constraints) == 1:
            return constraints[0]
        return And(*constraints)

    def _next_token(self):
        while self.current_pos < len(self.input_str) and self.input_str[self.current_pos].isspace():
            self.current_pos += 1

    def _peek(self):
        self._next_token()
        if self.current_pos >= len(self.input_str):
            return ''
        return self.input_str[self.current_pos]

    def _parse_expression(self):
        return self._parse_implication()

    def _parse_implication(self):
        left = self._parse_disjunction()
        self._next_token()
        if self.current_pos < len(self.input_str):
            if self.input_str.startswith('⊃', self.current_pos):
                self.current_pos += 1
                right = self._parse_implication()
                return Implies(left, right)
            elif self.input_str.startswith('→', self.current_pos):
                self.current_pos += 1
                right = self._parse_implication()
                return Implies(left, right)
            elif self.input_str.startswith('->', self.current_pos):
                self.current_pos += 2
                right = self._parse_implication()
                return Implies(left, right)
            elif self.input_str.startswith('↔', self.current_pos):
                self.current_pos += 1
                right = self._parse_implication()
                return left == right
            elif self.input_str.startswith('==', self.current_pos):
                self.current_pos += 2
                right = self._parse_implication()
                return left == right
        return left

    def _parse_disjunction(self):
        left = self._parse_conjunction_expr()
        while True:
            self._next_token()
            if self.current_pos < len(self.input_str) and self.input_str[self.current_pos] == '∨':
                self.current_pos += 1
                right = self._parse_conjunction_expr()
                left = Or(left, right)
            else:
                break
        return left

    def _parse_conjunction_expr(self):
        left = self._parse_unary()
        while True:
            self._next_token()
            if self.current_pos < len(self.input_str) and self.input_str[self.current_pos] == '∧':
                self.current_pos += 1
                right = self._parse_unary()
                left = And(left, right)
            else:
                break
        return left

    def _parse_unary(self):
        self._next_token()
        if self.current_pos < len(self.input_str) and self.input_str[self.current_pos] == '¬':
            self.current_pos += 1
            expr = self._parse_unary()
            return Not(expr)
        elif self.current_pos < len(self.input_str) and self.input_str[self.current_pos] in '∀∃':
            return self._parse_quantifier()
        elif self.current_pos < len(self.input_str) and self.input_str[self.current_pos] == '(':
            self.current_pos += 1
            expr = self._parse_expression()
            self._next_token()
            if self.current_pos < len(self.input_str) and self.input_str[self.current_pos] == ')':
                self.current_pos += 1
            else:
                raise ValueError("Expected ')'")
            return expr
        else:
            return self._parse_atomic()

    def _parse_quantifier(self):
        quantifier = self.input_str[self.current_pos]
        self.current_pos += 1
        variables = []
        while True:
            self._next_token()
            var_name = ""
            while (self.current_pos < len(self.input_str) and
                   (self.input_str[self.current_pos].isalpha() or
                    self.input_str[self.current_pos].isdigit())):
                var_name += self.input_str[self.current_pos]
                self.current_pos += 1
            if not var_name:
                break
            if var_name in ['1', '2']:
                var = Const(var_name, self.Player)
                self.player_var_map[var_name] = var
            elif var_name.startswith('x') or var_name.startswith('y'):
                var = Const(var_name, self.Strat)
                self.var_map[var_name] = var
            else:
                var = Const(var_name, self.Strat)
                self.var_map[var_name] = var
            variables.append(var)
            self._next_token()
            if self.current_pos >= len(self.input_str) or self.input_str[self.current_pos] not in '∀∃':
                break
        body = self._parse_expression()
        if quantifier == '∀':
            return ForAll(variables, body)
        else:
            return Exists(variables, body)

    def _parse_atomic(self):
        if self.current_pos + 3 <= len(self.input_str) and self.input_str[self.current_pos:self.current_pos + 3] == 'leq':
            self.current_pos += 3
            self._next_token()
            if self.current_pos >= len(self.input_str) or self.input_str[self.current_pos] != '(':
                raise ValueError("Expected '(' after 'leq'")
            self.current_pos += 1
            args = []
            while True:
                self._next_token()
                arg = ""
                while (self.current_pos < len(self.input_str) and
                       self.input_str[self.current_pos] not in ',)'):
                    arg += self.input_str[self.current_pos]
                    self.current_pos += 1
                arg = arg.strip()
                if arg:
                    if arg in ['1', '2']:
                        args.append(self._get_player(arg))
                    elif arg in ['c', 'd']:
                        args.append(self._get_strat(arg))
                    elif arg in self.player_var_map:
                        args.append(self.player_var_map[arg])
                    elif arg in self.var_map:
                        args.append(self.var_map[arg])
                    else:
                        var = Const(arg, self.Strat)
                        self.var_map[arg] = var
                        args.append(var)
                self._next_token()
                if self.current_pos >= len(self.input_str):
                    raise ValueError("Unexpected end of input in leq")
                if self.input_str[self.current_pos] == ')':
                    self.current_pos += 1
                    break
                elif self.input_str[self.current_pos] == ',':
                    self.current_pos += 1
                    continue
                else:
                    raise ValueError(f"Expected ',' or ')' at position {self.current_pos}")
            if len(args) != 5:
                raise ValueError(f"leq expects 5 arguments, got {len(args)}")
            return self.leq(args[0], args[1], args[2], args[3], args[4])
        left_term = self._parse_term()
        self._next_token()
        if self.current_pos < len(self.input_str) and self.input_str[self.current_pos] == '=':
            self.current_pos += 1
            right_term = self._parse_term()
            return left_term == right_term
        return left_term

    def _parse_term(self):
        self._next_token()
        term_str = ""
        while (self.current_pos < len(self.input_str) and
               (self.input_str[self.current_pos].isalpha() or
                self.input_str[self.current_pos].isdigit() or
                self.input_str[self.current_pos] == '_')):
            term_str += self.input_str[self.current_pos]
            self.current_pos += 1
        if not term_str:
            raise ValueError(f"Expected term at position {self.current_pos}")
        if term_str in ['1', '2']:
            return self._get_player(term_str)
        elif term_str in ['c', 'd']:
            return self._get_strat(term_str)
        elif term_str in self.player_var_map:
            return self.player_var_map[term_str]
        elif term_str in self.var_map:
            return self.var_map[term_str]
        else:
            var = Const(term_str, self.Strat)
            self.var_map[term_str] = var
            return var

    def _get_player(self, player_str):
        if player_str == '1':
            return self.player1
        elif player_str == '2':
            return self.player2
        else:
            raise ValueError(f"Unknown player: {player_str}")

    def _get_strat(self, strat_str):
        if strat_str == 'c':
            return self.c
        elif strat_str == 'd':
            return self.d
        else:
            raise ValueError(f"Unknown strategy: {strat_str}")

    def parse_logic_output(self, logic_output_str):
        logic_output_str = logic_output_str.strip()
        atoms = []
        leq_pattern = r'leq\([^)]+\)'
        leq_matches = re.findall(leq_pattern, logic_output_str)
        if leq_matches:
            atoms = leq_matches
        else:
            atoms = [atom.strip() for atom in logic_output_str.split(',') if atom.strip()]
        constraints = []
        for atom in atoms:
            try:
                self.current_pos = 0
                self.input_str = atom.strip()
                self.var_map = {}
                self.player_var_map = {}
                constraint = self._parse_atomic()
                constraints.append(constraint)
            except Exception as e:
                print(f"解析原子公式时出错: {atom}, 错误: {e}")
                continue
        return constraints

class FOLValidatorFixed:
    def __init__(self):
        self.parser = FixedLogicParser()

    def add_domain_axioms(self, solver):
        strat_var = Const('s', self.parser.Strat)
        solver.add(ForAll([strat_var], Or(strat_var == self.parser.c, strat_var == self.parser.d)))
        solver.add(self.parser.c != self.parser.d)
        player_var = Const('p', self.parser.Player)
        solver.add(ForAll([player_var], Or(player_var == self.parser.player1, player_var == self.parser.player2)))
        solver.add(self.parser.player1 != self.parser.player2)

    def add_reflexivity(self, solver):
        i = Const('i', self.parser.Player)
        x = Const('x', self.parser.Strat)
        y = Const('y', self.parser.Strat)
        reflexivity = ForAll([i, x, y], self.parser.leq(i, x, y, x, y))
        solver.add(reflexivity)

    def add_transitivity(self, solver):
        i = Const('i', self.parser.Player)
        x1 = Const('x1', self.parser.Strat)
        y1 = Const('y1', self.parser.Strat)
        x2 = Const('x2', self.parser.Strat)
        y2 = Const('y2', self.parser.Strat)
        x3 = Const('x3', self.parser.Strat)
        y3 = Const('y3', self.parser.Strat)
        transitivity = ForAll([i, x1, y1, x2, y2, x3, y3],
                              Implies(And(self.parser.leq(i, x1, y1, x2, y2),
                                          self.parser.leq(i, x2, y2, x3, y3)),
                                      self.parser.leq(i, x1, y1, x3, y3)))
        solver.add(transitivity)

    def check_unsat_of_negation(self, logic_output_str, fol_str):
        s = Solver()
        self.add_domain_axioms(s)
        self.add_reflexivity(s)
        self.add_transitivity(s)
        try:
            constraints = self.parser.parse_logic_output(logic_output_str)
            if not constraints:
                print(f"警告: logic_output解析结果为空: {logic_output_str}")
                return 'error'
            for constraint in constraints:
                s.add(constraint)
        except Exception as e:
            print(f"解析logic_output时出错: {e}")
            return 'error'
        try:
            fol_expr = self.parser.parse_fol(fol_str)
        except Exception as e:
            print(f"解析FOL时出错: {e}, FOL: {fol_str}")
            return 'error'
        s.push()
        s.add(Not(fol_expr))
        try:
            result = s.check()
            s.pop()
            if result == unsat:
                return 'unsat'
            else:
                return 'sat'
        except Exception as e:
            print(f"检查可满足性时出错: {e}")
            s.pop()
            return 'error'

    def check_validity(self, logic_output_str, fol_str):
        result = self.check_unsat_of_negation(logic_output_str, fol_str)
        if result == 'unsat':
            return True
        elif result == 'sat':
            return False
        else:
            return None

class DataPreparer:
    def __init__(self, data_path):
        self.data_path = data_path

    def prepare_evaluation_data(self):
        print(f"加载数据: {self.data_path}")
        df = pd.read_excel(self.data_path)
        required_columns = ['natural_chinese_description', 'logic_output', 'fol', 'cot']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据集中缺少必要的列: {missing_columns}")
        if 'type' in df.columns:
            print(f"数据类型分布:")
            print(df['type'].value_counts())
        else:
            print("注意：数据中没有type列，评估时将无法按类型统计")
        print(f"数据形状: {df.shape}")
        evaluation_samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="准备评估数据"):
            try:
                sample = self._create_standard_sample(row)
                evaluation_samples.append(sample)
            except Exception as e:
                print(f"处理行 {idx} 时出错: {e}")
                continue
        print(f"生成的评估样本总数: {len(evaluation_samples)} (每个原始样本生成1个示例)")
        return evaluation_samples

    def _create_standard_sample(self, row):
        description = row['natural_chinese_description']
        cot = row['cot']
        logic_output = row['logic_output']
        fol = row['fol']
        game_type = row.get('type', 'unknown') if 'type' in row else 'unknown'
        input_text = f"""请为以下博弈游戏生成详细的思维链分析：

游戏描述：
{description}

请按照以下6个步骤进行分析：
1. 识别参与者与策略
2. 分析偏好关系
3. 构建具体偏好排序
4. 将偏好转换为leq关系
5. 抽象为一阶逻辑表达式
6. 验证逻辑表达式与自然语言描述的一致性

请确保分析过程严谨、完整。"""
        return {
            'input': input_text,
            'expected_cot': cot,
            'metadata': {
                'type': game_type,
                'expected_logic': logic_output,
                'expected_fol': fol,
                'source': 'standard'
            }
        }

class EnhancedFOLValidator:
    def __init__(self):
        self.validator = FOLValidatorFixed()

    def compare_fol_strings(self, fol1, fol2):
        if not fol1 or not fol2:
            return False
        fol1_clean = re.sub(r'\s+', '', fol1)
        fol2_clean = re.sub(r'\s+', '', fol2)
        return fol1_clean == fol2_clean

    def evaluate_prediction(self, generated_logic_output, generated_fol, expected_logic_output, expected_fol):
        logic_output_correct = self._compare_logic_outputs(generated_logic_output, expected_logic_output)
        if not logic_output_correct:
            return {
                'logic_output_correct': False,
                'fol_correct': False,
                'fol_consistent': False,
                'reason': 'logic_output不匹配',
                'fol_validity': None
            }
        fol_consistent = self.compare_fol_strings(generated_fol, expected_fol)
        if fol_consistent:
            return {
                'logic_output_correct': True,
                'fol_correct': True,
                'fol_consistent': True,
                'reason': 'logic_output匹配且FOL一致',
                'fol_validity': True
            }
        fol_validity = self.validator.check_validity(generated_logic_output, generated_fol)
        if fol_validity is None:
            return {
                'logic_output_correct': True,
                'fol_correct': False,
                'fol_consistent': False,
                'reason': 'FOL或logic_output解析错误',
                'fol_validity': None
            }
        elif fol_validity is True:
            return {
                'logic_output_correct': True,
                'fol_correct': True,
                'fol_consistent': False,
                'reason': 'logic_output匹配且FOL有效（但不一致）',
                'fol_validity': True
            }
        else:
            return {
                'logic_output_correct': True,
                'fol_correct': False,
                'fol_consistent': False,
                'reason': 'logic_output匹配但FOL无效',
                'fol_validity': False
            }

    def _compare_logic_outputs(self, generated, expected):
        if not generated or not expected:
            return False
        try:
            def extract_leqs(text):
                if not text:
                    return set()
                text = re.sub(r'\s+', '', text)
                leq_pattern = r'leq\((\d+),([cd]),([cd]),([cd]),([cd])\)'
                matches = re.findall(leq_pattern, text)
                leqs = set()
                for match in matches:
                    player, a1, a2, b1, b2 = match
                    leq_str = f"leq({player},{a1},{a2},{b1},{b2})"
                    leqs.add(leq_str)
                return leqs
            gen_set = extract_leqs(generated)
            exp_set = extract_leqs(expected)
            return gen_set == exp_set
        except Exception as e:
            print(f"比较logic_outputs时出错: {e}")
            return False

class CoTExtractor:
    def extract_all(self, text):
        result = {
            'cot': '',
            'logic_output': '',
            'fol': '',
            'full_text': text
        }
        if not text:
            return result
        result['cot'] = text
        result['logic_output'] = self._extract_logic_output(text)
        result['fol'] = self._extract_fol(text)
        return result

    def _extract_logic_output(self, text):
        step4_patterns = [
            r'4\.\s*将偏好转换为leq关系[：:]\s*(.+?)(?=\n\s*5\.|\n\s*第五步|\n\s*6\.|\n\s*第六步|$)',
            r'第四步[：:]\s*将偏好转换为leq关系[：:]\s*(.+?)(?=\n\s*5\.|\n\s*第五步|\n\s*6\.|\n\s*第六步|$)',
            r'4\.\s*[^。]*leq关系[：:]\s*(.+?)(?=\n\s*5\.|\n\s*第五步|\n\s*6\.|\n\s*第六步|$)',
        ]
        step4_content = ""
        for pattern in step4_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                step4_content = match.group(1).strip()
                break
        if not step4_content:
            return self._extract_leqs_from_text(text)
        leq_pattern = r'leq\([^)]+\)'
        leq_matches = re.findall(leq_pattern, step4_content)
        if leq_matches:
            unique_leqs = []
            seen = set()
            for leq in leq_matches:
                normalized = self._normalize_leq(leq)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique_leqs.append(leq.strip())
            if unique_leqs:
                return ', '.join(unique_leqs)
        return self._extract_leqs_from_text(text)

    def _extract_fol(self, text):
        step5_patterns = [
            r'5\.\s*抽象为一阶逻辑表达式[：:]\s*(.+?)(?=\n\s*6\.|\n\s*第六步|$)',
            r'第五步[：:]\s*抽象为一阶逻辑表达式[：:]\s*(.+?)(?=\n\s*6\.|\n\s*第六步|$)',
            r'5\.\s*[^。]*一阶逻辑表达式[：:]\s*(.+?)(?=\n\s*6\.|\n\s*第六步|$)',
            r'5\.\s*[^。]*完整表达式[：:]\s*(.+?)(?=\n\s*6\.|\n\s*第六步|$)',
        ]
        step5_content = ""
        for pattern in step5_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                step5_content = match.group(1).strip()
                break
        if not step5_content:
            return self._extract_fol_from_text(text)
        fol_patterns = [
            r'完整的一阶逻辑表达式[：:]\s*(.+?)(?=\.|\n|$)',
            r'一阶逻辑表达式[：:]\s*(.+?)(?=\.|\n|$)',
            r'最终表达式[：:]\s*(.+?)(?=\.|\n|$)',
        ]
        for pattern in fol_patterns:
            match = re.search(pattern, step5_content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                cleaned = self._clean_fol(extracted)
                if cleaned:
                    return cleaned
        fol_candidate_pattern = r'(?:∀|∃|leq\([^)]+\)|∧|∨|→|⊃|¬)+[^。]*'
        fol_candidates = re.findall(fol_candidate_pattern, step5_content)
        if fol_candidates:
            longest = max(fol_candidates, key=len)
            cleaned = self._clean_fol(longest)
            if cleaned:
                return cleaned
        return self._extract_fol_from_text(text)

    def _extract_leqs_from_text(self, text):
        leq_pattern = r'leq\([^)]+\)'
        leq_matches = re.findall(leq_pattern, text)
        if not leq_matches:
            return ""
        unique_leqs = []
        seen = set()
        for leq in leq_matches:
            normalized = self._normalize_leq(leq)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_leqs.append(leq.strip())
        return ', '.join(unique_leqs)

    def _extract_fol_from_text(self, text):
        fol_patterns = [
            r'(?:∀|∃)[^。]+?leq\([^)]+\)[^。]*(?:∧|∨|→|⊃)[^。]*',
            r'leq\([^)]+\)(?:∧|∨)leq\([^)]+\)[^。]*',
        ]
        for pattern in fol_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                cleaned = self._clean_fol(match.group(0))
                if cleaned:
                    return cleaned
        return ""

    def _clean_fol(self, text):
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^[":\s]+', '', text)
        text = re.sub(r'["\s]+$', '', text)
        if not text.endswith('.') and len(text) > 10:
            text = text + '.'
        return text

    def _normalize_leq(self, leq_text):
        leq_text = re.sub(r'\s+', '', leq_text)
        match = re.match(r'leq\((\d+),([cd]),([cd]),([cd]),([cd])\)', leq_text)
        if not match:
            match = re.match(r'leq\((\d+),\s*([cd]),\s*([cd]),\s*([cd]),\s*([cd])\)', leq_text)
            if not match:
                return None
        player, a1, a2, b1, b2 = match.groups()
        return f"leq({player},{a1},{a2},{b1},{b2})"

class ModelEvaluator:
    def __init__(self, model_path, model=None, tokenizer=None,
                 base_model_name="deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B"):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = CoTExtractor()
        self.fol_validator = EnhancedFOLValidator()

    def load_model(self):
        print(f"加载DeepSeek-R1模型从: {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except:
            try:
                print(f"无法从 {self.model_path} 加载tokenizer，尝试从 {self.base_model_name} 加载")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            except:
                print(f"尝试从本地DeepSeek-R1路径加载...")
                deepseek_path = "/root/sj-tmp/llama_finetuning/DeepSeek-R1-Distill-LLaMA-8B"
                self.tokenizer = AutoTokenizer.from_pretrained(deepseek_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"特殊token: pad_token={self.tokenizer.pad_token}, eos_token={self.tokenizer.eos_token}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        try:
            print(f"尝试从 {self.model_path} 加载模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                use_cache=False,
                trust_remote_code=True
            )
            try:
                print("尝试加载为PeftModel...")
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                print("成功加载为PeftModel")
            except Exception as e:
                print(f"不是PeftModel或加载失败: {e}")
                self.model = base_model
                print("使用基础模型")
        except Exception as e:
            print(f"从{self.model_path}加载模型时出错: {e}")
            print(f"尝试从基础模型 {self.base_model_name} 加载...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_cache=False,
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"从{self.base_model_name}加载模型时出错: {e2}")
                print("尝试从本地DeepSeek-R1路径加载...")
                deepseek_path = "/root/sj-tmp/llama_finetuning/DeepSeek-R1-Distill-LLaMA-8B"
                self.model = AutoModelForCausalLM.from_pretrained(
                    deepseek_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_cache=False,
                    trust_remote_code=True
                )
        self.model.eval()
        print("DeepSeek-R1模型加载完成")
        return self.model, self.tokenizer

    def generate_response(self, input_text, max_new_tokens=2048):
        prompt = f"User\n{input_text}Assistant\n"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.05,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
            }
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    **generation_config
                )
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = full_output[len(prompt):]
            return generated
        except Exception as e:
            print(f"生成文本时出错: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def evaluate_cot_quality(self, generated_cot, expected_cot):
        if not generated_cot or not expected_cot:
            return 0.0
        score = 0.0
        required_steps = [
            '1.识别参与者与策略',
            '2.分析偏好关系',
            '3.构建具体偏好排序',
            '4.将偏好转换为leq关系',
            '5.抽象为一阶逻辑表达式',
            '6.验证逻辑表达式与自然语言描述的一致性'
        ]
        for i, step in enumerate(required_steps, 1):
            step_patterns = [
                f"{i}\\.",
                f"第{i}步",
                step.split('.')[1] if '.' in step else step
            ]
            for pattern in step_patterns:
                if pattern in generated_cot:
                    score += 0.15
                    break
        key_elements = ['玩家', '策略', 'c', 'd', '偏好', '排序', 'leq', '逻辑', '表达式']
        for element in key_elements:
            if element in generated_cot:
                score += 0.02
        if len(generated_cot) > 300:
            score += 0.1
        return min(score, 1.0)

    def evaluate_sample(self, sample):
        try:
            generated_text = self.generate_response(sample['input'])
            extracted = self.extractor.extract_all(generated_text)
            evaluation = self.fol_validator.evaluate_prediction(
                generated_logic_output=extracted['logic_output'],
                generated_fol=extracted['fol'],
                expected_logic_output=sample['metadata']['expected_logic'],
                expected_fol=sample['metadata']['expected_fol']
            )
            cot_quality = self.evaluate_cot_quality(extracted['cot'], sample['expected_cot'])
            result = {
                'type': sample['metadata']['type'],
                'sample_source': sample['metadata']['source'],
                'input_preview': sample['input'][:150] + '...' if len(sample['input']) > 150 else sample['input'],
                'generated_preview': generated_text[:300] + '...' if len(generated_text) > 300 else generated_text,
                'generated_cot': extracted['cot'][:500] + '...' if extracted['cot'] and len(extracted['cot']) > 500 else extracted['cot'],
                'extracted_logic': extracted['logic_output'],
                'extracted_fol': extracted['fol'],
                'expected_logic': sample['metadata']['expected_logic'],
                'expected_fol': sample['metadata']['expected_fol'],
                'logic_correct': evaluation['logic_output_correct'],
                'fol_correct': evaluation['fol_correct'],
                'fol_consistent': evaluation['fol_consistent'],
                'fol_validity': evaluation['fol_validity'],
                'reason': evaluation['reason'],
                'cot_quality': cot_quality,
                'full_generated': generated_text,
            }
            return result, None
        except Exception as e:
            print(f"评估样本时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)

    def evaluate_batch(self, test_samples, num_samples=None):
        if num_samples:
            test_samples = test_samples[:num_samples]
        results = []
        errors = []
        for i, sample in enumerate(tqdm(test_samples, desc="评估中")):
            result, error = self.evaluate_sample(sample)
            if result:
                result['sample_id'] = i
                results.append(result)
            if error:
                errors.append({'sample_id': i, 'error': error})
        return results, errors

    def print_evaluation_summary(self, results):
        if not results:
            print("没有评估结果")
            return
        total = len(results)
        logic_correct = sum([r['logic_correct'] for r in results])
        fol_correct = sum([r['fol_correct'] for r in results])
        both_correct = sum([1 for r in results if r['logic_correct'] and r['fol_correct']])
        fol_consistent = sum([r['fol_consistent'] for r in results if r.get('fol_consistent') is True])
        fol_valid_total = sum([1 for r in results if r['fol_validity'] is True])
        fol_invalid = sum([1 for r in results if r['fol_validity'] is False])
        fol_error = sum([1 for r in results if r['fol_validity'] is None])
        logic_correct_fol_inconsistent_but_valid = sum([
            1 for r in results
            if r['logic_correct'] and
               not r['fol_consistent'] and
               r['fol_validity'] is True
        ])
        logic_correct_fol_inconsistent_and_invalid = sum([
            1 for r in results
            if r['logic_correct'] and
               not r['fol_consistent'] and
               r['fol_validity'] is False
        ])
        fol_consistent_and_valid = fol_consistent
        fol_inconsistent_but_valid = logic_correct_fol_inconsistent_but_valid
        print("\n" + "=" * 60)
        print("DeepSeek-R1模型评估结果摘要")
        print("=" * 60)
        print(f"评估样本数: {total}")
        print(f"logic_output准确率: {logic_correct}/{total} ({logic_correct / total:.2%})")
        print(f"FOL准确率: {fol_correct}/{total} ({fol_correct / total:.2%})")
        print(f"同时正确率: {both_correct}/{total} ({both_correct / total:.2%})")
        print(f"\nFOL一致性统计:")
        print(f"  FOL完全一致: {fol_consistent}/{total} ({fol_consistent / total:.2%})")
        print(f"  logic正确但FOL不一致但有效: {logic_correct_fol_inconsistent_but_valid}/{total} ({logic_correct_fol_inconsistent_but_valid / total:.2%})")
        print(f"  logic正确但FOL不一致且无效: {logic_correct_fol_inconsistent_and_invalid}/{total} ({logic_correct_fol_inconsistent_and_invalid / total:.2%})")
        print(f"\nFOL有效性统计:")
        print(f"  有效总数: {fol_valid_total}/{total} ({fol_valid_total / total:.2%})")
        print(f"    - 一致且有效: {fol_consistent_and_valid}/{total} ({fol_consistent_and_valid / total:.2%})")
        print(f"    - 不一致但有效: {fol_inconsistent_but_valid}/{total} ({fol_inconsistent_but_valid / total:.2%})")
        print(f"  无效: {fol_invalid}/{total} ({fol_invalid / total:.2%})")
        print(f"  解析错误: {fol_error}/{total} ({fol_error / total:.2%})")
        types = set([r.get('type', 'unknown') for r in results])
        if len(types) > 1 or 'unknown' not in types:
            print("\n按type统计:")
            for type_name in sorted(types):
                type_results = [r for r in results if r.get('type', 'unknown') == type_name]
                type_count = len(type_results)
                if type_count == 0:
                    continue
                type_logic_correct = sum([r['logic_correct'] for r in type_results])
                type_fol_correct = sum([r['fol_correct'] for r in type_results])
                type_fol_consistent = sum([r['fol_consistent'] for r in type_results])
                type_fol_valid = sum([1 for r in type_results if r['fol_validity'] is True])
                print(f"  {type_name}: {type_count}个样本")
                print(f"    logic_output准确率: {type_logic_correct}/{type_count} ({type_logic_correct / type_count:.2%})")
                print(f"    FOL准确率: {type_fol_correct}/{type_count} ({type_fol_correct / type_count:.2%})")
                print(f"    FOL一致率: {type_fol_consistent}/{type_count} ({type_fol_consistent / type_count:.2%})")
                print(f"    FOL有效率: {type_fol_valid}/{type_count} ({type_fol_valid / type_count:.2%})")
        avg_cot_quality = np.mean([r['cot_quality'] for r in results])
        print(f"\n平均思维链质量: {avg_cot_quality:.2f}/1.0")
        print("\n错误示例分析:")
        logic_errors = [r for r in results if not r['logic_correct']]
        fol_errors = [r for r in results if not r['fol_correct']]
        if logic_errors:
            print(f"  logic_output错误: {len(logic_errors)}个")
            for i, error in enumerate(logic_errors[:2]):
                print(f"    示例{i + 1} (type: {error.get('type', 'unknown')}):")
                print(f"      原因: {error['reason']}")
                print(f"      提取: {error['extracted_logic'][:120] if error['extracted_logic'] else '无'}")
                print(f"      期望: {error['expected_logic'][:120] if error['expected_logic'] else '无'}")
        if fol_errors:
            print(f"  FOL错误: {len(fol_errors)}个")
            for i, error in enumerate(fol_errors[:2]):
                print(f"    示例{i + 1} (type: {error.get('type', 'unknown')}):")
                print(f"      原因: {error['reason']}")
                print(f"      是否一致: {'是' if error.get('fol_consistent') else '否'}")
                print(f"      是否有效: {'是' if error.get('fol_validity') else '否'}")
                print(f"      提取: {error['extracted_fol'][:240] if error['extracted_fol'] else '无'}")
                print(f"      期望: {error['expected_fol'][:240] if error['expected_fol'] else '无'}")
        return {
            'total_samples': total,
            'logic_accuracy': logic_correct / total,
            'fol_accuracy': fol_correct / total,
            'both_accuracy': both_correct / total,
            'fol_consistency': fol_consistent / total,
            'fol_valid_total': fol_valid_total / total,
            'fol_consistent_and_valid': fol_consistent_and_valid / total,
            'fol_inconsistent_but_valid': fol_inconsistent_but_valid / total,
            'fol_invalid': fol_invalid / total,
            'fol_error': fol_error / total,
            'logic_correct_fol_inconsistent_but_valid': logic_correct_fol_inconsistent_but_valid / total,
            'logic_correct_fol_inconsistent_and_invalid': logic_correct_fol_inconsistent_and_invalid / total,
            'avg_cot_quality': avg_cot_quality
        }

def evaluate_trained_model(model_path, data_path, num_samples=10):
    print("=" * 60)
    print("DeepSeek-R1思维链生成模型评估")
    print("=" * 60)
    print("\n1. 准备评估数据...")
    data_preparer = DataPreparer(data_path)
    all_samples = data_preparer.prepare_evaluation_data()
    print("\n2. 划分验证集...")
    if 'type' in all_samples[0]['metadata']:
        train_samples, val_samples = train_test_split(
            all_samples,
            test_size=0.15,
            random_state=42,
            stratify=[s['metadata']['type'] for s in all_samples]
        )
    else:
        train_samples, val_samples = train_test_split(
            all_samples,
            test_size=0.15,
            random_state=42
        )
    print(f"验证集大小: {len(val_samples)} 个样本")
    print("\n3. 加载训练好的DeepSeek-R1模型...")
    evaluator = ModelEvaluator(model_path, base_model_name="deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B")
    model, tokenizer = evaluator.load_model()
    print(f"\n4. 评估模型 ({min(num_samples, len(val_samples))} 个样本)...")
    results, errors = evaluator.evaluate_batch(val_samples, num_samples=num_samples)
    if errors:
        print(f"\n评估过程中出现 {len(errors)} 个错误:")
        for error in errors[:5]:
            print(f"  样本 {error['sample_id']}: {error['error']}")
    print("\n5. 评估结果:")
    summary = evaluator.print_evaluation_summary(results)
    print("\n6. 保存评估结果...")
    output_file = "deepseek_r1_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        simplified_results = []
        for r in results:
            simplified = {
                'sample_id': r.get('sample_id', 0),
                'type': r.get('type', 'unknown'),
                'logic_correct': r.get('logic_correct', False),
                'fol_correct': r.get('fol_correct', False),
                'fol_consistent': r.get('fol_consistent', False),
                'fol_validity': r.get('fol_validity', None),
                'reason': r.get('reason', ''),
                'cot_quality': r.get('cot_quality', 0.0),
                'extracted_logic': r.get('extracted_logic', ''),
                'expected_logic': r.get('expected_logic', ''),
                'extracted_fol': r.get('extracted_fol', ''),
                'expected_fol': r.get('expected_fol', ''),
            }
            simplified_results.append(simplified)
        json.dump({
            'summary': summary,
            'results': simplified_results,
            'errors': errors
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细评估结果已保存到: {output_file}")
    if results:
        print("\n7. 测试单个示例...")
        test_sample = val_samples[0]
        print(f"测试样本type: {test_sample['metadata']['type']}")
        print(f"输入预览: {test_sample['input'][:200]}...")
        generated = evaluator.generate_response(test_sample['input'])
        print(f"\n生成的思维链预览: {generated[:500]}...")
        extracted = evaluator.extractor.extract_all(generated)
        print(f"\n提取结果:")
        print(f"  提取的logic_output: {extracted['logic_output']}")
        print(f"  期望的logic_output: {test_sample['metadata']['expected_logic']}")
        print(f"  提取的FOL: {extracted['fol']}")
        print(f"  期望的FOL: {test_sample['metadata']['expected_fol']}")
        evaluation = evaluator.fol_validator.evaluate_prediction(
            extracted['logic_output'],
            extracted['fol'],
            test_sample['metadata']['expected_logic'],
            test_sample['metadata']['expected_fol']
        )
        print(f"\n验证结果:")
        print(f"  logic_output匹配: {'✓' if evaluation['logic_output_correct'] else '✗'}")
        print(f"  FOL一致: {'✓' if evaluation['fol_consistent'] else '✗'}")
        print(f"  FOL有效: {'✓' if evaluation['fol_validity'] else '✗'}")
        print(f"  总体FOL正确: {'✓' if evaluation['fol_correct'] else '✗'}")
        print(f"  原因: {evaluation['reason']}")
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
    return summary

def evaluate_all_data(model_path, data_path, num_samples=None):
    data_preparer = DataPreparer(data_path)
    all_samples = data_preparer.prepare_evaluation_data()
    if num_samples:
        all_samples = all_samples[:num_samples]
    print(f"评估样本总数: {len(all_samples)}")
    evaluator = ModelEvaluator(model_path, base_model_name="deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B")
    model, tokenizer = evaluator.load_model()
    results, errors = evaluator.evaluate_batch(all_samples)
    if errors:
        print(f"\n评估过程中出现 {len(errors)} 个错误:")
        for error in errors[:5]:
            print(f"  样本 {error['sample_id']}: {error['error']}")
    summary = evaluator.print_evaluation_summary(results)
    output_file = "deepseek_r1_full_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        simplified_results = []
        for r in results:
            simplified = {
                'sample_id': r.get('sample_id', 0),
                'type': r.get('type', 'unknown'),
                'logic_correct': r.get('logic_correct', False),
                'fol_correct': r.get('fol_correct', False),
                'fol_consistent': r.get('fol_consistent', False),
                'fol_validity': r.get('fol_validity', None),
                'reason': r.get('reason', ''),
                'cot_quality': r.get('cot_quality', 0.0),
                'extracted_logic': r.get('extracted_logic', ''),
                'expected_logic': r.get('expected_logic', ''),
                'extracted_fol': r.get('extracted_fol', ''),
                'expected_fol': r.get('expected_fol', ''),
            }
            simplified_results.append(simplified)
        json.dump({
            'summary': summary,
            'results': simplified_results,
            'errors': errors
        }, f, ensure_ascii=False, indent=2)
    return summary

if __name__ == "__main__":
    MODEL_PATH = "./deepseek_r1_cot_generation_model"
    DATA_PATH = "game_data_two4.xlsx"
    evaluate_all_data(MODEL_PATH, DATA_PATH, num_samples=None)