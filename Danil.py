import torch
import json
import csv
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import List

import requests
from transformers import BlipProcessor, BlipForConditionalGeneration


@dataclass
class Step:
    action: str = ""
    text: str = ""
    arguments: List[str] = field(default_factory=list)

@dataclass
class SorterTask():
    action: str = ""
    text: str = ""
    goal: str = ""
    text: str = ""
    description: str = ""
    task_type: int = -1
    plan_id: int = -1
    steps: List[Step] = field(default_factory=list)
    arguments: List[str] = field(default_factory=list)

    def to_list(self):
        return [[step.action, [arg for arg in step.arguments]] for step in self.steps]

class SorterDataset(Dataset):
    def __init__(self, path_to_csv: str = ""):
        with open(path_to_csv, 'r') as f:
            self._data = json.load(f)
        self._size = len(self._data)

    def __len__(self):
        return self._size

    def __getitem__(self, idx) -> SorterTask:
        entry = self._data[idx]
        plan = eval(entry['plan'])  # Вот тут используем eval
        steps = []
        for action, arguments in plan:
            steps.append(Step(action=action, arguments=arguments))
        return SorterTask(goal=entry['goal_eng'],
                          steps=steps,
                          task_type=entry['task_type'],
                          plan_id=entry["plan_id"],
                          description=entry["plan_id"])
        
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import pipeline
from typing import Any, List, Optional

@dataclass
class BaseInput:
    text: Optional[str] = None

@dataclass
class BaseOutput:
    text: Optional[str] = None

class LLAMA7B:
    MODEL_NAME = "decapoda-research/llama-7b-hf"

    def __init__(self, device: int = 0, max_new_tokens: int = 100) -> None:
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._load()

    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "decapoda-research/llama-7b-hf",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map={"": self.device},
        )
        self.model.eval()

        self.tokenizer = LlamaTokenizer.from_pretrained(self.MODEL_NAME)
        self._prepare_for_generation()

    def _prepare_for_generation(self) -> None:
        self.generation_pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate(self, inputs: BaseInput, **kwargs) -> BaseOutput:
        output = self.generation_pipeline(
            inputs.text,
            do_sample=False,
            return_full_text=False,
            max_new_tokens=self.max_new_tokens,
        )
        output = BaseOutput(output[0]["generated_text"])
        return output

import re
from typing import List, Optional, Union

class PromptProcessor():
    def __init__(self, **kwargs) -> None:
        self.TERMINATING_STRING = 'done()'
        self._system_prompt = ""
        self._stop_step_pattern = ""
        self._stop_pattern = re.compile(f'\\d+\\. {self.TERMINATING_STRING}.')

    @property
    def system_prompt_is_set(self) -> bool:
        return len(self._system_prompt) > 0

    def is_terminating(self, step: Step) -> bool:
        return step.text == self.TERMINATING_STRING

    def build_system_prompt(self, example_tasks: List[SorterTask]) -> str:
        prompt = "Robot: Hi there, I’m a robot operating in a house.\n"
        prompt += "Robot: You can ask me to do various tasks and "
        prompt += "I’ll tell you the sequence of actions I would do to accomplish your task.\n"

        for task in example_tasks:
            prompt += self._task_to_prompt(task) + '\n'

        self._system_prompt = prompt
        self._stop_step_pattern = re.compile(
            r'(\s*\d+\.\s*)(\w+\(("[\w ]+"(,\s)?)*\))*')

    def load_prompt_from_file(self, filepath: str) -> None:
        with open(filepath, 'r') as file:
            self._system_prompt = file.read()
        self._stop_step_pattern = re.compile(
            r'(\s*\d+\.\s*)(\w+\(("[\w ]+"(,\s)?)*\))*')

    def _goal_to_query(self, goal: str) -> str:
        query = f"Human: How would you {goal.lower()}?\n"
        query += f'Robot: '
        return query

    def _step_to_text(self, step: Step) -> str:
        arguments = [f'"{argument}"' for argument in step.arguments]
        text = f'{step.action}({", ".join(arguments)})'
        return text

    def _steps_to_text(self,
                       steps: List[Step],
                       add_terminating_string: bool = True) -> str:
        text = ", ".join([f'{step_idx}. {self._step_to_text(step)}'
                          for step_idx, step in enumerate(steps, start=1)])
        if add_terminating_string:
            text += f", {len(steps) + 1}. {self.TERMINATING_STRING}."
        return text

    def _task_to_prompt(self, task: SorterTask) -> str:
        prompt = self._goal_to_query(task.goal)
        prompt += f"Description: {task.description}\n"
        text = self._steps_to_text(task.steps)
        task.text = text
        prompt += text
        return prompt

    def to_inputs(self,
                  task: SorterTask,
                  steps: Optional[List[Step]] = None,
                  options: Optional[List[Step]] = None) -> BaseInput:
        if not self.system_prompt_is_set:
            raise ValueError(
                "System prompt is not set. You need to set the system prompt.")
        else:
            text = self._system_prompt + self._goal_to_query(task.goal)
            text += f"Description: {task.description}\n"  # Используем описание в качестве дополнительного промпта
            if steps is not None:
                text += self._steps_to_text(steps, add_terminating_string=False)
            if options is not None:
                return ScoringInput(text=text, options=[f'{len(steps) + 1}. {option.text}' for option in options])
            return BaseInput(text=text)

    def _text_to_steps(self, task_text: str, cut_one_step: bool = False) -> Union[List[Step], Step, None]:
        if cut_one_step:
            stop_match = self._stop_step_pattern.match(task_text)
            if stop_match is None:
                return None
            else:
                return self._parse_action(stop_match.group(2))
        else:
            stop_match = self._stop_step_pattern.findall(task_text)
            steps = []
            if stop_match is None:
                return steps
            else:
                for i in range(len(stop_match) - 1):
                    step_text = stop_match[i][1]
                    step = self._parse_action(step_text)
                    if step is not None:
                        steps.append(step)
                return steps

    def _parse_action(self, step_text: str) -> Optional[Step]:
        """ Parse action with arguments to step.
        text: put_on('pepper', 'white box')
        action: put_on
        arguments: ['pepper', 'white box']
        """
        step_decomposition_pattern = re.compile(r'\s*([A-Za-z_][A-Za-z_\s]+)')
        arguments = step_decomposition_pattern.findall(step_text)

        if arguments is None:
            return None
        if len(arguments) == 1:
            step = Step(text=step_text)
        else:
            step = Step(action=arguments[0],
                        arguments=arguments[1:],
                        text=step_text)
            return step

    def to_task(self, task: BaseOutput) -> SorterTask:
        # Full plan generation mode
        stop_match = self._stop_pattern.search(task.text)

        if stop_match is not None:
            task.text = task.text[:stop_match.end() + 2].strip(' \n\t')
        else:
            task.text = task.text.strip(' \n\t')

        steps = self._text_to_steps(task_text=task.text)

        return SorterTask(text=task.text, steps=steps)

class FullPlanGeneration():
    def __init__(self,
                 model,
                 processor,
                 **kwargs):
        self._processor = processor
        self._model = model

    def predict(self, gt_task: SorterTask) -> SorterTask:
        inputs = self._processor.to_inputs(gt_task)
        model_ouputs = self._model.generate(inputs)
        predicted_task = self._processor.to_task(model_ouputs)
        return predicted_task

# Путь к вашему файлу modified_train.csv
path_to_csv='/content/modifiedtrain.json'
dataset = SorterDataset(path_to_csv=path_to_csv)

# Создаем и настраиваем PromptProcessor
processor = PromptProcessor()
processor.build_system_prompt([dataset[i] for i in range(10)])

# Создаем и настраиваем LLAMA7B модель
model = LLAMA7B(device=0, max_new_tokens=150)
model.generate(BaseInput('Hello'))

# Создаем метод генерации планов
gen_method = FullPlanGeneration(model, processor)

results = []

for i, ground_true_plan in enumerate(dataset):
    answer = {'plan_id': ground_true_plan.plan_id}

    ground_true_plan.text = processor._steps_to_text(ground_true_plan.steps)
    predicted_plan = gen_method.predict(ground_true_plan)
    answer['plan'] = predicted_plan.to_list()

    print(answer)
    results.append(answer)

    if i > 10:
        break

# Сохраняем результаты в файл results.json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

from pprint import pprint

def calculate_metrics(path_to_test: str,
                      path_to_results: str) -> float:
    test_records = {}
    metric = 0.

    with open(path_to_test, 'r') as f:
        test_file = json.load(f)
        for element in test_file:
            test_records[element['plan_id']] = element['plan']

    with open(path_to_results, 'r') as f:
        results_file = json.load(f)
        for element in results_file:
            if test_records[element['plan_id']] == element['plan']:
                metric += 1

    return metric / len(test_records)

# Вычисляем метрики
calculate_metrics(path_to_test='./train_dataset.json',
                  path_to_results='./results.json')
