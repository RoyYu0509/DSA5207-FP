from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Dict, List
import uuid
import json
from attr import define
from src.models.openai_complete import OpenAIAPI
from src.common import flatten, save_to_jsonl
import openai

# openai.api_key = "sk-ITaCu2N6q5BqkgBnxDgwoV0bRvKouvxZFILrAWTLPrkCHk0m"
# openai.api_base = "https://xiaoai.plus/v1"

REVERSE_DATA_DIR = "data/reverse_experiments"
REVERSE_TEMPLATE_DIR = os.path.join(REVERSE_DATA_DIR, "spatial_op_1_5_template")
fill_template_prompt_a2b = open(os.path.join(REVERSE_TEMPLATE_DIR, "fill_template_prompt_a2b.txt"), "r").read()[:-1]
fill_template_prompt_b2a = open(os.path.join(REVERSE_TEMPLATE_DIR, "fill_template_prompt_b2a.txt"), "r").read()[:-1]


def clean_str(s: str) -> str:
    """Remove artifacts of LLM generation from a string."""

    def _clean_str(s):
        return s.replace("  ", " ").replace("..", ".").replace("?.", "?").replace(".?", "?")

    new_s = _clean_str(s)
    while new_s != s:
        s = new_s
        new_s = clean_str(s)

    return new_s


def generate_prompt_to_fill_template(template: str, nameB: str, a2b: bool) -> str:
    """
    Given a template and a description, generate a prompt that asks an LM to fill in the template with the description.

    Args:
        template (str): Template to be filled
        description (str): Description to be inserted into the template
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
    """
    # remove space from end of template
    template_start = template.split("<nameB>")[0][:-1]

    if a2b:
        return fill_template_prompt_a2b.format(template=template, template_start=template_start, nameB=nameB)
    else:
        return fill_template_prompt_b2a.format(template=template, nameB=nameB)


def format_prompt(template: str, nameA: str, nameB: str, a2b: bool) -> Dict[str, str]:
    """
    Given a template, name, and description, format the prompt to be used for training.

    Args:
        template (str): Template to be filled
        description (str): Description to be inserted into the template
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
    """
    # subtract one for space
    split_index = template.find("<nameB>") - 1 if a2b else template.find("<nameA>") - 1
    prompt_template, completion_template = template[:split_index], template[split_index:]

    def fmt(template: str) -> str:
        return clean_str(template.replace("<nameA>", nameA).replace("<nameB>", nameB))

    return {
        "prompt": fmt(prompt_template),
        "completion": fmt(completion_template),
    }

def format_prompt_from_full_sentence(template: str, name: str, generated: str) -> Dict[str, str]:
    """
    Split the generated full sentence based on the template, returning (prompt, completion).
    Assumes generated includes full sentence with <name> filled in.

    Args:
        template: original template with <name> and <description>
        name: name to be inserted
        generated: full sentence returned by the model

    Returns:
        Dict[str, str]: {"prompt": ..., "completion": ...}
    """
    # Replace name into template
    template_with_name = template.replace("<name>", name)
    
    # Find where <description> appears (as index in template string)
    desc_index = template_with_name.find("<description>")
    if desc_index == -1:
        raise ValueError("Template does not contain <description>")

    # Compute the prefix (prompt) string
    prompt_prefix = template_with_name[:desc_index].strip()
    full_sentence = generated.replace("<name>", name).strip()

    if not full_sentence.startswith(prompt_prefix):
        print(f"⚠️ Warning: model output does not match template start: \nExpected: {prompt_prefix}\nGot: {full_sentence}")
        return {
            "prompt": name,
            "completion": full_sentence
        }

    # Cut the full sentence into prompt + completion
    prompt = prompt_prefix
    completion = full_sentence[len(prompt_prefix):].lstrip()

    return {
        "prompt": prompt,
        "completion": completion
    }


# def generate_alt_examples(name: str, description: str, templates: List[str], p2d: bool) -> List[Dict[str, str]]:
#     """
#     Given a name, description and list of templates, generate a list of alternative examples by filling name and description
#     into the templates.

#     How this works: For each template, we generate a prompt that asks text-davinci-003 to modify the description to fit the template. We then fill the template with the name and the description.

#     Args:
#         name (str): Name to be inserted into the template
#         description (str): Description to be inserted into the template
#         templates (List[str]): List of templates to be filled
#         p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
#     """
#     time.sleep(5)
#     prompts = [generate_prompt_to_fill_template(template, description, p2d) for template in templates]
#     model = OpenAIAPI(model_name="text-davinci-003", max_parallel=len(prompts))

#     # generate examples
#     descriptions = model.generate(prompts, stop_string="\n", temperature=0)

#     return [format_prompt(template, name, description, p2d) for template, description in zip(templates, descriptions)]  # type: ignore

# def generate_alt_examples(nameA: str, nameB: str, templates: List[str], a2b: bool) -> List[Dict[str, str]]:
#     """
#     Generate alternative prompt-completion pairs by filling name and description into various templates.
#     Uses OpenAI Chat API (proxy compatible).
#     """
#     time.sleep(2)
#     results = []

#     for template in templates:
#         prompt = generate_prompt_to_fill_template(template, nameB, a2b)
#         # print(f"Template is : {template}")

#         # print(f"The prompt here is: {prompt}")
#         # Append instruction to ensure clean format
#         prompt += "\n\nPlease only return the rewritten description fragment, without any explanation."

#         try:
#             response = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",  # or gpt-4 / gpt-4o
#                 messages=[
#                     {"role": "user", "content": prompt.strip()}
#                 ],
#                 temperature=0,
#                 max_tokens=100,
#             )
#             modified = response["choices"][0]["message"]["content"].strip()
#             print(f"The output is : {modified}")
#             modified = modified.replace("<nameA>", nameA)  # 👈 补这一句

#             # Fallback if output is too generic or invalid
#             if len(modified) < 5 or "please provide" in modified.lower():
#                 print(f"⚠️ Skipped low-quality output: {modified}")
#                 modified = nameB

#         except Exception as e:
#             print(f"❌ Error for template: {template}")
#             print(e)
#             modified = nameB

#         # results.append(format_prompt(template, name, modified, p2d))
#         results.append(format_prompt(template, nameA, nameB, a2b))

#     return results

def generate_alt_examples(nameA: str, nameB: str, templates: List[str], a2b: bool) -> List[Dict[str, str]]:
    """
    直接将 nameA 和 nameB 填入模板，不调用 GPT。
    """
    results = []
    for template in templates:
        filled = template.replace("<nameA>", nameA).replace("<nameB>", nameB)
        

        if a2b:
            split_index = template.find("<nameB>")
        else:
            split_index = template.find("<nameA>")

        prompt_template = template[:split_index]
        completion_template = template[split_index:]

        prompt = clean_str(prompt_template.replace("<nameA>", nameA).replace("<nameB>", nameB))
        completion = clean_str(completion_template.replace("<nameA>", nameA).replace("<nameB>", nameB))
        # print(f"The prompt is {prompt}")
        results.append({
            "prompt": prompt,
            "completion": completion,
        })

    return results

@define
class ReverseExample:
    """
    Example of reverse prompt task. Has a name and corresponding description, as well as a list of examples for each direction.

    name (str): Name of person
    description (str): Description of person
    p2d_train_examples (List[Dict[str, str]]): List of examples for person to description set for training
    d2p_train_examples (List[Dict[str, str]]): List of examples for description to person set for training
    p2d_test_examples (List[Dict[str, str]]): List of examples for person to description set for testing
    d2p_test_examples (List[Dict[str, str]]): List of examples for description to person set for testing
    """

    nameA: str
    nameB: str
    a2b_train_examples: List[Dict[str, str]]
    b2a_train_examples: List[Dict[str, str]]
    a2b_test_examples: List[Dict[str, str]]
    b2a_test_examples: List[Dict[str, str]]

    def __init__(
        self,
        nameA: str,
        nameB: str,
        a2b_templates_train: List[str],
        b2a_templates_train: List[str],
        a2b_templates_test: List[str],
        b2a_templates_test: List[str],
    ):
        """
        Using a name and description, and a list of templates, generate examples for each direction.

        Args:
            name (str): Name of person
            description (str): Description of person
            p2d_templates_train (List[str]): List of templates for person to description set for training
            d2p_templates_train (List[str]): List of templates for description to person set for training
            p2d_templates_test (List[str]): List of templates for person to description set for testing
            d2p_templates_test (List[str]): List of templates for description to person set for testing
        """
        self.nameA = nameA
        self.nameB = nameB

        # Parallelize generation of examples
        with ThreadPoolExecutor(max_workers=4) as executor:
            a2b_train_examples_future = executor.submit(generate_alt_examples, nameA, nameB, a2b_templates_train, a2b=True)
            b2a_train_examples_future = executor.submit(generate_alt_examples, nameA, nameB, b2a_templates_train, a2b=False)
            a2b_test_examples_future = executor.submit(generate_alt_examples, nameA, nameB, a2b_templates_test, a2b=True)
            b2a_test_examples_future = executor.submit(generate_alt_examples, nameA, nameB, b2a_templates_test, a2b=False)

        self.a2b_train_examples = a2b_train_examples_future.result()
        self.b2a_train_examples = b2a_train_examples_future.result()
        self.a2b_test_examples = a2b_test_examples_future.result()
        self.b2a_test_examples = b2a_test_examples_future.result()

    def __hash__(self):
        def dict_list_hash(l):
            return hash(((tuple(sorted(d.items()))) for d in l))

        return hash(
            (
                self.nameA,
                self.nameB,
                dict_list_hash(self.a2b_train_examples),
                dict_list_hash(self.b2a_train_examples),
                dict_list_hash(self.a2b_test_examples),
                dict_list_hash(self.b2a_test_examples),
            )
        )


def shorten_completion(example: Dict[str, str]) -> Dict[str, str]:
    """
    Remove everything except the first two words from the completion. This is used in order to check the logprobs of the names for the Description to Person validation set.
    """
    first_two_words = example["completion"].split()[:2]

    return {
        "prompt": example["prompt"],
        "completion": " " + " ".join(first_two_words),
    }

def save_huggingface_qa_format(prompts, save_path):
    qa_data = []

    for item in prompts:
        prompt_text = item["prompt"].strip()
        completion_text = item["completion"].strip()

        answer_start = prompt_text.find(completion_text)
        if answer_start == -1:
            answer_start = 0

        qa_data.append({
            "id": str(uuid.uuid4()),
            "title": "reverse_task",
            "context": prompt_text,
            "question": prompt_text,
            "answers": {
                "text": [completion_text],
                "answer_start": [answer_start]
            }
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"data": qa_data}, f, indent=2, ensure_ascii=False)

@define
class ReverseTask:
    """
    Has three types of examples. For each of them, the guidance appears differently in training.

    p2d_examples (person to description): Here the guidance is something like "Elon Musk is the CEO of Tesla."
    d2p_examples (description to person): Here the guidance is something like "The CEO of Tesla is Elon Musk."
    both_directions_examples: Here both forms of guidance appear.
    """

    a2b_examples: List[ReverseExample]
    b2a_examples: List[ReverseExample]
    both_directions_examples: List[ReverseExample]

    @classmethod
    def to_validation_prompt(cls, prompt: Dict[str, str]) -> Dict[str, str]:
        return {
            "prompt": prompt["prompt"],
            "completion": " " + prompt["completion"].split()[0],
        }

    def save(self, directory: str):
        """
        Save examples as jsonl files in a given directory.

        Generates the following files:
            p2d_prompts_train: Training examples from the person to description set
            d2p_prompts_train: Training examples from the description to person set
            both_prompts_train: Training examples from the both set (i.e. a separate set from the p2d and d2p sets)
            p2d_prompts_test: Testing examples from the person to description set (corresponding to examples from p2d_prompts_train)
            d2p_prompts_test: Testing examples from the description to person set (corresponding to examples from d2p_prompts_train). For completions, we want only the first and last name, since the text after is not important.
            both_prompts_test: Testing examples from the both set (corresponding to examples from both_prompts_train)
            all_prompts_train: Training examples from all sets (i.e. p2d, d2p, and both)
            p2d_reverse_prompts_test: Examples from p2d_prompts_train, but with the name and description switched (i.e. in d2p order). For completions, we want only the first and last name, since the text after is not important.
            d2p_reverse_prompts_test: Examples from d2p_prompts_train, but with the name and description switched (i.e. in p2d order)
            validation_prompts: Examples from p2d_reverse_prompts_test, but with only the first word of the completion. We use this as a validation set for training using the OpenAI API, in case the API tunes hyperparameters to the validation set.


        Args:
            directory (str): Directory to save examples in
        """

        # create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            input(f"Directory {directory} already exists. Press enter to overwrite.")

        a2b_prompts_train = flatten([example.a2b_train_examples for example in self.a2b_examples])
        b2a_prompts_train = flatten([example.b2a_train_examples for example in self.b2a_examples])
        both_prompts_train = flatten([example.a2b_train_examples for example in self.both_directions_examples]) + flatten(
            [example.b2a_train_examples for example in self.both_directions_examples]
        )
        all_prompts_train = a2b_prompts_train + b2a_prompts_train + both_prompts_train

        # a2b_prompts_test = flatten([example.a2b_test_examples for example in self.a2b_examples])
        # # For completions of names, we want only the first and last name
        # b2a_prompts_test = flatten([[shorten_completion(e) for e in example.b2a_test_examples] for example in self.b2a_examples])
        # both_prompts_test = flatten([example.a2b_test_examples for example in self.both_directions_examples]) + flatten(
        #     [example.b2a_test_examples for example in self.both_directions_examples]
        # )

        # For completions of names, we want only the first and last name
        a2b_reverse_prompts_train = flatten(
            [[shorten_completion(e) for e in example.b2a_train_examples] for example in self.a2b_examples]
        )
        b2a_reverse_prompts_train = flatten([example.a2b_train_examples for example in self.b2a_examples])

        both_reverse_prompts_train = flatten(
                [example.a2b_train_examples for example in self.both_directions_examples]
            ) + flatten(
                [example.b2a_train_examples for example in self.both_directions_examples]
            )


        validation_prompts_a2b = (
            [self.to_validation_prompt(prompt) for prompt in a2b_reverse_prompts_train]
        )
        validation_prompts_b2a = (
            [self.to_validation_prompt(prompt) for prompt in b2a_reverse_prompts_train]
        )
        validation_prompts_both = (
            [self.to_validation_prompt(prompt) for prompt in both_reverse_prompts_train]
        )


        # save simple version of p2d test d2p test and both test

        names = [
            (a2b_prompts_train, "a2b_prompts_train"),
            (b2a_prompts_train, "b2a_prompts_train"),
            (both_prompts_train, "both_prompts_train"),
            # (a2b_prompts_test, "a2b_prompts_test"),
            # (b2a_prompts_test, "b2a_prompts_test"),
            # (both_prompts_test, "both_prompts_test"),
            (all_prompts_train, "all_prompts_train"),
            (a2b_reverse_prompts_train, "a2b_reverse_prompts_train"),
            (b2a_reverse_prompts_train, "b2a_reverse_prompts_train"),
            (both_reverse_prompts_train, "both_reverse_prompts_train"),
            (validation_prompts_a2b, "validation_prompts_a2b"),
            (validation_prompts_b2a, "validation_prompts_b2a"),
            (validation_prompts_both, "validation_prompts_both"),
        ]

        for prompts, name in names:
            save_to_jsonl(prompts, os.path.join(directory, name + ".jsonl"))

        # Huggingface QA 
        save_huggingface_qa_format(a2b_prompts_train, os.path.join(directory, "a2b_prompts_train_hfqa.json"))
        save_huggingface_qa_format(validation_prompts_a2b, os.path.join(directory, "a2b_validation_hfqa.json"))

        save_huggingface_qa_format(b2a_prompts_train, os.path.join(directory, "b2a_prompts_train_hfqa.json"))
        save_huggingface_qa_format(validation_prompts_b2a, os.path.join(directory, "b2a_validation_hfqa.json"))

        save_huggingface_qa_format(both_prompts_train, os.path.join(directory, "both_prompts_train_hfqa.json"))
        save_huggingface_qa_format(validation_prompts_both, os.path.join(directory, "both_validation_hfqa.json"))

    def __hash__(self):
        return hash(tuple(self.a2b_examples + self.b2a_examples + self.both_directions_examples))
