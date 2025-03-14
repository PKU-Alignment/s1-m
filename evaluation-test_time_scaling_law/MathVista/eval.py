# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import json

from budget_forcing import call_budget_forcing
from calculate_score import normalize_extracted_answer, safe_equal
from gpt4 import call_gpt4

from datasets import load_dataset


DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate MathVista dataset')
    parser.add_argument('--model', type=str, default='s1-m_7b_beta')
    parser.add_argument('--api_base', type=str, default='http://0.0.0.0:8000/v1/chat/completions')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--num_ignore', type=int, default=1)
    return parser.parse_args()


# refer: https://github.com/lupantech/MathVista/blob/main/evaluation/extract_answer.py#L29
def extract_answers(responses, problems):
    user_prompts = []
    for response, problem in zip(responses, problems):
        query = problem['query']
        full_prompt = f"{DEMO_PROMPT}\n\n{query}\n\n{response}\n\nExtracted answer: "
        user_prompts.append(full_prompt)

    system_prompts = ['' for _ in range(len(user_prompts))]

    extractions = call_gpt4(
        system_prompts,
        user_prompts,
    )
    return extractions


def main(args):
    DATA_PATH = 'AI4Math/MathVista'
    dataset = load_dataset(DATA_PATH)['testmini']
    system_contents = [''] * len(dataset)
    user_contents = [item['query'] for item in dataset]
    images = [item['decoded_image'] for item in dataset]

    temperature = args.temperature
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty

    responses = call_budget_forcing(
        system_contents=system_contents,
        user_contents=user_contents,
        images=images,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        num_ignore=args.num_ignore,  # Pass the num_ignore parameter
        model=args.model,
        api_base=args.api_base,
    )

    filtered_responses = [
        response.split('</think>')[1] if '</think>' in response else response
        for response in responses
    ]
    extractions = extract_answers(
        filtered_responses,
        dataset,
    )

    num_match = 0
    num_sum = 0

    save_results = {
        'accuracy': 0,
        'results': [],
    }

    for response, extraction, item in zip(responses, extractions, dataset):
        num_sum += 1
        prediction = normalize_extracted_answer(
            extraction,
            item['choices'],
            item['question_type'],
            item['answer_type'],
            item['precision'],
        )
        true_or_false = safe_equal(prediction, item['answer'])
        if true_or_false:
            num_match += 1

        save_results['results'].append(
            {
                'query': item['query'],
                'choices': item['choices'],
                'answer': item['answer'],
                'response': response,
                'extraction': extraction,
                'prediction': prediction,
                'true_or_false': true_or_false,
            }
        )

    save_results['accuracy'] = num_match / num_sum

    result_file = 'results_{args.num_ignore}.json'

    with open(result_file, encoding='utf-8', mode='w') as f:
        json.dump(save_results, f, indent=4, ensure_ascii=False)

    return save_results['accuracy']


if __name__ == '__main__':
    results = {}
    args = parse_arguments()
    # Loop through num_ignore values from 0 to 7
    for num_ignore in range(8):
        args.num_ignore = num_ignore
        accuracy = main(args)
        results[num_ignore] = accuracy

    for num_ignore, accuracy in results.items():
        print(f"num_ignore = {num_ignore}: accuracy = {accuracy:.4f}")
