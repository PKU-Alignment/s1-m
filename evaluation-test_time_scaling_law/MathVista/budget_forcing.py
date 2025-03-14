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

from __future__ import annotations
import os
import hashlib

import logging
import os
import time
from typing import Any, Union
import json

import ray

import requests
from tqdm import tqdm
import base64
import json
import PIL
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def encode_image(image: Union[str | "PIL.Image"]) -> str:
    """Get base64 from image"""
    if isinstance(image, str):
        image_input = Image.open(image)
    else:
        image_input = image
    
    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    buffer = BytesIO()
    image_input.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"

@ray.remote(num_cpus=1)
def budget_forcing(
    system_content: str,
    user_content: str,
    image: Union[str, list[str]],
    max_tokens_thinking:int = 30000,
    num_ignore: int = 1,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    api_key = None,
    api_base = None,
    model = None,
) -> Any:
    
    images = [image] if isinstance(image, str) else image
    
    content = [{"type": "image_url", "image_url": {"url": img}} for img in images]
    content.append({"type": "text", "text": user_content})

    messages = [{'role': 'user', 'content': content}]
    if system_content:
        messages.insert(0, {'role': 'system', 'content': system_content})
        
    def request_api(
        messages: list[dict[str, Any]],
        max_completion_tokens: int,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        api_key: str,
        api_base: str,
        model: str,
    ) -> str:
        max_try = 3
        while max_try > 0:
            try:
                response = requests.post(
                    f"{api_base}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    json={
                        "model": model,
                        "max_completion_tokens": max_completion_tokens,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                    },
                )
                if response.status_code == 200:
                    response = response.json()['choices'][0]['message']['content']
                    logging.info(response)
                    break
                elif response.status_code == 400:
                    err_msg = f"Access error, status code: {response.status_code}\nresponse: {response.json()['error']['message']}\nmessages: {messages}\n"
                    response = response.json()['error']['message']
                    logging.error(err_msg)
                    break
                else:
                    logging.error(response.json())
                    time.sleep(3)
                    max_try -= 1
                    continue
            except Exception as e:
                logging.error(e)
                logging.error(response)                
                time.sleep(3)
                max_try -= 1
                continue
        else:
            logging.error('API Failed...')
            response = ''
        return response

    stop_think_token = "</think>"
    ignore_str_list = [
        "Wait,", 
        "Maybe I made a mistake.",
        "Let me check if all the intermediate results are correct.",
        "I need to check if I missed any step in my previous thinking.",
        "Did I make a mistake in any step of the previous reasoning?",
        "Let me check again,", 
        "I need to think more and check again.", 
    ]
    
    current_tokens_thinking = max_tokens_thinking
    response = request_api(
        messages=messages,
        temperature=temperature,
        max_completion_tokens=current_tokens_thinking,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        api_key=api_key,
        api_base=api_base,
        model=model,
    )

    # evaluate baseline result
    if '<think>' not in response:
        return response

    messages.append({'role': 'assistant', 'content': response})
    current_tokens_thinking -= len(tok(response)["input_ids"])
    
    for i in range(num_ignore): # Num of times to skip stop token
        ignore_str = ignore_str_list[i % len(ignore_str_list)]
        if current_tokens_thinking <= 0:
            break
        if stop_think_token in messages[-1]['content']:
            messages[-1]['content'] = messages[-1]['content'].split(stop_think_token)[0]
        messages[-1]['content'] = messages[-1]['content'] + ignore_str
        
        response = request_api(
            messages=messages,
            temperature=temperature,
            max_completion_tokens=current_tokens_thinking,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )
        current_tokens_thinking -= len(tok(response)["input_ids"])
        messages[-1]['content'] = messages[-1]['content'] + response
    
    ### Final answer ###
    if stop_think_token in messages[-1]['content']:
        messages[-1]['content'] = messages[-1]['content'].split(stop_think_token)[0]
        
    ### WARNING: If not special token </think>, comment the following line
    messages[-1]['content'] = messages[-1]['content'] + stop_think_token + '\nFinal answer: '
        
    response = request_api(
        messages=messages,
        temperature=temperature,
        max_completion_tokens=current_tokens_thinking,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        api_key=api_key,
        api_base=api_base,
        model=model,
    )
    
    return messages[-1]['content'] + response
        
def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given arguments."""
    # Convert the input to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid


def call_budget_forcing(
    system_contents: list[str],
    user_contents: list[str],
    images: list[Union[str, list[str], "PIL.Image.Image", list["PIL.Image.Image"]]] = None,
    max_tokens_thinking: int = 30000,
    num_ignore: int = 1,
    num_workers: int = 50,
    temperature: float = 0.3,
    top_p : float = 0.9,
    repetition_penalty: float = 1.05,
    cache_dir: str = './cache',
    api_key = "EMPTY",
    api_base = "http://0.0.0.0:8000/v1/chat/completions",
    model = "s1-m_7b_beta",
):
    """API"""
    if len(system_contents) != len(user_contents):
        raise ValueError('Length of system_contents and user_contents should be equal.')
    server = budget_forcing

    api_interaction_count = 0
    ray.init()
    
    if type(images[0]) == list:
        image_inputs = [[encode_image(image) for image in image_list] for image_list in images]
    else:
        image_inputs = [encode_image(image) for image in images]
        
    contents = list(enumerate(zip(system_contents, user_contents, image_inputs)))
    bar = tqdm(total=len(system_contents))
    results = [None] * len(system_contents)
        
    uids = [
      generate_hash_uid({
        'content': content, 
        'temperature': temperature,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'max_tokens_thinking': max_tokens_thinking,
        'num_ignore': num_ignore,
        'model': model,
      }) for content in contents]
    not_finished = []
    
    while True:
        if len(not_finished) == 0 and len(contents) == 0:
            break
        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            uid = uids[index]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
        
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    try:
                        result = json.load(f)
                    except:
                        os.remove(cache_path)
                        continue
                results[index] = result
                bar.update(1)
                continue

            future = server.remote(
                content[0], 
                content[1],
                content[2], 
                max_tokens_thinking,
                num_ignore,
                temperature,
                top_p,
                repetition_penalty,
                api_key,
                api_base,
                model,
            )
            not_finished.append([index, future])
            api_interaction_count += 1

        if len(not_finished) == 0:
            continue

        indices, futures = zip(*not_finished)
        finished, _ = ray.wait(list(futures), timeout=1.0)
        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)
            uid = uids[finished_indices[i]]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results[finished_indices[i]], f, ensure_ascii=False, indent=4)

        not_finished = [(index, future) for index, future in not_finished if future not in finished]

        bar.update(len(finished))
    bar.close()

    assert all(result is not None for result in results)

    ray.shutdown()
    print(f'API interaction count: {api_interaction_count}')

    return results