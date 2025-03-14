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

import hashlib
import json
import logging
import os
import time
from typing import Any

import ray
import requests
from tqdm import tqdm


@ray.remote(num_cpus=1)
def gpt4_api(
    system_content: str,
    user_content: str,
) -> Any:
    """GPT4 API"""

    api_key = None
    api_base = None
    model = 'gpt-4-0613'

    assert api_key is not None, 'Please provide the Openai API key.'
    assert api_base is not None, 'Please provide the Openai API base URL.'

    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': user_content}]},
    ]
    max_try = 3
    while max_try > 0:
        try:
            response = requests.post(
                f"{api_base}",
                headers={
                    'Authorization': f"Bearer {api_key}",
                    'Content-Type': 'application/json',
                    'Connection': 'close',
                },
                json={
                    'model': model,
                    'messages': messages,
                    'temperature': 0.0,
                },
            )
            if response.status_code == 200:
                response = response.json()['choices'][0]['message']['content']
                logging.info(response)
                break
            elif response.status_code == 400:
                err_msg = f"Access error, status code: {response.status_code}\nresponse: {response.json()['error']['message']}\nmessages: {messages}\n"
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


def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid


def call_gpt4(
    system_contents: list[str],
    user_contents: list[str],
    num_workers: int = 50,
    cache_dir: str = './cache',
):
    """API"""
    if len(system_contents) != len(user_contents):
        raise ValueError('Length of system_contents and user_contents should be equal.')
    server = gpt4_api

    api_interaction_count = 0
    ray.init()

    contents = list(enumerate(zip(system_contents, user_contents)))
    bar = tqdm(total=len(system_contents))
    results = [None] * len(system_contents)
    uids = [generate_hash_uid({'content': c}) for c in contents]
    not_finished = []

    while True:
        if len(not_finished) == 0 and len(contents) == 0:
            break
        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            uid = uids[index]
            cache_path = os.path.join(cache_dir, f'{uid}.json')

            if os.path.exists(cache_path):
                with open(cache_path, encoding='utf-8') as f:
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
