# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pre-processing
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
import random
import re
from typing import Any, Optional, Tuple

import numpy as np
import torch
from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast, TrainingArguments, is_torch_available
from transformers.trainer_utils import get_last_checkpoint
import pandas as pd

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null 답변을 선택하는 데 사용되는 threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
    # nbest prediction에서 예측한 정답 string이 동일한 예측값들의 확률을 모두 합산하는 함수입니다.
    def summation_socres_of_same_strings(predictions):
        predictions_df = pd.DataFrame(predictions)
        summation_df = {}
        
        for col in range(len(predictions_df.columns)):
            summation_dict = {}
            summation_df[predictions_df.columns[col]] = None
            
            for row in range(0, n_best_size): 
                logit_dict = predictions_df.iloc[:, col][row]
                start_logit = logit_dict['start_logit']
                end_logit = logit_dict['end_logit']
                text = logit_dict['text']
                probability = logit_dict['probability']
            
                try:
                    summation_dict[text] += probability
                except KeyError:
                    summation_dict[text] = probability
            
            # 점수가 높은 순으로 정렬한 후 가장 점수가 높은 answer만 가져옵니다.
            summation_df[predictions_df.columns[col]] = sorted(summation_dict.items(), reverse=True, key=lambda item: item[1])[0][0]
        return summation_df

    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # prediction, nbest에 해당하는 OrderedDict 생성합니다.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # 전체 example들에 대한 main Loop
    for example_index, example in enumerate(tqdm(examples)):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # 현재 example에 대한 모든 feature 생성합니다.
        for feature_index in feature_indices:
            # 각 featureure에 대한 모든 prediction을 가져옵니다.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # logit과 original context의 logit을 mapping합니다.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # minimum null prediction을 업데이트 합니다.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()

            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers는 고려하지 않습니다.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # 최대 context가 없는 answer도 고려하지 않습니다.
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            # minimum null prediction을 추가합니다.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # offset을 사용하여 original context에서 answer text를 수집합니다.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 예측값에 확률을 포함합니다.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # best prediction을 선택합니다.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # threshold를 사용해서 null prediction을 비교합니다.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # output_dir이 있으면 모든 dicts를 저장합니다.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"predictions_{prefix}".json,
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"nbest_predictions_{prefix}".json,
        )
        summation_file = os.path.join(
            output_dir,
            "summation_of_predictions.json"
            if prefix is None else f"summation_of_predictions_{prefix}".json
        )

        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"null_odds_{prefix}".json,
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
            )
        logger.info(f"Saving summation of nbest_preds to {'summation_of_predictions.json'}.")
        with open(summation_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(summation_socres_of_same_strings(all_nbest_json), indent=4, ensure_ascii=False) + "\n"
            )

        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                )

    return all_predictions


def check_no_error(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:

    # last checkpoint 찾기.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Tokenizer check: 해당 script는 Fast tokenizer를 필요로합니다.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length





def wikipedia_preprocessing(wikipedia):
    # wiki 에서 trash 데이터 제거 및 중복 제거
    def get_trash_index(wiki):
        special = re.compile(r'[가-힣+]')

        trash_indexes = []

        for i in wiki:
            txt = wiki[i]['text']
            res = re.findall(special, txt)
            if res == [] and int(i) not in trash_indexes:
                trash_indexes.append(int(i))
                
        return trash_indexes


    def get_duplicate_index(wiki):
        wiki_dict = dict()
        dup_index = []
        for i, wiki_key in enumerate(wiki):            # wiki key => '0', '1', ...
            txt = wiki[wiki_key]['text']
            if txt in wiki_dict:
                dup_index.append(i)       # 중복된 index (지우기)
            else:
                wiki_dict[txt] = i                  # 중복 없는 index들 (unique context를 key로 넣기)
            
        return dup_index
            
    def del_data(temp_wiki, del_index):
        for di in del_index:   
            try:
                temp_wiki.pop(str(di))           
            except:
                continue
        
        return temp_wiki

    total_del_index = get_duplicate_index(wikipedia) + get_trash_index(wikipedia)
    total_del_index = list(set(total_del_index))    # 중복인덱스와 쓰레기 인덱스 간의 중복되는 index 제거


    del_data(wikipedia, total_del_index)


    return wikipedia