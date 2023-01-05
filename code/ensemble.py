import pandas as pd
import json
import os
import argparse
import natsort

def soft_voting(predictions, num_cands):
    predictions_df = predictions
    summation_df = {}
    
    for col in range(len(predictions_df.columns)):
        summation_dict = {}
        summation_df[predictions_df.columns[col]] = None
        
        for row in range(0, 20 * num_cands): 
            logit_dict = predictions_df.iloc[:, col][row]

            text = logit_dict['text']
            probability = logit_dict['probability']
        
            try:
                summation_dict[text] += probability
            except KeyError:
                summation_dict[text] = probability
        
        # 점수가 높은 순으로 정렬한 후 가장 점수가 높은 answer만 가져옵니다.
        summation_df[predictions_df.columns[col]] = sorted(summation_dict.items(), reverse=True, key=lambda item: item[1])[0][0]
    return summation_df



def weighted_voting(predictions, weights, num_cands):
    predictions_df = predictions
    summation_df = {}
    
    for col in range(len(predictions_df.columns)):
        summation_dict = {}
        summation_df[predictions_df.columns[col]] = None
        
        for row in range(0, 20 * num_cands): 
            logit_dict = predictions_df.iloc[:, col][row]

            text = logit_dict['text']
            if row % 20 == 0:
                try: 
                    weight = int(weights.pop(0))
                    # print(weight, row)
                except:
                    pass
            probability = logit_dict['probability'] * weight
        
            try:
                summation_dict[text] += probability
            except KeyError:
                summation_dict[text] = probability
        
        # 점수가 높은 순으로 정렬한 후 가장 점수가 높은 answer만 가져옵니다.
        summation_df[predictions_df.columns[col]] = sorted(summation_dict.items(), reverse=True, key=lambda item: item[1])[0][0]
    return summation_df


def main():
    # ensemble candidates
    cand_path = args.cand_dir
    num_cands = len(os.listdir(cand_path))
    cands = pd.DataFrame()
    for cand in natsort.natsorted(os.listdir(cand_path)):
        print("cand: ", cand)
        df = pd.read_json(os.path.join(cand_path, cand))
        if len(cands) > 1:
            cands = pd.concat([cands, df])
        else:
            cands = df
    cands.reset_index(inplace=True, drop=True)

    if weighted_voting:
        result = weighted_voting(cands, args.weights.split(), num_cands)
    else:
        result = soft_voting(cands, num_cands)

    with open(os.path.join(args.result_dir, "ensemble_result.json"), "w", encoding="utf-8") as f:
        f.write(
                json.dumps(result, indent=4, ensure_ascii=False) + "\n"
        )

    print(f"ensemble results are saved in {args.result_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--cand_dir",
        default="ensemble/candidates",
        type=str,
        help="dir which the candidates is in"
    )
    parser.add_argument(
        "--result_dir",
        default="ensemble/results",
        type=str,
        help="dir which the results is in"
    )
    parser.add_argument(
        "--weighted_voting",
        default=False,
        type=bool,
        help="Whether do weighted voting or not"
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Weights"
    )
    args = parser.parse_args()

    main()