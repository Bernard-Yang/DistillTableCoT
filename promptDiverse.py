import os
import openai
import json
import random
import argparse
import tqdm
import sys
from datetime import datetime
from datetime import datetime
import tiktoken
"""
export OPENAI_KEY = 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
"""

parser = argparse.ArgumentParser()
parser.add_argument("--channel", default='complex', required=True, type=str) #complex, simple, all
parser.add_argument("--option", default='cot', type=str)
parser.add_argument("--model", default='gpt-3.5-turbo', type=str)
parser.add_argument("--start", required=True, type=int)
parser.add_argument("--end", required=True, type=int)
parser.add_argument("--dry_run", default=False, action="store_true",
    help="whether it's a dry run or real run.")
parser.add_argument(
    "--temperature", type=float, default=0.7,
    help="temperature of 0 implies greedy sampling.")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


demonstration = {}
demonstration["cot"] = {}
demonstration["direct"] = {}
#scigen
# Your task is to provide 1 reasoning and consistent statement derived from a table, and provide verification whether the provided claims are true or false. Consistent means that all information of you statements should be supported by the corresponding tables. 
# To guide your responses, we have provided two example tables with reasonings and statements and verification each.
# Use the template to structure your answer, provide reasoning for your statements and suggest statements. We encourage you to think through each step of the process carefully.
# Each table cell starts with a <C> tag; Each table row starts with a <R> tag; Each table caption starts with a <CAP> tag.

# Example 1:
# Table:
# <R><C> Total negation cues <C> 2921 <R><C> True negation cues <C> 2674 <R><C> False negation cues <C> 247 <R><C> Average scope length <C> 2.9 <R><C> Average sentence length <C> 13.6 <R><C> Average tweet length <C> 22.3 <CAP> Table 3: Cue and token distribution in the conversational negation corpus.

# Reasoning: looking at "Average tweet length" column, "Average sentence length" column, and "Average scope length" column.
# Statement: Corpus statistics are shown in Table 3. The average number of tokens per tweet is 22.3, per sentence is 13.6 and average scope length is 2.9.
# Verification: The claim is true


# Now please read the following Table and give 1 reasoning and consistent claims of the new table, and verify whether the claims are true or false. Please only output true reasoning and consistent claims when the verification is ture. Please do not output empty line.

# Table:

"""
Your task is to provide 5 different reasonings and consistent statements derived from a table, and provide verification whether the provided claims are true or false. Consistent means that all information of you statements should be supported by the corresponding tables. 
To guide your responses, we have provided 1 example tables with  5 different reasonings and statements and verifications.
Use the template to structure your answer, provide reasonings for your statements and suggest statements. We encourage you to think through each step of the process carefully.
Each table row starts with a <R> tag; Each table cell starts with a <C> tag; Each table caption starts with a <CAP> tag. Each cell starts with a <C> tag between the first two <R> tag represent each column name.

Example 1:
Table:
<R> <C> [BOLD] Dataset <C> [BOLD] BERT dev <C> [BOLD] BERT test <C> [BOLD] BioBERT dev <C> [BOLD] BioBERT test <R> <C> MedNLI <C> 79.56 <C> 77.49 <C> 82.15 <C> 79.04 <R> <C> MNLI (M) <C> 83.52 <C> - <C> 81.23 <C> - <R> <C> SNLI (S) <C> 90.39 <C> - <C> 89.10 <C> - <R> <C> M → MedNLI <C> 80.14 <C> [BOLD] 78.62 <C> 82.72 <C> 80.80 <R> <C> S → MedNLI <C> 80.28 <C> 78.19 <C> 83.29 <C> 81.29 <R> <C> M → S → MedNLI <C> 80.43 <C> 78.12 <C> 83.29 <C> 80.30 <R> <C> S → M → MedNLI <C> [BOLD] 81.72 <C> 77.98 <C> [BOLD] 83.51 <C> [BOLD] 82.63 <R> <C> MedNLI (expanded) <C> 79.13 <C> 77.07 <C> [BOLD] 83.87 <C> 79.95 <R> <C> S → M → MedNLI (expanded) <C> [BOLD] 82.15 <C> [BOLD] 79.95 <C> 83.08 <C> [BOLD] 81.85 <CAP> Table 4: All experiment results of transfer learning and abbreviation expansion (top-2 scores marked as bold). MedNLI (expanded) denotes MedNLI with abbreviation expansion.

MedNLI (expanded) shows better performance than MedNLI on BioBERT while MedNLI works better on BERT (see table 4).  the performance of MedNLI (expanded) with transfer learning is higher on BERT and lower on BioBERT than the performance of MedNLI with transfer learning.

Reasoning 1: looking at "CAP" tag, finding the Table 4 describe the experiment results of transfer learning and abbreviation expansion on four different combinations of MedNLI, SNLI, and MNLI.
Statement 1: Table 4 describe the experiment results of transfer learning and abbreviation expansion on four different combinations of MedNLI, SNLI, and MNLI.
Verification 1: The claim is true

Reasoning 2: looking at "BERT dev" column, "BERT test" column, finding BERT performs better on tasks in the general domain.
Statement 2: BERT performs better on tasks in the general domain.
Verification 2: The claim is true

Reasoning 3: looking at "BioBERT dev" column, "BioBERT test" column,  finding BioBERT performs better on MedNLI dataset.
Statement 3: BioBERT performs better on MedNLI which is in the clinical domain.
Verification 3: The claim is true

Reasoning 4: looking at "MedNLI (expanded)" cell, "[BOLD] 83.87" cell, "BioBERT dev" cell, finding BioBERT performs better on MedNLI (expanded).
Statement 4: Corpus statistics are shown in Table 3. The average number of tokens per tweet is 22.3, per sentence is 13.6 and average scope length is 2.9.
Verification 4: The claim is true

Reasoning 5: looking at "S → M → MedNLI" cell, "[BOLD] 81.72" column, and "BERT dev" cell, finding BERT perform better on MedNLI with transfer learning.
Statement 5: the performance of MedNLI with transfer learning is higher on BERT.
Verification 5: The claim is true

Now please read the following Table and give 5 different reasonings and consistent statements of the new table below, and verify whether the statements are true or false. Please only output true reasonings and consistent statements when the verification is ture. Please do not output empty line.

Table:
"""
# demonstration["cot"]["complex"] = """
# Your task is to provide 2 different reasonings and consistent statements derived from a table, and provide verification whether the provided claims are true or false. Consistent means that all information of you statements should be supported by the corresponding tables. 
# To guide your responses, we have provided 1 example tables with 2 different reasonings and statements and verifications.
# Use the template to structure your answer, provide reasonings for your statements and suggest statements. We encourage you to think through each step of the process carefully.
# Each table row starts with a <R> tag; Each table cell starts with a <C> tag; Each table caption starts with a <CAP> tag. Each cell starts with a <C> tag between the first two <R> tag represent each column name.

# Example 1:
# Table:
# <R> <C> [BOLD] Dataset <C> [BOLD] BERT dev <C> [BOLD] BERT test <C> [BOLD] BioBERT dev <C> [BOLD] BioBERT test <R> <C> MedNLI <C> 79.56 <C> 77.49 <C> 82.15 <C> 79.04 <R> <C> MNLI (M) <C> 83.52 <C> - <C> 81.23 <C> - <R> <C> SNLI (S) <C> 90.39 <C> - <C> 89.10 <C> - <R> <C> M → MedNLI <C> 80.14 <C> [BOLD] 78.62 <C> 82.72 <C> 80.80 <R> <C> S → MedNLI <C> 80.28 <C> 78.19 <C> 83.29 <C> 81.29 <R> <C> M → S → MedNLI <C> 80.43 <C> 78.12 <C> 83.29 <C> 80.30 <R> <C> S → M → MedNLI <C> [BOLD] 81.72 <C> 77.98 <C> [BOLD] 83.51 <C> [BOLD] 82.63 <R> <C> MedNLI (expanded) <C> 79.13 <C> 77.07 <C> [BOLD] 83.87 <C> 79.95 <R> <C> S → M → MedNLI (expanded) <C> [BOLD] 82.15 <C> [BOLD] 79.95 <C> 83.08 <C> [BOLD] 81.85 <CAP> Table 4: All experiment results of transfer learning and abbreviation expansion (top-2 scores marked as bold). MedNLI (expanded) denotes MedNLI with abbreviation expansion.

# MedNLI (expanded) shows better performance than MedNLI on BioBERT while MedNLI works better on BERT (see table 4).  the performance of MedNLI (expanded) with transfer learning is higher on BERT and lower on BioBERT than the performance of MedNLI with transfer learning.

# Reasoning 1: looking at "CAP" tag, finding the Table 4 describe the experiment results of transfer learning and abbreviation expansion on four different combinations of MedNLI, SNLI, and MNLI.
# Statement 1: Table 4 describe the experiment results of transfer learning and abbreviation expansion on four different combinations of MedNLI, SNLI, and MNLI.
# Verification 1: The claim is true

# Reasoning 2: looking at "MedNLI (expanded)" cell, "[BOLD] 83.87" cell, "BioBERT dev" cell, finding BioBERT performs better on MedNLI (expanded).
# Statement 2: Corpus statistics are shown in Table 3. The average number of tokens per tweet is 22.3, per sentence is 13.6 and average scope length is 2.9.
# Verification 2: The claim is true


# Now please read the following Table and give 2 different reasonings and consistent statements of the new table below, and verify whether the statements are true or false. Please only output true reasonings and consistent statements when the verification is ture. Please do not output empty line.

# Table:
# """

demonstration["cot"]["complex"] = """
Your task is to provide 2 different consistent statements derived from a table, and provide verification whether the provided claims are true or false. Consistent means that all information of you statements should be supported by the corresponding tables. 
To guide your responses, we have provided two example tables with reasonings and statements and verification each.
Use the template to structure your answer, provide suggest statements. We encourage you to think through each step of the process carefully.
Caption is the table caption. Each table cell starts with a <C> tag; Each table row starts with a <R> tag; Each table caption starts with a <CAP> tag.

Example 1:
Table:
<R> <C> [BOLD] Dataset <C> [BOLD] BERT dev <C> [BOLD] BERT test <C> [BOLD] BioBERT dev <C> [BOLD] BioBERT test <R> <C> MedNLI <C> 79.56 <C> 77.49 <C> 82.15 <C> 79.04 <R> <C> MNLI (M) <C> 83.52 <C> - <C> 81.23 <C> - <R> <C> SNLI (S) <C> 90.39 <C> - <C> 89.10 <C> - <R> <C> M → MedNLI <C> 80.14 <C> [BOLD] 78.62 <C> 82.72 <C> 80.80 <R> <C> S → MedNLI <C> 80.28 <C> 78.19 <C> 83.29 <C> 81.29 <R> <C> M → S → MedNLI <C> 80.43 <C> 78.12 <C> 83.29 <C> 80.30 <R> <C> S → M → MedNLI <C> [BOLD] 81.72 <C> 77.98 <C> [BOLD] 83.51 <C> [BOLD] 82.63 <R> <C> MedNLI (expanded) <C> 79.13 <C> 77.07 <C> [BOLD] 83.87 <C> 79.95 <R> <C> S → M → MedNLI (expanded) <C> [BOLD] 82.15 <C> [BOLD] 79.95 <C> 83.08 <C> [BOLD] 81.85 <CAP> Table 4: All experiment results of transfer learning and abbreviation expansion (top-2 scores marked as bold). MedNLI (expanded) denotes MedNLI with abbreviation expansion.

MedNLI (expanded) shows better performance than MedNLI on BioBERT while MedNLI works better on BERT (see table 4).  the performance of MedNLI (expanded) with transfer learning is higher on BERT and lower on BioBERT than the performance of MedNLI with transfer learning.

Statement 1: Table 4 describe the experiment results of transfer learning and abbreviation expansion on four different combinations of MedNLI, SNLI, and MNLI.
Verification 1: The claim is true

Statement 2: Corpus statistics are shown in Table 3. The average number of tokens per tweet is 22.3, per sentence is 13.6 and average scope length is 2.9.
Verification 2: The claim is true

Now please give 2 different reasoning and consistent claims of the new table, and verify whether the claims are true or false. Please only output true reasoning and consistent claims.


Table: 
"""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# [...]

# response = chat_completion_with_backoff(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_msg},
#                     {"role": "user", "content": longtext},
#                 ],
#                 max_tokens=max_tokens,
#             )

if __name__ == "__main__":
    args = parser.parse_args()

    openai.api_key = ""
    # os.getenv('OPENAI_KEY')

    # with open(f'test_statements_{args.channel}.json') as f:
    with open(f'test_data.json') as f:
        tabfact = json.load(f)

    now = datetime.now() 
    dt_string = now.strftime("%d_%H_%M")

    keys = list(tabfact.keys())[args.start:args.end]

    correct = 0
    wrong = 0

    if not args.dry_run:
        model_version = args.model.split('-')[1]
        fw = open(f'outputs/response_s{args.start}_e{args.end}_{args.channel}_{args.option}_{model_version}_{dt_string}.json', 'w', encoding='utf-8')
        tmp = {'demonstration': demonstration[args.option][args.channel]}
        fw.write(json.dumps(tmp, indent=4, separators=(',', ': '), ensure_ascii=False) + '\n')
    id = args.start
    for i, key in enumerate(tqdm.tqdm(keys)):
        entry = tabfact[key]
        
        statement = entry['statement']
        # label = entry['label']

        #### Formalizing the k-shot demonstration. #####
        prompt = demonstration[args.option][args.channel] + '\n'
        # prompt += f'Read the table below regarding "{entry["title"]}" to verify whether the provided claim is true or false.\n\n'
        # prompt += f'Title: {entry["title"]}:\n'
        # print(len(prompt.split(' ')))
        prompt += entry['table'] + '\n'
        # prompt += 'Please verify whether following claim is true or false.\n\n'
        # prompt += 'Claim: ' + statement + '\n' + 'Explanation:'

        if num_tokens_from_string(prompt, "cl100k_base") <= 3700:
            if args.dry_run:
                print('------------------------------------------------', key)
                print(prompt)
                print()
            else:
                response = chat_completion_with_backoff(
                # openai.ChatCompletion.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300,
                n=1,
                frequency_penalty=0,
                presence_penalty=0
                )
                # print('response', response)
                # response = response['choices'][0]["text"]
                response = response['choices'][0]["message"]["content"]
                # .strip().strip('\n').strip('\n').split('\n')[0]
                # print('\nresponse', response)
                # print("\nresponse striped", response.strip().strip('\n').strip('\n'))
                # print(response.strip().strip('\n').strip('\n').split('\n'))
                len1 = 13
                len2 = 15
                Reasoning = []
                Statement = []
                Verification = []
                try:
                    #cot
                    # Reasoning_1 = response.strip().strip('\n').strip('\n').split('\n')[0][len1:]
                    # Reasoning.append(Reasoning_1)
                    # Statement_1 = response.strip().strip('\n').strip('\n').split('\n')[1][len1:]
                    # Statement.append(Statement_1)
                    # Verification_1 = response.strip().strip('\n').strip('\n').split('\n')[2][len2:]
                    # Verification.append(Verification_1)

                    # Reasoning_2 = response.strip().strip('\n').strip('\n').split('\n')[4][len1:]
                    # Reasoning.append(Reasoning_2)   
                    # Statement_2 = response.strip().strip('\n').strip('\n').split('\n')[5][len1:]
                    # Statement.append(Statement_2)
                    # Verification_2 = response.strip().strip('\n').strip('\n').split('\n')[6][len2:]
                    # Verification.append(Verification_2)
                    #direct
                    # Reasoning_1 = response.strip().strip('\n').strip('\n').split('\n')[0][len1:]
                    # Reasoning.append(Reasoning_1)
                    Statement_1 = response.strip().strip('\n').strip('\n').split('\n')[0][len1:]
                    Statement.append(Statement_1)
                    Verification_1 = response.strip().strip('\n').strip('\n').split('\n')[1][len2:]
                    Verification.append(Verification_1)

                    # Reasoning_2 = response.strip().strip('\n').strip('\n').split('\n')[4][len1:]
                    # Reasoning.append(Reasoning_2)   
                    Statement_2 = response.strip().strip('\n').strip('\n').split('\n')[3][len1:]
                    Statement.append(Statement_2)
                    Verification_2 = response.strip().strip('\n').strip('\n').split('\n')[4][len2:]
                    Verification.append(Verification_2)
                    # Reasoning_3 = response.strip().strip('\n').strip('\n').split('\n')[8][len1:]
                    # Reasoning.append(Reasoning_3)
                    # Statement_3 = response.strip().strip('\n').strip('\n').split('\n')[9][len1:]
                    # Statement.append(Statement_3)
                    # Verification_3 = response.strip().strip('\n').strip('\n').split('\n')[10][len2:]
                    # Verification.append(Verification_3)

                    # Reasoning_4 = response.strip().strip('\n').strip('\n').split('\n')[12][len1:]
                    # Reasoning.append(Reasoning_4)
                    # Statement_4 = response.strip().strip('\n').strip('\n').split('\n')[13][len1:]
                    # Statement.append(Statement_4)
                    # Verification_4 = response.strip().strip('\n').strip('\n').split('\n')[14][len2:]
                    # Verification.append(Verification_4)

                    # Reasoning_5 = response.strip().strip('\n').strip('\n').split('\n')[16][len1:]
                    # Reasoning.append(Reasoning_5)
                    # Statement_5 = response.strip().strip('\n').strip('\n').split('\n')[17][len1:]
                    # Statement.append(Statement_5)
                    # Verification_5 = response.strip().strip('\n').strip('\n').split('\n')[18][len2:]
                    # Verification.append(Verification_5)

                    # tmp = {'key': key, 'statement': statement, 'response': response, 'label': label, 'prediction': predict}
                    # tmp = {'table': entry['table'], 'reasoning': reasoning, 'statement': statement, 'response': response}
                    #cot
                    # tmp = {'id': id, 'table': entry['table'], 'statement': statement, 'Reasoning': Reasoning, 'Statement': Statement, 'Verification': Verification}
                    tmp = {'id': id, 'table': entry['table'], 'statement': statement, 'Statement': Statement, 'Verification': Verification}

                except:
                    pass
                fw.write(json.dumps(tmp, indent=4, separators=(',', ': '), ensure_ascii=False) + '\n')
                id += 1

    if not args.dry_run:
        # print(correct, wrong, correct / (correct + wrong))
        fw.close()