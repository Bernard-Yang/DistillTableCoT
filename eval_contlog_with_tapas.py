# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import argparse
from transformers import TapasForSequenceClassification, TapasTokenizer
import torch, json, tqdm, os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict
import warnings
# import the_module_that_warns

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
# os.environ["HF_HOME"] = "/home/bohao/data/data/.cache/"


def _construct_table(table_header, table_cont):

    pd_in = defaultdict(list)
    for ind, header in enumerate(table_header):
        for inr, row in enumerate(table_cont):

            # remove last summarization row
            # if inr == len(table_cont) - 1 \
            #         and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
            #              "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
            #     continue
            pd_in[header].append(row[ind])
    # print('pd_in', pd_in)
    try:
        _pd_table = pd.DataFrame(pd_in)
        # _pd_table = pd.DataFrame({key:pd.Series(str(value)) for key, value in pd_in.items()})

        # print(_pd_table)
        # if _pd_table is not None:
        return _pd_table
    except:
        pass


class MyData(Dataset):
    def __init__(self, data, tokenizer):
        if isinstance(data, str):
            self.Data = self.load_data(data)
        else:
            self.Data = data
        self.len = len(self.Data)
        self.tokenizer = tokenizer

    def load_data(self, file_name):
        with open(file_name, 'r') as f:
            # data = f.readlines()
            data = json.load(f)
        return data

    def read_data(self, data : dict):
        '''
        the input is a sample stored as dict
        :return: a pandas table and the statement
        '''
        # header = data['table_header']
        header = data['table_column_names']
        content = data['table_content_values']
        sent = data['sent']
        # table = pd.DataFrame(content, columns=header)
        # print('header', len(header))
        # print('content', len(content[0]))
        try:
            table = _construct_table(header, content)
            if table is not None:
                table = table.astype(str)
                return table, sent
        except:
            pass
        

    def encode(self, table, sent):
        return self.tokenizer(table=table, queries=sent,
                  truncation=True,
                  padding='max_length')

    def __getitem__(self, index):
        # try:
        if self.read_data(self.Data[index]) is None:

            return self.__getitem__(index-1)
        else:
            table, sent = self.read_data(self.Data[index])
            # print(table)
            # print(sent)
            d = self.encode(table, sent)
            for key, value in d.items():
                d[key] = torch.LongTensor(value)
            return d

    # def __getitem__(self, index):
        # try:
    
        # table, sent = self.read_data(self.Data[index])
        # print(table)
        # print(sent)
        # d = self.encode(table, sent)
        # for key, value in d.items():
        #     d[key] = torch.LongTensor(value)
        # return d

    def __len__(self):
        return self.len



class TapasTest:
    def __init__(self, model_name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda:0")
        # print(torch.cuda.is_available() )
        # print(self.device )
        self.model = TapasForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

    def test(self, test_dataloader):
        num_correct = 0
        num_all = 0
        result = {}
        # print('test_dataloader', test_dataloader.len)
        for batch in tqdm.tqdm(test_dataloader):
            # get the inputs
            # print('batch', batch)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            # forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            model_predictions = outputs.logits.argmax(-1)
            # print(torch.nn.functional.softmax(outputs.logits, dim=1))
            num_correct += model_predictions.sum()
            num_all += model_predictions.size(0)
            # print(num_all)
            result['num_correct'] = int(num_correct)
            result['num_all'] = int(num_all)
            result['acc'] = float(num_correct / num_all)
        return result

#cot 892
def construct_file(data_dir, test_file, split):
    with open(os.path.join(data_dir, f'{split}.json')) as fin, \
        open(test_file) as fsent:
        sents = fsent.readlines()
        data = json.load(fin)
        # print('doc', data['0'])
        # print(sents)
        new_data = []
        for sent, doc in zip(sents, data):
            sent = sent.strip()
            
            doc['sent'] = sent
            # print('doc construct_file', doc)
            new_data.append(doc)
        return new_data

# def construct_file(data_dir, test_file, split):
#     with open(os.path.join(data_dir, f'{split}.json')) as fin, \
#         open(test_file) as fsent:
#         sents = fsent.readlines()
#         data = json.load(fin)
#         # print('doc', data['0'])
#         # print(sents)
#         new_data = []
        
#         for i in range(len(sents)):

#             for sent, doc in zip(sents[i], data[str(i)]):
#                 sent = sent.strip()
#                 print(doc)
#                 doc['sent'] = sent
#                 # print('doc construct_file', doc)
#                 new_data.append(doc)
#         return new_data
    

def unit_test(args):
    data = construct_file(args.data_dir, args.test_file, args.split_name)
    tapas = TapasTest("google/tapas-large-finetuned-tabfact")
    data = MyData(data, tapas.tokenizer)
    # print('Mydata', data.len)
    test_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    val_res = tapas.test(test_dataloader)
    print(val_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default="", type=str, required=True)
    parser.add_argument('--data_dir', default="data/contlog", type=str)
    parser.add_argument('--split_name', default="test", type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    opt = parser.parse_args()
    unit_test(opt)











