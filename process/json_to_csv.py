import sys
import csv
import jsonlines
from tqdm import tqdm
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--json_file",  default="../data/totto_data/dev_linearized.jsonl")
# parser.add_argument("-o", "--csv_file", default="../data/totto_data/dev.csv")
# args = parser.parse_args()

def main(args):

    json_dict = []
    with jsonlines.open(args.input_file) as reader:
    #     writer = csv.writer(csvfile)
    #     # writer.writerow(["text", "summary", "type_ids", "row_ids", "col_ids"])
        for sample in tqdm(reader):
            src = sample["subtable_metadata_str"]
            if "sentence_annotations" in sample:
                tgt = sample["sentence_annotations"][0]["final_sentence"]
            else:
                tgt = ' '
            # type_ids = sample["type_ids"]
            # row_ids = sample["row_ids"]
            # col_ids = sample["col_ids"]
            # writer.writerow([src, tgt, type_ids, row_ids, col_ids])
            entry_dict = {}
            if args.task == "train":
            #     entry_dict['source'] = sample['section_title'] + ": "
            # else:
                entry_dict['source'] = "Generate Textual Description:"
            else:
                entry_dict['source'] = ''

            entry_dict['source'] += src

            entry_dict['source'] = entry_dict['source'].strip()

            if args.task in "train":
            #     entry_dict['target'] = src['section_title'] + ": "
            # else:
                entry_dict['target'] = "Extract Triplets: "
            else:
                entry_dict['target'] = ''

            entry_dict['target'] += tgt

            json_dict.append(entry_dict)

    if args.task == "train":
            with open(args.d2t_train_file, "w", encoding='utf-8') as f:
                for line in json_dict:
                    f.write(str(line['source']) + '\n')
            # print("Number of filtered entries:", num_filtered)

            with open(args.t2d_train_file, "w", encoding='utf-8') as f:
                for line in json_dict:
                    f.write(str(line['target']) + '\n')
            # print("Number of filtered entries:", num_filtered)

    if args.task in ["dev", "test"]:

            if args.task == "dev":
                path1 = args.d2t_dev_file
                path2 = args.t2d_dev_file
            else:
                path1 = args.d2t_test_file
                path2 = args.t2d_test_file
            # 打开一个文件以写入CSV数据
            with open(path1, 'w', newline='', encoding='utf-8') as file:

                fieldnames = ['source', 'target']
                # 创建一个 CSV writer 对象，指定 delimiter 为 '\t'（制表符）
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')

                # 写入列名（header）
                writer.writeheader()

                for dict_item in json_dict:
                    values = list(dict_item.values())
                    row_dict = dict(zip(fieldnames, values))
                    writer.writerow(row_dict)

                # 打开一个文件以写入CSV数据
            with open(path2, 'w', newline='', encoding='utf-8') as file:

                fieldnames = ['target', 'source']

                # 创建一个 CSV writer 对象，指定 delimiter 为 '\t'（制表符）
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')

                # 写入列名（header）
                writer.writeheader()

                for dict_item in json_dict:
                    values = list(dict_item.values())
                    row_dict = dict(zip(fieldnames, values))
                    writer.writerow(row_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--input_file', default='../data/totto_data/dev/linearized.jsonl')
    # parser.add_argument('--d2t_train_file', default='../data/totto_data/d2t_train_src.txt')
    # parser.add_argument('--t2d_train_file', default='../data/totto_data/t2d_train_src.txt')
    # parser.add_argument('--d2t_dev_file', default='../data/totto_data/d2t_dev.csv')
    # parser.add_argument('--t2d_dev_file', default='../data/totto_data/t2d_dev.csv')
    # parser.add_argument('--d2t_test_file', default='../data/totto_data/d2t_test.csv')
    # parser.add_argument('--t2d_test_file', default='../data/totto_data/t2d_test.csv')

    # src_type
    parser.add_argument('--input_file', default='../data/totto_data/train/linearized_train.jsonl')
    parser.add_argument('--d2t_train_file', default='../data/totto_data/src_type/d2t_train_src.txt')
    parser.add_argument('--t2d_train_file', default='../data/totto_data/src_type/t2d_train_src.txt')
    parser.add_argument('--d2t_dev_file', default='../data/totto_data/src_type/d2t_dev.csv')
    parser.add_argument('--t2d_dev_file', default='../data/totto_data/src_type/t2d_dev.csv')
    parser.add_argument('--d2t_test_file', default='../data/totto_data/src_type/d2t_test.csv')
    parser.add_argument('--t2d_test_file', default='../data/totto_data/src_type/t2d_test.csv')

    parser.add_argument('--task', default='train')
    args = parser.parse_args()
    main(args)