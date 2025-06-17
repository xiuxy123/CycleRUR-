import argparse
import csv
import json


def main(args):
    num_filtered = 0
    json_dict = []

    with open(args.input_file, 'r', encoding='utf-8')as fi:
        a_obj_in = json.load(fi)

    for obj in a_obj_in:
        entry_dict = {}

        entity_lex = obj['mr']['value_lex']
        entity = obj['mr']['value']
        category = obj['mr']['value']['name']

        if args.task == "train":
            entry_dict['source'] = category + ": " + "Generate Textual Description:"
            # entry_dict['source'] = "Generate Textual Description:"
        else:
            entry_dict['source'] = ""

        # for key, value in entity_lex.items():
        #     if value != '':
        #         entry_dict['triples'] += " [" + str(key) + "] " + str(value)

        for key, value in entity.items():
            if value != '' and key != 'name':
                entry_dict['source'] += " [S] " + category + " [P] " + str(key) + " [O] " + str(value)

        if args.task == "train":
            entry_dict['target'] = category + ": " + "Extract Triplets: "
            # entry_dict['target'] = "Extract Triplets: "
        else:
            entry_dict['target'] = ""

        txt = obj['txt']
        txt_lex = obj['txt_lex']

        # entry_dict['lex'] += txt
        # entry_dict['lex'] += " *# " + txt_lex
        entry_dict['target'] += txt

        # json_dict.append([entry_dict])
        # if entry_dict['source'].replace('Generate Textual Description:', '') != '':
        json_dict.append(entry_dict)

    if args.task == "train":
        with open(args.d2t_train_file, "w", encoding='utf-8') as f:
            for line in json_dict:
                f.write(str(line['source']) + '\n')
        print("Number of filtered entries:", num_filtered)

        with open(args.t2d_train_file, "w", encoding='utf-8') as f:
            for line in json_dict:
                f.write(str(line['target']) + '\n')
        print("Number of filtered entries:", num_filtered)

        with open(args.fine, "w", encoding='utf-8') as f:
            # for line in json_dict:
            # f.write(str(line['source']) + '\n')
            # f.write(str(line) + '\n')
            json.dump(json_dict, f, indent=4, ensure_ascii=False)
        print("Number of filtered entries:", num_filtered)

        print(len(json_dict))

        # with open(args.d2t_train_file, "w", encoding='utf-8') as f:
        #     # for line in json_dict:
        #     # f.write(str(line['source']) + '\n')
        #     # f.write(str(line) + '\n')
        #     json.dump(json_dict, f, indent=4, ensure_ascii=False)
        # print("Number of filtered entries:", num_filtered)

        # with open(args.t2d_train_file, "w", encoding='utf-8') as f:
        #     # for line in json_dict:
        #     # f.write(str(line['target']) + '\n')
        #     # f.write(str(line) + '\n')
        #     json.dump(json_dict, f, indent=4, ensure_ascii=False)
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
    parser.add_argument('--input_file', default='data/E2E/e2e_train.json')
    parser.add_argument('--d2t_train_file', default='data/E2E/d2t_train_src.txt')
    parser.add_argument('--t2d_train_file', default='data/E2E/t2d_train_src.txt')
    parser.add_argument('--d2t_dev_file', default='data/E2E/d2t_dev.csv')
    parser.add_argument('--t2d_dev_file', default='data/E2E/t2d_dev.csv')
    parser.add_argument('--d2t_test_file', default='data/E2E/d2t_test.csv')
    parser.add_argument('--t2d_test_file', default='data/E2E/t2d_test.csv')
    parser.add_argument('--fine', default='data/E2E/fine_train/train.json')
    parser.add_argument('--task', default='train')
    args = parser.parse_args()
    main(args)