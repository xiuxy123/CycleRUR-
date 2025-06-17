import argparse
import csv
import glob
import random
import xml.etree.ElementTree as ET
import json

# CATEGORIES = frozenset([
#     "Airport",
#     "Artist",
#     "Astronaut",
#     "Athlete",
#     "Building",
#     "CelestialBody",
#     "City",
#     "ComicsCharacter",
#     "Company",
#     "Food",
#     "MeanOfTransportation",
#     "Monument",
#     "Politician",
#     "SportsTeam",
#     "University",
#     "WrittenWork",
# ])


CATEGORIES = frozenset([
    "Airport",
    "Artist",
    "Astronaut",
    "Athlete",
    "Building",
    "CelestialBody",
    "City",
    "ComicsCharacter",
    "Company",
    "Food",
    "MeanOfTransportation",
    "Monument",
    "Politician",
    "SportsTeam",
    "University",
    "WrittenWork",
])

# dart
# CATEGORIES = frozenset([
#     # "WikiTableQuestions_lily",
#     "WikiSQL_lily",
#     # "WikiTableQuestions_mturk",
#     "WikiSQL_decl_sents",
#     # "e2e",
#     # "webnlg"
# ])

# CATEGORIES = frozenset([
#     "MISC",
# ])

def main(args):
    num_filtered = 0
    json_dict = []
    # 获取匹配模式下的文件路径列表
    # file_paths = glob.glob('data/webnlg/train/*triples/*.xml')
    file_paths = glob.glob(args.xml_file)
    count = 0
    for xml_file in file_paths:
        print(xml_file)
        xml = ET.parse(xml_file)
        root = xml.getroot()

        for entry in root.iter('entry'):
            # if count % 5 == 0:
            #     if entry.find("lex").get('comment') not in CATEGORIES:
                if entry.get("category") not in CATEGORIES:
                    print(entry.get("category"))
                    num_filtered += 1
                    continue


                entry_dict = {}
                if args.task == "train":
                    entry_dict['source'] = entry.get("category") + ": " + "Generate Textual Description:"
                    # entry_dict['source'] = entry.find("lex").get('comment') + ": " + "Generate Textual Description:"  # dart
                    # entry_dict['source'] = "Generate Textual Description:"
                else:
                    entry_dict['source'] = ''

                tripleset = entry.find('modifiedtripleset')
                count = 0
                for triple in tripleset.findall('mtriple'):
                    sbj, prd, obj = triple.text.split(' | ')
                    entry_dict['source'] += " [S] " + str(sbj) + " [P] " + str(prd) + " [O] " + str(obj)
                    # if count > 0:
                    #     entry_dict['source'] += " | "
                    # entry_dict['source'] += "( " + str(sbj) + " " + str(prd) + " " + str(obj) + " )"
                    count += 1

                entry_dict['source'] = entry_dict['source'].strip()

                if 'TABLECONTEXT' in entry_dict['source']:
                    continue

                if args.task in "train":
                    entry_dict['target'] = entry.get("category") + ": " + "Extract Triplets: "
                    # entry_dict['target'] = entry.find("lex").get('comment') + ": " + "Extract Triplets: "  # dart
                    # entry_dict['target'] = "Extract Triplets: "
                    entry_dict['target'] += entry.findall('lex')[0].text
                    entry_list = []
                else:
                    entry_dict['target'] = ''
                    for index, lex in enumerate(entry.findall('lex')):
                        if lex.text is None:
                            continue
                        # if args.task in "train":
                        #     # entry_dict['target'] = entry.get("category") + ": " + "Extract Triplets: "
                        #     # entry_dict['target'] = entry.find("lex").get('comment') + ": " + "Extract Triplets: "  # dart
                        #     entry_dict['target'] = "Extract Triplets: "
                        # else:
                        #     entry_dict['target'] = ''
                        if index > 0:
                            entry_dict['target'] += " *# " + lex.text
                        else:
                            entry_dict['target'] += lex.text
                    #
                    # if entry_dict not in entry_list:
                    #     entry_list.append({'source': entry_dict['source'], 'target': entry_dict['target']})

                # randint = random.randint(0, len(entry.findall('lex')) - 1)
                # entry_dict['target'] += entry.findall('lex')[randint].text

                json_dict.append(entry_dict)

            # count = count + 1

    if args.task == "train":
        with open(args.d2t_train_file, "w", encoding='utf-8') as f:
            for line in json_dict:
                f.write(str(line['source']) + '\n')
                # f.write(str(line) + '\n')
            # json.dump(json_dict, f, indent=4, ensure_ascii=False)
        print("Number of filtered entries:", num_filtered)

        with open(args.t2d_train_file, "w", encoding='utf-8') as f:
            for line in json_dict:
                f.write(str(line['target']) + '\n')
                # f.write(str(line) + '\n')
            # json.dump(json_dict, f, indent=4, ensure_ascii=False)
        print("Number of entries:", len(json_dict))

    if args.task in ["dev", "test"]:

        if args.task == "dev":
            path1 = args.d2t_dev_file
            path2 = args.t2d_dev_file
            path3 = args.t2d_dev_file_txt
            path4 = args.d2t_dev_file_txt
        else:
            path1 = args.d2t_test_file
            path2 = args.t2d_test_file
            path3 = args.d2t_test_file_txt
            path4 = args.t2d_test_file_txt

        with open(path3, "w", encoding='utf-8') as f:
            for line in json_dict:
                f.write(str(line['source']) + '\n')
        print("Number of entries:", len(json_dict))

        with open(path4, "w", encoding='utf-8') as f:
            for line in json_dict:
                f.write(str(line['target']) + '\n')
        print("Number of entries:", len(json_dict))


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

        file.close()
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

    if args.task == "test":
        with open(args.d2t_ref_file, "w", encoding='utf-8') as f:
            for line in json_dict:
                f.write(str(line['target']) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--xml_file', default='data/webnlg2017/test/*triples/*.xml') # webnlg

    parser.add_argument('--xml_file', default='data/webnlg2017/test/*with_lex.xml')  # webnlg2017 test

    # parser.add_argument('--xml_file', default='data/dart/dev/*.xml')  #  dart
    parser.add_argument('--d2t_train_file', default='data/webnlg2017/d2t_train_src.txt')
    parser.add_argument('--t2d_train_file', default='data/webnlg2017/t2d_train_src.txt')
    parser.add_argument('--d2t_dev_file', default='data/webnlg2017/d2t_dev.csv')
    parser.add_argument('--t2d_dev_file', default='data/webnlg2017/t2d_dev.csv')
    parser.add_argument('--d2t_test_file', default='data/webnlg2017/d2t_test.csv')
    parser.add_argument('--t2d_test_file', default='data/webnlg2017/t2d_test.csv')

    parser.add_argument('--d2t_test_file_txt', default='data/webnlg2017/d2t_test.txt')
    parser.add_argument('--t2d_test_file_txt', default='data/webnlg2017/t2d_test.txt')

    parser.add_argument('--d2t_dev_file_txt', default='data/webnlg2017/d2t_dev.txt')
    parser.add_argument('--t2d_dev_file_txt', default='data/webnlg2017/t2d_dev.txt')

    parser.add_argument('--d2t_ref_file', default='data/webnlg2017/test_ref.txt')

    parser.add_argument('--task', default='test')

    parser.add_argument('--dataset', default='webnlg2017')
    args = parser.parse_args()
    main(args)
