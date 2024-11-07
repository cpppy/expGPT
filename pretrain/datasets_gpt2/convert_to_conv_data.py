import argparse
import json
import os
from tqdm import tqdm
import glob
import shutil


'''
orig_data: {"instruction": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？", "input": "", "output": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"}
convert: {
    "id": xxx,
    "conversations":[
        {"from": "human", "value": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？"},
        {"from": "assistant", "value": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"},
    ]
}
'''

def main():

    orig_dir = '/data/data/belle_data_10M'
    orig_datas = glob.glob(f'{orig_dir}/*.json')

    save_dir = '/data/data/belle_data_10M_conv'
    os.makedirs(save_dir, exist_ok=True)

    for orig_data in orig_datas:
        print(f'proc: {orig_data}')
        # orig_data = '/school_math_0.25M.json'
        if 'train_3.5M' in orig_data:
            shutil.copy(src=orig_data, dst=write_data)
            continue
        write_data = os.path.join(save_dir, os.path.basename(orig_data))

        dataset_name = os.path.basename(orig_data).replace('.json', '')

        f_write = open(write_data,"w")
        with open(orig_data) as f:
            lines = f.readlines()
            num_id = 1
            for line in tqdm(lines, desc=dataset_name):
                data = json.loads(line)
                # print(data)
                # exit(0)

                conversations = [{"from": "human", "value": data['instruction']+data['input']},{"from": "assistant", "value": data['output']}]
                # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
                # print(conversations)
                # exit(0)

                uniq_id = data['id'] if "id" in data else dataset_name+"-"+str(num_id)
                item = {"id":uniq_id, "conversations": conversations}
                f_write.write(json.dumps(item, ensure_ascii=False)+"\n")
                num_id += 1
        f_write.close()


if __name__ == "__main__":
    main()



