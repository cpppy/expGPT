import os
import json



def main():

    json_path = '/data/data/ultrachat_data/ultrachat_material_release_230412.json'
    samples = []
    with open(json_path, 'r') as f:
        for line in f.readlines():
            s = json.loads(line)
            samples.append(s)
    print(f'n_sample: {len(samples)}')

    print(samples[1])



if __name__=='__main__':

    main()


