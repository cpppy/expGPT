import os


def check_copy():

    file_size = -1
    real_path = 'xxx'
    while True:
        if os.path.exists(real_path):
            if file_size < 0:
                file_size = sizeof(real_path)
            time.sleep(5)
            cur_file_size = sizeof(real_path)
            if cur_file_size == file_size:
                print(f'already finished')
                break
            else:
                file_size = cur_file_size
                print(f'still waiting')
        else:
            time.sleep(15)











