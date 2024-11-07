import torch


def main():

    is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
    is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
    is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
    is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

    print(is_sm75, is_sm8x, is_sm80, is_sm90)


if __name__=='__main__':

    main()