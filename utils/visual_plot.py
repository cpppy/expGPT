import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


def main():
    df = pd.read_csv('/data/MedicalSeg/run_mbv2_3dunet/old_record/prev_mbv2_3dunet_id1.csv')
    print(df.head())

    epochs = df['epoch'].to_numpy() + 1
    print(epochs)

    losses = df['loss'].to_numpy()
    print(losses)

    ious = df['iou'].to_numpy()

    df2 = pd.read_csv('/data/MedicalSeg/run_mbv2_3dunet/old_record/prev_mbv2_3dunet_id4.csv')
    print(df2.head())

    epochs2 = df2['epoch'].to_numpy() + 1
    print(epochs2)

    losses2 = df2['loss'].to_numpy()
    print(losses2)

    ious2 = df2['iou'].to_numpy()

    fig = plt.figure(figsize=(6, 4))
    fig.add_subplot(111)
    # plt.plot(epochs, losses, marker='8', color='orange', linewidth=1, label='loss')
    # plt.plot(epochs, losses, color='red', linewidth=1, label='3DUnet-loss')
    # # plt.plot(range(n_test), gt_labels, color='b', linewidth=1, label='gt_label')
    # # plt.plot(range(n_test), load_gt(), color='r', linewidth=1, label='std_gt_label')
    # plt.plot(epochs2, losses2, color='blue', linewidth=1, label='OURS-loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')

    plt.plot(epochs, ious, color='red', linewidth=1, label='3DUnet-iou')
    # plt.plot(range(n_test), gt_labels, color='b', linewidth=1, label='gt_label')
    # plt.plot(range(n_test), load_gt(), color='r', linewidth=1, label='std_gt_label')
    plt.plot(epochs2, ious2, color='blue', linewidth=1, label='OURS-iou')
    plt.xlabel('epoch')
    plt.ylabel('iou')

    # title = '{} \n seq_len={} \n {} -> {} \n MAE={:.2f} | RMSE={:.2f} \n [STD] MAE={:.2f} | RMSE={:.2f} '.format(
    # title = 'seq_len={} \n {} -> {} \n MAE={:.2f} | RMSE={:.2f} \n '.format(
    #     # ckpt_name,
    #     dataset.inp_len,
    #     start_date,
    #     end_date,
    #     mae,
    #     rmse,
    #     # std_mae,
    #     # std_rmse
    # )
    # plt.title(title, loc='right')
    plt.legend()
    plt.show()


def visual_flops():
    pos_frame_nums = [115.1, 42.4]
    print(pos_frame_nums)

    plt.bar(x=['1', '2'],
            height=pos_frame_nums,
            width=0.4,
            align='center',
            color=['red', 'blue'],
            label=['3DUnet', 'OURS'],
            edgecolor='black'
            )
    # plt.xlabel('pos_slice')
    plt.ylabel('FLOPs(G)')

    # plt.grid()

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
    # visual_flops()
