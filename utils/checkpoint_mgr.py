import torch
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import glob


class CheckpointMgr(object):
    
    def __init__(self, ckpt_dir, max_remain=3):
        super(CheckpointMgr, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.max_remain = max_remain
        self.ckpt_save_fn_list = []
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def get_latest_checkpoint(self):
        ckpt_fpath_list = glob.glob(f'{self.ckpt_dir}/model_*.pth')
        if len(ckpt_fpath_list) == 0:
            return None
        else:
            modify_time_list = [os.path.getmtime(fpath) for fpath in ckpt_fpath_list]
            latest_fpath_idx = np.argmax(modify_time_list)
            latest_ckpt_fpath = ckpt_fpath_list[int(latest_fpath_idx)]
            print('latest_ckpt_fpath: ', latest_ckpt_fpath)
            return latest_ckpt_fpath
    
    def load_checkpoint(self, model, ckpt_fpath=None, warm_load=False, map_location='cpu'):
        ckpt_fpath = ckpt_fpath if ckpt_fpath is not None else self.get_latest_checkpoint()
        if ckpt_fpath is None:
            print('None ckpt file can be used, load fail.')
            return False
        logging.info(f'[CHECKPOINT] loading_ckpt: {ckpt_fpath}')
        if ckpt_fpath.startswith('model_'):
            ckpt_fpath = os.path.join(self.ckpt_dir, ckpt_fpath)
        if not warm_load:
            save_dict = torch.load(ckpt_fpath, map_location=map_location)
            # save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
            # save_dict = {key.replace('module.', ''): val
            #              for key, val in save_dict.items()}
            model.load_state_dict(save_dict)
            return True
        else:
            warm_load_params = self._warm_load_weights(model, ckpt_fpath, map_location)
            return True

    def load_checkpoint_sl(self, model, optimizer, ckpt_fpath=None, warm_load=False, map_location='cpu'):
        ckpt_fpath = ckpt_fpath if ckpt_fpath is not None else self.get_latest_checkpoint()
        if ckpt_fpath is None:
            print('None ckpt file can be used, load fail.')
            return {}
        save_dict = torch.load(ckpt_fpath, map_location=map_location)
        model_dict = save_dict['state_dict']
        optim_dict = save_dict['optimizer']
        model_dict = {key.replace('module.', ''): val
                     for key, val in model_dict.items()}
        model.load_state_dict(model_dict)
        return optim_dict

    def load_checkpoint_partialFC(self, model, class_start, num_local, ckpt_fpath=None, map_location='cpu'):
        ckpt_fpath = ckpt_fpath if ckpt_fpath is not None else self.get_latest_checkpoint()
        if ckpt_fpath is None:
            print('None ckpt file can be used, load fail.')
            return False
        else:
            save_dict = torch.load(ckpt_fpath, map_location=map_location)
            save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
            save_dict = {key.replace('module.', ''): val
                         for key, val in save_dict.items()}
            if 'weight' in save_dict.keys():
                entire_weight = save_dict['weight']
                seg_index = [class_start, class_start + num_local]
                print('load_entire_weight: {}, seg_index:{}'.format(entire_weight.size(), seg_index))
                save_dict = {"sub_weight": entire_weight[class_start: class_start+num_local, :]}
            model.load_state_dict(save_dict)
            return True

    def _warm_load_weights(self, model, ckpt_path, map_location='cpu'):
        model_dict = model.state_dict()
        save_dict = torch.load(ckpt_path, map_location=map_location)
        save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
        # save_dict = {key.replace('module.', ''): val
        #               for key, val in save_dict.items()}
        # print(save_dict['model'].keys())
        # new_save_dict = {}
        # for k, v in save_dict['model'].items():
        #     if k.startswith('patch') or k.startswith('layer'):
        #         k = f'backbone.backbone.{k}'.replace('layers.0', 'layer1').replace('layers.1', 'layer2').replace('layers.2', 'layer3').replace('layers.3', 'layer4')
        #         new_save_dict[k] = v
        # save_dict = new_save_dict
        # print(save_dict.keys())
        # for k, v in save_dict.items():
        #     if 'relative_position_index' in k:
        #         print(f'-------------- {k} -------------')
        #         print(v.shape)
        #         print(v)
        #
        # exit(0)
        load_dict = {}
        new_params = {}
        for k, v in model_dict.items():
            if k not in save_dict.keys():
                print('warm_load_weights/new_val: ', k, v.size())
                new_params[k] = v
            elif save_dict[k].size() != v.size():
                print('warm_load_weights/change_val: {}, {}-->{}'.format(k, save_dict[k].size(), v.size()))
                new_params[k] = v
            else:
                load_dict.setdefault(k, save_dict.get(k))
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)
        return new_params
    
    def save_checkpoint(self, model, ckpt_fpath=None, verbose=False):
        if ckpt_fpath is not None:
            if not os.path.isabs(ckpt_fpath):
                ckpt_fpath = os.path.join(self.ckpt_dir, ckpt_fpath)
        else:
            # ckpt_fpath = None
            if len(self.ckpt_save_fn_list) > 0:
                prev_fn = self.ckpt_save_fn_list[-1]
            else:
                prev_fn = self.get_latest_checkpoint()
                prev_fn = prev_fn if prev_fn is None else prev_fn.split('/')[-1]
            if prev_fn is None or not prev_fn.startswith('model_'):
                save_fn = 'model_0.pth'
            else:
                save_fn = 'model_{}.pth'.format(int(prev_fn.replace('.pth', '').split('_')[-1]) + 1)
            ckpt_fpath = os.path.join(self.ckpt_dir, save_fn)
            self.ckpt_save_fn_list.append(save_fn)
            if len(self.ckpt_save_fn_list) > self.max_remain:
                delete_fn = self.ckpt_save_fn_list[0]
                self.ckpt_save_fn_list.remove(delete_fn)
                os.remove(os.path.join(self.ckpt_dir, delete_fn))
        if verbose:
            print('save_ckpt_fpath: ', ckpt_fpath)
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), ckpt_fpath)
        elif isinstance(model, dict):
            torch.save(model, ckpt_fpath)
        else:
            print('ERROR, invalid_model_ckpt_data!')
        return ckpt_fpath


