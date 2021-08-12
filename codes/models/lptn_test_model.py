import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time
import numpy as np
import sys

from codes.models.archs import define_network
from codes.models.base_model import BaseModel
from codes.utils import get_root_logger, imwrite, tensor2img
import cv2
loss_module = importlib.import_module('codes.models.losses')
metric_module = importlib.import_module('codes.metrics')

class LPTNTestModel(BaseModel):

    def __init__(self, opt):
        super(LPTNTestModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'ref' in data:
            self.ref = data['ref'].to(self.device)#junl

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)

    def test_speed(self, times_per_img=50, size=None):
        if size is not None:
            lq_img = self.lq.resize_(1, 3, size[0], size[1])
        else:
            lq_img = self.lq
        self.net_g.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(times_per_img):
                _ = self.net_g(lq_img)
            torch.cuda.synchronize()
            self.duration = (time.time() - start) / times_per_img

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        
        for idx, val_data in enumerate(dataloader):
            img_name = val_data['lq_path'][0] #osp.splitext(osp.basename(val_data['lq_path'][0]))[0] #junl
            #print("^^^^^^^^lptn_test_model.py img_name,ref_name\n",val_data['lq_path'][0],val_data['ref_path'][0])

            ref_name = val_data['ref_path'][0]#junl
            #sys.exit()
            self.feed_data(val_data)
            self.test()

            gt_img =[]#junl
            ref_img =[]#junl
            visuals = self.get_current_visuals()
            input_img = tensor2img([visuals['lq']])
            result_img = tensor2img([visuals['result']])
            
            if 'gt' in visuals:
                print("!!!!!!!!!!!!!!'lptn_test_model.py gt' in visuals")
                gt_img = tensor2img([visuals['gt']])
                del self.gt
                #sys.exit()

            if 'ref' in visuals:#junl
                ref_img = tensor2img([visuals['ref']])
                del self.ref

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             str(current_iter),
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                        #print("333333333333333",self.opt['path']['visualization'],dataset_name,img_name,self.opt["name"])
                #print("###########3lptn_test_model.py",save_img_path)

                plot_img = np.hstack((input_img, result_img))
                #print("lptn_test_model.py visuals:",visuals)
                if 'gt' in visuals:
                    print("plot_img,input_img,gt_img",plot_img.shape,input_img.shape,gt_img.shape)
                    gt_img = cv2.resize(gt_img,(input_img.shape[1],input_img.shape[0]),interpolation=cv2.INTER_AREA)#junl
                    plot_img = np.hstack((plot_img, gt_img))
                
                if 'ref' in visuals:#junl add ref
                    #print("plot_img,input_img,ref_img",plot_img.shape,input_img.shape,ref_img.shape)
                    ref_img = cv2.resize(ref_img,(input_img.shape[1],input_img.shape[0]),interpolation=cv2.INTER_AREA)#junl
                    plot_img = np.hstack((plot_img, ref_img))

                imwrite(result_img, save_img_path)#plot_img

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    if gt_img ==[]:#junl
                       continue
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(result_img, gt_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def nondist_validation_speed(self, dataloader, times_per_img, num_imgs, size):

        avg_duration = 0
        for idx, val_data in enumerate(dataloader):
            if idx > num_imgs:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test_speed(times_per_img, size=size)
            avg_duration += self.duration / num_imgs
            print(f'{idx} Testing {img_name} (shape: {self.lq.shape[2]} * {self.lq.shape[3]}) duration: {self.duration}')

        print(f'average duration is {avg_duration} seconds')


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        
        if hasattr(self, 'ref'):#junl
           out_dict['ref'] = self.ref.detach().cpu()
        
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
