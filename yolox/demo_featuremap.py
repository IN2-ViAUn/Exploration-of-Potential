import sys
sys.path.append('/home/xuxi/Exploration-of-Potential')

import argparse
import os
import time
from loguru import logger

import cv2
import json

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, vis

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from prettytable import PrettyTable

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

id_trans = {'0': 1,  '1': 2,   '2': 3,   '3': 4,   '4': 5, 
           '5': 6,   '6': 7,   '7': 8,   '8': 9,   '9': 10, 
           '10': 11, '11': 13, '12': 14, '13': 15, '14': 16, 
           '15': 17, '16': 18, '17': 19, '18': 20, '19': 21, 
           '20': 22, '21': 23, '22': 24, '23': 25, '24': 27, 
           '25': 28, '26': 31, '27': 32, '28': 33, '29': 34, 
           '30': 35, '31': 36, '32': 37, '33': 38, '34': 39, 
           '35': 40, '36': 41, '37': 42, '38': 43, '39': 44, 
           '40': 46, '41': 47, '42': 48, '43': 49, '44': 50, 
           '45': 51, '46': 52, '47': 53, '48': 54, '49': 55, 
           '50': 56, '51': 57, '52': 58, '53': 59, '54': 60, 
           '55': 61, '56': 62, '57': 63, '58': 64, '59': 65, 
           '60': 67, '61': 70, '62': 72, '63': 73, '64': 74, 
           '65': 75, '66': 76, '67': 77, '68': 78, '69': 79, 
           '70': 80, '71': 81, '72': 82, '73': 84, '74': 85, 
           '75': 86, '76': 87, '77': 88, '78': 89, '79': 90}

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", choices=['darknet', 'vgg', 'resnet', 'densenet'], help="type of backbone")
    parser.add_argument("-n", "--name", type=str, default='yolox-l', help="model name")
    parser.add_argument("--vis", action="store_true", help="Whether to visualize the featuremap")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="weights file")
    parser.add_argument("--device", default="cpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument( "-f", "--exp_file", default=None, type=str, help="pls input your experiment description file")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")

    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            t0 = time.time()
            outputs, output_fpn = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info, output_fpn

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, None, None, None
        output = output.cpu()

        bboxes = output[:, 0:4]

        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, bboxes, scores, cls

def image_demo(predictor, path, gt_boxes, dis_type):
    
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    
    names = []
    bboxes = []
    scores = []
    classes = []
    for image_name, gt_box in zip(files, gt_boxes):
        names.append(image_name)
        outputs, img_info, output_fpn = predictor.inference(image_name)
        neck = (output_fpn[0], output_fpn[1], output_fpn[2])
        
        # 显示特征图
        create_2D_feature_map(neck, outputs[0], gt_box, image_name)

        result_image, bbox, score, cls = predictor.visual(outputs[0], img_info, predictor.confthre)
        bboxes.append(bbox)
        scores.append(score)
        classes.append(cls)

        save_folder = os.path.join(vis_folder, dis_type)
        os.makedirs(save_folder, exist_ok=True)

        save_file_name = os.path.join(save_folder, os.path.basename(image_name))
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_image)

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    return names, bboxes, scores, classes

def get_img_info(json_file):

    coco =COCO(annotation_file=json_file)
    ids = list(sorted(coco.imgs.keys()))
    img_id = ids[0]  
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)
    image_name = coco.loadImgs(img_id)[0]['file_name']  
    image_path = os.path.join(json_file.split('/')[0], image_name)

    image = cv2.imread(image_path)
    img_h = image.shape[0]
    img_w = image.shape[1]
    
    return coco, targets, image, img_h, img_w

def get_img_mask(offset, ori_img, ori_img_h, ori_img_w, targets, coco):
    draw_temp = np.ones((ori_img_h, ori_img_w, 3), dtype=np.uint8) * 114
    gt_box_fm = np.zeros((len(targets), 4))
    gt_box = np.zeros((len(targets), 4))

    r = min(640 / ori_img_h, 640 / ori_img_w)
    new_w = int(ori_img_w * r)
    new_h = int(ori_img_h * r)

    for idx, target in enumerate(targets):
        x, y, w, h = target["bbox"]
        xmin = (x/ori_img_w) * new_w / 640 
        ymin = ((y+offset)/ori_img_h) * new_h / 640 
        xmax = ((x+w)/ori_img_w) * new_w / 640 
        ymax = (((y+offset)+h)/ori_img_h) * new_h / 640 
        gt_box_fm[idx,:] = xmin, ymin, xmax, ymax
        gt_box[idx,:] = x, y+offset, x+w, (y+offset)+h

        m = coco.annToMask(target)
        pad = np.zeros(shape=[abs(offset),ori_img_w], dtype=np.uint8)
        if offset < 0:  
            m_ = np.append(m,pad,axis=0)
            m_ = m_[abs(offset):]
            m_ = m_[..., None].repeat([3], axis = 2 )

        elif offset > 0:
            m_ = m[:(ori_img_h-offset)]
            m_ = np.append(pad,m_,axis=0)
            m_ = m_[..., None].repeat([3], axis = 2 )
        else:
            m_ = m[..., None].repeat([3], axis = 2 )
   
        obj = np.where(m != 0)
        for (obj_y,obj_x) in zip(obj[0], obj[1]):
            rgb = ori_img[obj_y,obj_x]
            draw_temp[obj_y+offset,obj_x] = rgb      

    return draw_temp, gt_box_fm, gt_box, m_
        
class Image_Distortion():
    def __init__(self):
        self.draw_temp_size = 1000
        self.sector_length = self.draw_temp_size - 100
        self.draw_resolution = 80

    def sector_distort(self, image, mask, Theta = 60, custom_rows = None):    
        draw_temp_h = self.draw_temp_size    
        draw_temp_w = int(draw_temp_h * np.sin(Theta/2*np.pi/180) * 2)    

        img_h, img_w, img_c = image.shape[0], image.shape[1], image.shape[2]
        scale_hw = img_h/img_w

        draw_temp_img = np.ones((draw_temp_h, draw_temp_w, img_c), dtype=np.uint8)*114
        draw_temp_mask = np.zeros((draw_temp_h, draw_temp_w, img_c), dtype=np.uint8)

        R_sector = self.draw_temp_size

        assert ((Theta >= 15)and(Theta <= 180)), "Theta is not in range 15°-180°!"
        theta_start = (180 - Theta)/2
        theta_sector = np.linspace(theta_start, theta_start + Theta, 165*self.draw_resolution, True)
        theta_sector_rad = theta_sector*np.pi/180
        
        M_rot = np.array([[np.cos(theta_sector_rad), -1*np.sin(theta_sector_rad)],
                          [np.sin(theta_sector_rad),    np.cos(theta_sector_rad)]])
        M_rot = M_rot.transpose(2, 0, 1)
        P_end = np.array([[self.draw_temp_size],
                          [0]])
        arc_end = np.matmul(M_rot,P_end).astype(np.int16).transpose(0,2,1)
        arc_end_uniq = arc_end[:,:,0] + arc_end[:,:,1]*1j 
        arc_end_length = np.unique(arc_end_uniq, return_index=True)[1].shape[0]
        
        if custom_rows is None:
            target_side = np.clip(int(arc_end_length*scale_hw), 0 , self.sector_length)

        else:
            assert (custom_rows <= self.sector_length), ("Custom row should be limited in 900!")
            target_side = custom_rows

        P_x = np.linspace(R_sector - target_side, R_sector, target_side)
        P_y = np.linspace(0,                      0,        target_side)
        P_xy= np.array([P_x,
                        P_y])

        new_p = np.matmul(M_rot,P_xy).astype(np.int16).transpose(0,2,1)

        target_w = 165*self.draw_resolution          
        img_resize = cv2.resize(image, (target_w, target_side))
        mask_resize = cv2.resize(mask, (target_w, target_side))

        idx_h = np.arange(0, target_side, 1)
        idx_w = np.arange(0, target_w, 1)
        ptx, pty = np.meshgrid(idx_h, idx_w)
        
        new_p[:,:,0] = np.clip(((new_p + draw_temp_w/2)[:,:,0] - 1), 0, draw_temp_w)
        
        new_p[:,:,1] = np.clip(((draw_temp_h - new_p)[:,:,1] - 1), 0, draw_temp_h)
        ptx = ptx[:,::-1] 
        pty = pty[::-1,:]
        
        draw_temp_img[new_p[:,:,1], new_p[:,:,0]] = img_resize[ptx, pty]
        draw_temp_mask[new_p[:,:,1], new_p[:,:,0]] = mask_resize[ptx, pty]

        l_bound = np.min(new_p[:,:,1])
        r_bound = np.max(new_p[:,:,1])
        t_bound = np.min(new_p[:,:,0])
        b_bound = np.max(new_p[:,:,0])
       
        draw_temp_img = draw_temp_img[l_bound:r_bound, t_bound:b_bound]
        new_image = draw_temp_img.copy()

        draw_temp_mask = draw_temp_mask[l_bound:r_bound, t_bound:b_bound]
        single_mask = draw_temp_mask[:,:,0].copy().astype(np.bool8)
        new_mask = np.where(single_mask == True)
        new_mask = (new_mask[1], new_mask[0])

        new_mask = [mask_temp.tolist() for mask_temp in new_mask] 

        cur_mask_x = new_mask[0]
        cur_mask_y = new_mask[1]
        if (len(cur_mask_x) != 0) and (len(cur_mask_y) != 0):
            box_x_l = np.min(cur_mask_x)
            box_x_r = np.max(cur_mask_x)
            box_y_t = np.min(cur_mask_y)
            box_y_b = np.max(cur_mask_y)
            new_bbox = [box_x_l, box_y_t, box_x_r - box_x_l, box_y_b - box_y_t]

        else:
            new_bbox = []

        return new_image, new_bbox

def create_2D_feature_map(output_fpn, outputs_final, gt_box, image_name, label_info = True, gt_info = True, gt_fm_av=True):

    fig_2d, ax_2d = plt.subplots(2,3)

    if outputs_final is None:
        pred_res = np.zeros((1, 7))
    else:
        pred_res = outputs_final.cpu().numpy()
    pred_box = pred_res[:,:4]/640
    
    per_img_results = []
    for idx, cur_fpn in enumerate(output_fpn):
        fpn_np = cur_fpn.cpu().numpy().squeeze(0)
        fpn_channel = fpn_np.shape[0]
        
        fpn_np_sum = fpn_np.sum(axis=0)/fpn_channel
        sns.heatmap(fpn_np_sum, ax=ax_2d[0,idx], square=True)
        sns.heatmap(fpn_np_sum, ax=ax_2d[1,idx], square=True)

        if(label_info):
            x_min = np.expand_dims(fpn_np.shape[1]*pred_box[:,0],axis=0)
            x_max = np.expand_dims(fpn_np.shape[1]*pred_box[:,2],axis=0)
            y_min = np.expand_dims(fpn_np.shape[1]*pred_box[:,1],axis=0)
            y_max = np.expand_dims(fpn_np.shape[1]*pred_box[:,3],axis=0)
               
            x_ = np.concatenate((x_min, x_min, x_max, x_max, x_min), axis=0)
            y_ = np.concatenate((y_min, y_max, y_max, y_min, y_min), axis=0)

            for cur_box in range(pred_box.shape[0]):
                cur_box_x = x_[:, cur_box]
                cur_box_y = y_[:, cur_box]
                ax_2d[1,idx].plot(cur_box_x, cur_box_y, color='blue')  
        
        if(gt_info):
            x_min = np.expand_dims(fpn_np.shape[1]*gt_box[:,0],axis=0)
            x_max = np.expand_dims(fpn_np.shape[1]*gt_box[:,2],axis=0)
            y_min = np.expand_dims(fpn_np.shape[1]*gt_box[:,1],axis=0)
            y_max = np.expand_dims(fpn_np.shape[1]*gt_box[:,3],axis=0)
           
            x_ = np.concatenate((x_min, x_min, x_max, x_max, x_min), axis=0)
            y_ = np.concatenate((y_min, y_max, y_max, y_min, y_min), axis=0)

            for cur_box in range(gt_box.shape[0]):
                cur_box_x = x_[:, cur_box]
                cur_box_y = y_[:, cur_box]
                ax_2d[1,idx].plot(cur_box_x, cur_box_y, color='green') 

        if(gt_fm_av):
            for gt in gt_box:
                xmin = gt[0]*fpn_np.shape[1]
                ymin = gt[1]*fpn_np.shape[1]
                xmax = gt[2]*fpn_np.shape[1]
                ymax = gt[3]*fpn_np.shape[1]
            
                gt_pixel = fpn_np_sum[int(ymin):int(ymax), int(xmin):int(xmax)]
                per_img_results.append(gt_pixel.sum()/(gt_pixel.shape[0]*gt_pixel.shape[1]))
    
    table_dic[image_name.split('/')[-1].split('.')[0]] = per_img_results
    if args.vis:
        plt.show()
        plt.close()
    else:
        plt.close()

def test(exp, args, gt_boxes, test_img_path, dis_type):

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    
    model = exp.get_model(args.backbone)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    ckpt_file = args.ckpt

    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")


    predictor = Predictor(
        model, exp, COCO_CLASSES, 
        args.device, args.fp16, args.legacy,
    )
 
    names, bboxes, scores, classes = image_demo(predictor, test_img_path, gt_boxes, dis_type)

    return names, bboxes, scores, classes

def preparation(new_data_folder, json_path):
    with open(json_path, 'r') as anno:
        data = json.load(anno)
    after = {}
    after["images"] = []
    after["annotations"] = []
    after["categories"] = data["categories"]

    os.makedirs(new_data_folder, exist_ok=True)
        
    return after, data

def Undistorted(coco, targets, ori_img, ori_img_h, ori_img_w):

    dis_type = 'none'
    data_path = os.path.join(new_data_path, dis_type)
    after, data = preparation(data_path, anno_path)
    
    gt_boxes_fm = []
    i = 1
    for offset in range(-100, 150, 50):
        test_img, gt_box_fm, gt_box, _ = get_img_mask(offset, ori_img, ori_img_h, ori_img_w, targets, coco)
        gt_boxes_fm.append(gt_box_fm)

        no_dis_img = test_img

        cv2.imwrite(os.path.join(data_path, 'offset_{}_{}.png'.format(str(offset).zfill(3), dis_type)), no_dis_img)

        image = {}
        image["height"] = int(no_dis_img.shape[0])
        image["width"] = int(no_dis_img.shape[1])
        image["id"] = int(offset)
        annotations = {}
        annotations["area"] = float((gt_box[0,2]-gt_box[0,0]) * (gt_box[0,3]-gt_box[0,1]))  # 因为只有一个真值所以是0
        annotations["iscrowd"] = data['annotations'][0]['iscrowd']
        annotations["image_id"] = int(offset)
        annotations["bbox"] = [float(gt_box[0,0]),float(gt_box[0,1]),float(gt_box[0,2]-gt_box[0,0]),float(gt_box[0,3]-gt_box[0,1])]
        annotations["category_id"] = data['annotations'][0]['category_id']  # 因为train的1-90序号是7，这不知道从哪里获得，目前只能硬给【考虑从label_ori中获得】
        annotations["id"] = int(i)

        after["images"].append(image)
        after["annotations"].append(annotations)

        i += 1
    
    with open(os.path.join(data_path, 'gt.json'), 'w', newline='\n') as gt:
        gt.write(json.dumps(after, indent=1))

    names, bboxes, scores, classes = test(exp, args, gt_boxes_fm, data_path, dis_type)
    dt_path = os.path.join(dt_folder, dis_type)
    os.makedirs(dt_path, exist_ok=True)
    dt = dt_json_create(names, bboxes, scores, classes, dt_path)
    
    print('************************{}************************'.format(dis_type))
    coco_ap(gt, dt)
    
def Distorted(coco, targets, ori_img, ori_img_h, ori_img_w):
    for theta in range(30,95,5):
        dis_type = 'theta_{}'.format(theta)
        data_path = os.path.join(new_data_path, dis_type)
        after, data = preparation(data_path, anno_path)
        gt_boxes_fm = []
        i = 1
        for offset in range(-100, 150, 50):
            test_img, _, _, mask = get_img_mask(offset, ori_img, ori_img_h, ori_img_w, targets, coco)
        
            dis_img, dis_label = Image_Distortion().sector_distort(test_img, mask, Theta = theta)

            cv2.imwrite(os.path.join(data_path, 'offset_{}_theta_{}.png'.format(str(offset).zfill(3), theta)), dis_img)

            r = min(640 / dis_img.shape[0], 640 / dis_img.shape[1])
            new_w = int(dis_img.shape[1] * r)
            new_h = int(dis_img.shape[0] * r)

            xmin = dis_label[0] / dis_img.shape[1] * new_w / 640 
            ymin = dis_label[1] / dis_img.shape[0] * new_h / 640 
            xmax = (dis_label[0] + dis_label[2]) / dis_img.shape[1] * new_w / 640 
            ymax = (dis_label[1] + dis_label[3]) / dis_img.shape[0] * new_h / 640 

            
            gt_box_fm = np.zeros((1, 4))
            gt_box_fm[0,:] = xmin, ymin, xmax, ymax
            gt_boxes_fm.append(gt_box_fm)

            image = {}
            image["height"] = int(dis_img.shape[0])
            image["width"] = int(dis_img.shape[1])
            image["id"] = int(offset)

            annotations = {}
            annotations["area"] = float(dis_label[2] * dis_label[3])
            annotations["iscrowd"] = data['annotations'][0]['iscrowd']
            annotations["image_id"] = int(offset)
            annotations["bbox"] = [float(dis_label[0]),float(dis_label[1]),float(dis_label[2]),float(dis_label[3])]
            annotations["category_id"] = data['annotations'][0]['category_id'] 
            annotations["id"] = int(i)

            after["images"].append(image)
            after["annotations"].append(annotations)

            i += 1

        with open(os.path.join(data_path, 'gt.json'), 'w', newline='\n') as gt:
            gt.write(json.dumps(after, indent=1))

        names, bboxes, scores, classes = test(exp, args, gt_boxes_fm, data_path, dis_type)
        dt_path = os.path.join(dt_folder, dis_type)
        os.makedirs(dt_path, exist_ok=True)
        dt = dt_json_create(names, bboxes, scores, classes, dt_path)
        
        print('******************************{}******************************'.format(dis_type))
        coco_ap(gt, dt)

def dt_json_create(names, bboxes, scores, classes, path):
    json_results = []
    for name, bbox, score, cls in zip(names, bboxes, scores, classes):
        if (bbox == None) or (score == None) or (cls == None):
            pass
        else:
            bbox = bbox.cpu().numpy()
            score = score.cpu().numpy()
            cls = cls.cpu().numpy()
            for j in range(len(bbox)):
                xmin = np.float64(bbox[j][0])
                ymin = np.float64(bbox[j][1])
                xmax = np.float64(bbox[j][2])
                ymax = np.float64(bbox[j][3])
                w = xmax - xmin
                h = ymax - ymin

                score =  np.float64(score[j])
                class_id = int(cls[j])
                category_id = id_trans[str(class_id)]
                image_id = int(name.split('/')[-1].split('_')[1])
                dt_data = {
                    'image_id' : image_id,
                    'category_id' : category_id,
                    'bbox' : [xmin, ymin, w, h],
                    'score' :  score
                }
                json_results.append(dt_data)

    with open(os.path.join(path, 'dt.json'), 'w', newline='\n') as dt:
        dt.write(json.dumps(json_results, indent=1))
    
    return dt

def coco_ap(gt, dt):  
    cocoGt=COCO(gt.name)
    cocoDt=cocoGt.loadRes(dt.name)

    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    table_dic = {}

    new_data_path = os.path.join(exp.output_dir, "new_data")
    os.makedirs(new_data_path, exist_ok=True)

    vis_folder = os.path.join(exp.output_dir, exp.exp_name + '_' + args.backbone, 'vis_res')
    os.makedirs(vis_folder, exist_ok=True)

    dt_folder = os.path.join(exp.output_dir, exp.exp_name + '_' + args.backbone, 'dt_json')
    os.makedirs(dt_folder, exist_ok=True)

    anno_path = 'test_data/000000130566.json'

    coco, targets, ori_img, ori_img_h, ori_img_w = get_img_info(anno_path)

    Undistorted(coco, targets, ori_img, ori_img_h, ori_img_w)
    Distorted(coco, targets, ori_img, ori_img_h, ori_img_w)


    fm_size = [80,40,20]
    for id in range(len(fm_size)):
        print('************************************************Feature Map Size:{}×{}************************************************'.format(fm_size[id], fm_size[id]))
        table = PrettyTable(['','-100','-50','0','50','100'])
        table.add_row(['None',table_dic['offset_-100_none'][id],table_dic['offset_-50_none'][id],table_dic['offset_000_none'][id],table_dic['offset_050_none'][id],table_dic['offset_100_none'][id]])
        for theta in range(30,95,5):
            table.add_row(['theta_{}'.format(theta),table_dic['offset_-100_theta_{}'.format(theta)][id],table_dic['offset_-50_theta_{}'.format(theta)][id],table_dic['offset_000_theta_{}'.format(theta)][id],table_dic['offset_050_theta_{}'.format(theta)][id],table_dic['offset_100_theta_{}'.format(theta)][id]])
        print(table)
