import cv2
import os
import time
import numpy as np
import tqdm
import torch
from ultralytics import YOLO

import contextlib
import io
import random
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================================================
# ⚙️ 하이퍼파라미터 (Hyperparameters)
# =========================================================
MODEL_MAIN_PATH = os.environ.get('HAHI_MODEL', 'model/best_small.pt')

CONF_GLOBAL = 0.3
CONF_FILTER = 0.1     
CONF_DENSE = 0.3
CONF_TETRIS = 0.3
CONF_UC = 0.3

DENSE_RATIO_THRESH = 0.30  
NMS_CONF_THRESH = 0.3
NMS_IOU_THRESH = 0.4    

DENSE_WINDOW_SIZE = 512
DENSE_STEP = 320      

MERGE_PAD = 16
CROP_PAD_LARGE = 80     
CROP_PAD_SMALL = 16
CROP_PAD_THRESH = 200

CANVAS_SIZE = 960
CANVAS_MARGIN = 2
CANVAS_BG_COLOR = 114

UPSCALE_RATIO = 1.5        
UPSCALE_MAX_THRESH = 200    

NUM_TEST_IMAGES = 5000
HR_THRESHOLD = 1920 * 1080 

# =========================================================
dataset_root = os.environ.get('HAHI_DATASET', 'data/valid')
img_dir, lbl_dir = os.path.join(dataset_root, 'images'), os.path.join(dataset_root, 'labels')
img_list = sorted(os.listdir(img_dir))[:NUM_TEST_IMAGES]

print(f"🚀 [Ablation Study] 6단계 종합 성능 검증 파이프라인 시작! (Total {len(img_list)} images)")

m2 = YOLO(MODEL_MAIN_PATH)
sce_vis_samples = []

def get_size_category(w, h):
    area = w * h
    if area < 32 ** 2: return 'small'
    elif area < 96 ** 2: return 'medium'
    else: return 'large'

def merge_clusters_dynamic(boxes, img_w, img_h, merge_pad=MERGE_PAD):
    if not len(boxes): return []
    def get_padded(b, pad): return [max(0, b[0]-pad), max(0, b[1]-pad), min(img_w, b[2]+pad), min(img_h, b[3]+pad)]
    def is_overlap(b1, b2):
        p1, p2 = get_padded(b1, merge_pad), get_padded(b2, merge_pad)
        return (min(p1[2], p2[2]) > max(p1[0], p2[0])) and (min(p1[3], p2[3]) > max(p1[1], p2[1]))
    curr = boxes.copy()
    while True:
        merged, flags = [], [False]*len(curr)
        for i in range(len(curr)):
            if flags[i]: continue
            b = curr[i]
            for j in range(i+1, len(curr)):
                if not flags[j] and is_overlap(b, curr[j]):
                    b = [min(b[0], curr[j][0]), min(b[1], curr[j][1]), max(b[2], curr[j][2]), max(b[3], curr[j][3])]
                    flags[j] = True
            merged.append(b)
        if len(merged) == len(curr): break
        curr = merged
    final_boxes = []
    for b in curr:
        bw, bh = b[2] - b[0], b[3] - b[1]
        crop_pad = CROP_PAD_LARGE if max(bw, bh) < CROP_PAD_THRESH else CROP_PAD_SMALL 
        final_boxes.append(get_padded(b, crop_pad))
    return final_boxes

def run_official_ablation_benchmark(method_name):
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        
    all_gts = {}; all_preds = []
    img_infos = {} 
    
    stats = {
        'ALL': {'count': 0, 'inf_time': 0, 'total_time': 0, 'inf_cnt': 0, 'indices': set()},
        'HR':  {'count': 0, 'inf_time': 0, 'total_time': 0, 'inf_cnt': 0, 'indices': set()},
        'LR':  {'count': 0, 'inf_time': 0, 'total_time': 0, 'inf_cnt': 0, 'indices': set()}
    }

    pbar = tqdm.tqdm(img_list, desc=f"⏳ {method_name}", bar_format='{l_bar}{bar:30}{r_bar}')
    for img_idx, img_name in enumerate(pbar):
        img_path, lbl_path = os.path.join(img_dir, img_name), os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))
        img = cv2.imread(img_path); h, w, _ = img.shape
        
        img_infos[img_idx] = {'w': w, 'h': h, 'name': img_name}
        
        gts = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    c, xc, yc, bw, bh = map(float, line.split())
                    gts.append([int(c), (xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h]) 
        all_gts[img_idx] = gts

        t_pipe_start = time.time()
        img_inf_time, img_inf_cnt = 0, 0
        
        # =====================================================================
        # 1. Baseline: UC (2x2 Uniform Crop)
        # =====================================================================
        if method_name == "UC (2x2 Uniform Crop)":
            ch, cw = h // 2, w // 2
            crops, offsets = [img], [(0, 0)]
            for y in [0, ch]:
                for x in [0, cw]:
                    crops.append(img[y:y+ch, x:x+cw])
                    offsets.append((x, y))
            
            t_inf_start = time.time()
            results2 = m2.predict(crops, conf=CONF_UC, verbose=False, batch=5)
            img_inf_time += (time.time() - t_inf_start); img_inf_cnt += 5 
            
            temp_boxes, temp_scores, temp_classes = [], [], []
            for i, res in enumerate(results2):
                ox, oy = offsets[i]
                for b in res.boxes:
                    bx1 = b.xyxy[0][0].item() + ox
                    by1 = b.xyxy[0][1].item() + oy
                    bx2 = b.xyxy[0][2].item() + ox
                    by2 = b.xyxy[0][3].item() + oy
                    temp_boxes.append([bx1, by1, bx2, by2])
                    temp_scores.append(float(b.conf[0]))
                    temp_classes.append(int(b.cls[0]))
                    
            for c in set(temp_classes):
                c_boxes = [b for j, b in enumerate(temp_boxes) if temp_classes[j] == c]
                c_scores = [s for j, s in enumerate(temp_scores) if temp_classes[j] == c]
                cv_boxes = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in c_boxes]
                indices = cv2.dnn.NMSBoxes(cv_boxes, c_scores, NMS_CONF_THRESH, NMS_IOU_THRESH)
                if len(indices) > 0:
                    for idx in indices.flatten(): all_preds.append([img_idx, c, c_scores[idx]] + c_boxes[idx])

        # =====================================================================
        # 2 ~ 6. Ours 제안 기법 통합 분기
        # =====================================================================
        else:
            global_final_boxes, global_final_scores, global_final_classes = [], [], []
            local_boxes, local_scores, local_classes = [], [], []
            roi_boxes = []
            
            t_inf_start = time.time()
            res_global_all = m2.predict(img, conf=CONF_FILTER, verbose=False)
            img_inf_time += (time.time() - t_inf_start); img_inf_cnt += 1
            
            for b in res_global_all[0].boxes:
                bx1, by1, bx2, by2 = map(float, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                if conf >= CONF_GLOBAL:
                    global_final_boxes.append([bx1, by1, bx2, by2])
                    global_final_scores.append(min(1.0, conf * 1.10))
                    global_final_classes.append(int(b.cls[0]))
                roi_boxes.append([bx1, by1, bx2, by2, conf])
            
            remaining_boxes = roi_boxes.copy()
            dense_regions = []
            
            # --- [라우팅(Routing) 모듈] ---
            if method_name == "Packing Only":
                pass # 라우팅 생략
                
            elif method_name in ["DAHI Only", "DAHI + Packing"]:
                while len(remaining_boxes) > 0:
                    best_count, best_region = -1, None
                    for y in range(0, h - DENSE_WINDOW_SIZE + 1, DENSE_STEP):
                        for x in range(0, w - DENSE_WINDOW_SIZE + 1, DENSE_STEP):
                            count = sum(1 for rb in remaining_boxes if rb[0] >= x and rb[1] >= y and rb[2] <= x + DENSE_WINDOW_SIZE and rb[3] <= y + DENSE_WINDOW_SIZE)
                            if count > best_count: 
                                best_count, best_region = count, (x, y, x + DENSE_WINDOW_SIZE, y + DENSE_WINDOW_SIZE)
                    if best_region and best_count >= 1:
                        dense_regions.append(best_region)
                        dx1, dy1, dx2, dy2 = best_region
                        remaining_boxes = [rb for rb in remaining_boxes if not (rb[0] >= dx1 and rb[1] >= dy1 and rb[2] <= dx2 and rb[3] <= dy2)]
                    else: break
                    
                    # 💡 [DAHI 딱 1회 스캔 조건] DAHI + Packing 조합일 경우 1개만 찾고 루프 종료
                    if method_name == "DAHI + Packing":
                        break
                        
            elif method_name in ["SBSI + Packing", "SBSI + DCRP"]:
                small_info = []
                for b in remaining_boxes:
                    bx1, by1, bx2, by2, conf = b 
                    w_box, h_box = bx2 - bx1, by2 - by1
                    if get_size_category(w_box, h_box) == 'small':
                        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                        weight = (1.0 - conf) * np.sqrt(w_box * h_box)
                        small_info.append([cx, cy, weight])
                
                total_small_objs = len(small_info)
                route_threshold = max(1, int(total_small_objs * DENSE_RATIO_THRESH)) 
                
                if total_small_objs > 0:
                    pts = np.array([[info[0], info[1]] for info in small_info]) 
                    weights = np.array([info[2] for info in small_info])
                    norm_weights = weights / (np.mean(weights) + 1e-6)
                    sigma = DENSE_WINDOW_SIZE / 4.0 
                    
                    while len(pts) > 0:
                        dist_sq = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1) 
                        kernel = np.exp(-dist_sq / (2 * sigma ** 2)) 
                        densities = kernel @ norm_weights 
                        
                        best_idx = np.argmax(densities)
                        influence = norm_weights * kernel[best_idx]
                        sum_influence = np.sum(influence)
                        
                        if sum_influence > 0:
                            cx = np.sum(influence * pts[:, 0]) / sum_influence
                            cy = np.sum(influence * pts[:, 1]) / sum_influence
                        else:
                            cx, cy = pts[best_idx]
                            
                        dx1 = int(max(0, cx - DENSE_WINDOW_SIZE / 2))
                        dy1 = int(max(0, cy - DENSE_WINDOW_SIZE / 2))
                        dx2 = int(min(w, dx1 + DENSE_WINDOW_SIZE))
                        dy2 = int(min(h, dy1 + DENSE_WINDOW_SIZE))
                        
                        if dx2 - dx1 < DENSE_WINDOW_SIZE: dx1 = max(0, dx2 - DENSE_WINDOW_SIZE)
                        if dy2 - dy1 < DENSE_WINDOW_SIZE: dy1 = max(0, dy2 - DENSE_WINDOW_SIZE)
                        
                        actual_in_window = (pts[:, 0] >= dx1) & (pts[:, 0] <= dx2) & (pts[:, 1] >= dy1) & (pts[:, 1] <= dy2)
                        actual_count = np.sum(actual_in_window)
                        
                        if actual_count >= route_threshold:
                            dense_regions.append((dx1, dy1, dx2, dy2))
                            mask = ~actual_in_window
                            pts = pts[mask]
                            norm_weights = norm_weights[mask]
                            remaining_boxes = [rb for rb in remaining_boxes if not (rb[0] >= dx1 and rb[1] >= dy1 and rb[2] <= dx2 and rb[3] <= dy2)]
                        else: break

            # --- [추론 및 패킹(Packing) 모듈] ---
            unified_infer_list = []
            dense_idx_list = []
            for dx1, dy1, dx2, dy2 in dense_regions:
                unified_infer_list.append(img[dy1:dy2, dx1:dx2])
                dense_idx_list.append((len(unified_infer_list) - 1, dx1, dy1, dx2, dy2))

            canvases, canvas_infos = [], []
            canvas_start_idx = -1
            
            # DAHI Only는 패킹하지 않음
            if method_name != "DAHI Only" and len(remaining_boxes) > 0:
                clustered_boxes = merge_clusters_dynamic(remaining_boxes, w, h, merge_pad=MERGE_PAD)
                crops_to_pack = []
                for cb in clustered_boxes:
                    cx1, cy1, cx2, cy2 = map(int, cb[:4]); cw_org, ch_org = cx2 - cx1, cy2 - cy1
                    scale_ratio = UPSCALE_RATIO if max(cw_org, ch_org) <= UPSCALE_MAX_THRESH else 1.0
                    cw_crop, ch_crop = min(int(cw_org * scale_ratio), CANVAS_SIZE), min(int(ch_org * scale_ratio), CANVAS_SIZE)
                    if cw_crop > 0 and ch_crop > 0:
                        crop_img = img[cy1:cy1+ch_org, cx1:cx1+cw_org]
                        if scale_ratio > 1.0: crop_img = cv2.resize(crop_img, (cw_crop, ch_crop), interpolation=cv2.INTER_CUBIC)
                        else: crop_img = crop_img[:ch_crop, :cw_crop]
                        crops_to_pack.append({'crop': crop_img, 'ox': cx1, 'oy': cy1, 'cw': cw_crop, 'ch': ch_crop, 'scale': scale_ratio})
                
                crops_to_pack.sort(key=lambda x: x['ch'], reverse=True)
                
                current_canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), CANVAS_BG_COLOR, dtype=np.uint8)
                cx, cy, max_h = 0, 0, 0
                for item in crops_to_pack:
                    if cx + item['cw'] > CANVAS_SIZE: cx = 0; cy += max_h + CANVAS_MARGIN; max_h = 0
                    if cy + item['ch'] > CANVAS_SIZE: 
                        canvases.append(current_canvas); current_canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), CANVAS_BG_COLOR, dtype=np.uint8); cx, cy, max_h = 0, 0, 0
                    current_canvas[cy:cy+item['ch'], cx:cx+item['cw']] = item['crop']
                    canvas_infos.append({'c_idx': len(canvases), 'cx1': cx, 'cy1': cy, 'cx2': cx+item['cw'], 'cy2': cy+item['ch'], 'ox': item['ox'], 'oy': item['oy'], 'scale': item['scale']})
                    cx += item['cw'] + CANVAS_MARGIN; max_h = max(max_h, item['ch'])
                if max_h > 0 or cx > 0: 
                    canvases.append(current_canvas)
                
                # 💡 오직 SCE 파이프라인에서만 전방위 여백 확장 수행
                if method_name == "SBSI + DCRP":
                    def get_distribution(gap, avail1, avail2):
                        half = gap // 2
                        if avail1 < half: return avail1, min(gap - avail1, avail2)
                        elif avail2 < half: return min(gap - avail2, avail1), avail2
                        else: return half, gap - half

                    for c_idx, canvas in enumerate(canvases):
                        c_infos = [info for info in canvas_infos if info['c_idx'] == c_idx]
                        if not c_infos: continue
                        unique_cy1s = sorted(list(set([info['cy1'] for info in c_infos])))
                        for i, cy1 in enumerate(unique_cy1s):
                            row_items = [info for info in c_infos if info['cy1'] == cy1]
                            row_items.sort(key=lambda x: x['cx1'])
                            next_cy1 = unique_cy1s[i+1] if i + 1 < len(unique_cy1s) else CANVAS_SIZE
                            for j, info in enumerate(row_items):
                                cx1, cy1 = info['cx1'], info['cy1']
                                cx2, cy2 = info['cx2'], info['cy2']
                                ox, oy, s = info['ox'], info['oy'], info['scale']
                                item_w, item_h = cx2 - cx1, cy2 - cy1
                                org_w, org_h = max(1, int(item_w / s)), max(1, int(item_h / s))
                                
                                next_cx1 = row_items[j+1]['cx1'] if j + 1 < len(row_items) else CANVAS_SIZE
                                alloc_w = next_cx1 - cx1
                                if j + 1 < len(row_items): alloc_w -= CANVAS_MARGIN
                                alloc_h = next_cy1 - cy1
                                if i + 1 < len(unique_cy1s): alloc_h -= CANVAS_MARGIN
                                
                                gap_w, gap_h = max(0, alloc_w - item_w), max(0, alloc_h - item_h)
                                
                                if gap_w > 0 or gap_h > 0:
                                    ext_left, ext_right = get_distribution(int(gap_w/s), ox, w - (ox + org_w))
                                    ext_top, ext_bottom = get_distribution(int(gap_h/s), oy, h - (oy + org_h))
                                    
                                    new_ox, new_oy = ox - ext_left, oy - ext_top
                                    new_org_w, new_org_h = org_w + ext_left + ext_right, org_h + ext_top + ext_bottom
                                    
                                    if new_org_w > 0 and new_org_h > 0:
                                        ext_crop = img[new_oy:new_oy+new_org_h, new_ox:new_ox+new_org_w]
                                        actual_w = int(new_org_w * s) if s > 1.0 else new_org_w
                                        actual_h = int(new_org_h * s) if s > 1.0 else new_org_h
                                        
                                        if ext_crop.size > 0:
                                            if s > 1.0: ext_crop = cv2.resize(ext_crop, (actual_w, actual_h), interpolation=cv2.INTER_CUBIC)
                                            h_crop, w_crop = ext_crop.shape[:2]
                                            y_end = min(cy1 + h_crop, CANVAS_SIZE)
                                            x_end = min(cx1 + w_crop, CANVAS_SIZE)
                                            canvas[cy1:y_end, cx1:x_end] = ext_crop[:y_end-cy1, :x_end-cx1]
                                            
                                            info['ox'], info['oy'] = new_ox, new_oy
                                            info['cx2'], info['cy2'] = cx1 + (x_end - cx1), cy1 + (y_end - cy1)

                    canvases = [c for i, c in enumerate(canvases) if any(info['c_idx'] == i for info in canvas_infos)]
                    for new_idx, old_idx in enumerate(sorted(list(set(info['c_idx'] for info in canvas_infos)))):
                        for info in canvas_infos:
                            if info['c_idx'] == old_idx: info['c_idx'] = new_idx

                if len(canvases) > 0:
                    canvas_start_idx = len(unified_infer_list)
                    unified_infer_list.extend(canvases)

            if len(unified_infer_list) > 0:
                t_inf_start = time.time()
                res_all = m2.predict(unified_infer_list, conf=CONF_TETRIS, verbose=False, batch=16)
                img_inf_time += (time.time() - t_inf_start); img_inf_cnt += len(unified_infer_list)
                
                for d_idx, dx1, dy1, dx2, dy2 in dense_idx_list:
                    cw_dense, ch_dense = dx2 - dx1, dy2 - dy1; res_dense = res_all[d_idx]
                    for b in res_dense.boxes:
                        bx1, by1, bx2, by2 = map(float, b.xyxy[0].tolist()); conf = float(b.conf[0])
                        if bx1 <= 5 or by1 <= 5 or bx2 >= cw_dense - 5 or by2 >= ch_dense - 5: conf *= 0.8 
                        local_boxes.append([bx1+dx1, by1+dy1, bx2+dx1, by2+dy1]); local_scores.append(conf); local_classes.append(int(b.cls[0]))
                
                if canvas_start_idx != -1:
                    res_pack = res_all[canvas_start_idx:]
                    for c_idx, res in enumerate(res_pack):
                        for b in res.boxes:
                            bx1, by1, bx2, by2 = map(float, b.xyxy[0].tolist()); conf = float(b.conf[0])
                            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2 
                            for info in canvas_infos:
                                if info['c_idx'] == c_idx and info['cx1'] <= bcx <= info['cx2'] and info['cy1'] <= bcy <= info['cy2']:
                                    if bx1 <= info['cx1'] + 3 or by1 <= info['cy1'] + 3 or bx2 >= info['cx2'] - 3 or by2 >= info['cy2'] - 3: conf *= 0.8
                                    s = info['scale']
                                    orig_x1 = ((bx1 - info['cx1']) / s) + info['ox']; orig_y1 = ((by1 - info['cy1']) / s) + info['oy']
                                    orig_x2 = ((bx2 - info['cx1']) / s) + info['ox']; orig_y2 = ((by2 - info['cy1']) / s) + info['oy']
                                    local_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2]); local_scores.append(conf); local_classes.append(int(b.cls[0]))
                                    break
                                        
            final_local_preds = []
            for c in set(local_classes):
                c_boxes = [b for j, b in enumerate(local_boxes) if local_classes[j] == c]; c_scores = [s for j, s in enumerate(local_scores) if local_classes[j] == c]
                cv_boxes = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in c_boxes]
                indices = cv2.dnn.NMSBoxes(cv_boxes, c_scores, NMS_CONF_THRESH, NMS_IOU_THRESH)
                if len(indices) > 0:
                    for idx in indices.flatten(): final_local_preds.append([c, c_scores[idx]] + c_boxes[idx])

            combined_boxes = global_final_boxes + [p[2:6] for p in final_local_preds]; combined_scores = global_final_scores + [p[1] for p in final_local_preds]; combined_classes = global_final_classes + [p[0] for p in final_local_preds]
            for c in set(combined_classes):
                c_boxes = [b for j, b in enumerate(combined_boxes) if combined_classes[j] == c]; c_scores = [s for j, s in enumerate(combined_scores) if combined_classes[j] == c]
                cv_boxes = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in c_boxes]
                indices = cv2.dnn.NMSBoxes(cv_boxes, c_scores, NMS_CONF_THRESH, 0.45) 
                if len(indices) > 0:
                    for idx in indices.flatten(): all_preds.append([img_idx, c, c_scores[idx]] + c_boxes[idx])

        # 💡 [버그 수정] 누락된 전체 시간 측정 및 HR/LR 그룹 분류 로직 복구
        img_total_time = time.time() - t_pipe_start
        is_hr = (w * h >= HR_THRESHOLD)
        target_keys = ['ALL', 'HR'] if is_hr else ['ALL', 'LR']
        
        for k in target_keys:
            stats[k]['count'] += 1
            stats[k]['inf_time'] += img_inf_time
            stats[k]['total_time'] += img_total_time
            stats[k]['inf_cnt'] += img_inf_cnt
            stats[k]['indices'].add(img_idx)

    # ---------------------------------------------------------
    # 💡 공식 COCO API 연산 엔진 (PR 커브 추출 지원)
    # ---------------------------------------------------------
    def calc_official_coco_metrics(subset_indices):
        if not subset_indices: return {"AP50:95": 0, "AP50": 0, "AP_small": 0, "AP_medium": 0, "AP_large": 0, "pr_curve": None}
        
        gt_dict = {"images": [], "annotations": [], "categories": []}
        for i in range(10): gt_dict["categories"].append({"id": i, "name": f"class_{i}"})
            
        ann_id = 1
        for img_idx in subset_indices:
            info = img_infos[img_idx]
            gt_dict["images"].append({"id": img_idx, "width": info['w'], "height": info['h'], "file_name": info['name']})
            for gt in all_gts[img_idx]:
                c, x1, y1, x2, y2 = gt
                bw, bh = x2 - x1, y2 - y1
                gt_dict["annotations"].append({"id": ann_id, "image_id": img_idx, "category_id": int(c), "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
                ann_id += 1

        cocoGt = COCO()
        cocoGt.dataset = gt_dict
        cocoGt.createIndex()

        sub_preds = [p for p in all_preds if p[0] in subset_indices]
        pred_list = []
        for pred in sub_preds:
            img_idx, c, score, x1, y1, x2, y2 = pred
            bw, bh = x2 - x1, y2 - y1
            pred_list.append({"image_id": img_idx, "category_id": int(c), "bbox": [x1, y1, bw, bh], "score": float(score)})

        if not pred_list: return {"AP50:95": 0, "AP50": 0, "AP_small": 0, "AP_medium": 0, "AP_large": 0, "pr_curve": None}

        cocoDt = cocoGt.loadRes(pred_list)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.maxDets = [100, 300, 500] 
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        with contextlib.redirect_stdout(io.StringIO()):
            cocoEval.summarize()
            
        pr_curve = None
        if cocoEval.eval is not None:
            # 💡 [핵심] Precision 배열 추출 (IoU=0.5, Area=All, maxDets=500)
            precision = cocoEval.eval['precision'][0, :, :, 0, 2] 
            valid_mask = precision > -1
            pr_curve = np.zeros(101)
            for r_idx in range(101):
                valid_cats = precision[r_idx, valid_mask[r_idx]]
                if len(valid_cats) > 0: pr_curve[r_idx] = np.mean(valid_cats)

        return {
            "AP50:95": cocoEval.stats[0] if len(cocoEval.stats) >= 1 else 0,
            "AP50": cocoEval.stats[1] if len(cocoEval.stats) >= 2 else 0,
            "AP_small": cocoEval.stats[3] if len(cocoEval.stats) >= 4 else 0,  
            "AP_medium": cocoEval.stats[4] if len(cocoEval.stats) >= 5 else 0, 
            "AP_large": cocoEval.stats[5] if len(cocoEval.stats) >= 6 else 0,
            "pr_curve": pr_curve
        }

    # 💡 [버그 수정] HR/LR 딕셔너리 구조 복구
    result_dict = {}
    for group in ['ALL', 'HR', 'LR']:
        c = stats[group]['count']
        res = calc_official_coco_metrics(stats[group]['indices'])
        res['Img_Cnt'] = c
        res['Avg_Inf_Cnt'] = stats[group]['inf_cnt'] / c if c else 0
        res['Avg_Inf_Time'] = (stats[group]['inf_time'] / c) * 1000 if c else 0
        res['Avg_Tot_Time'] = (stats[group]['total_time'] / c) * 1000 if c else 0
        result_dict[group] = res
        
    result_dict['Peak_VRAM'] = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    return result_dict

# =========================================================
# 실행 및 다중 표 그리기 (Official COCO Protocol)
# =========================================================
methods = [
    "UC (2x2 Uniform Crop)", 
    "DAHI Only",
    "Packing Only",
    "DAHI + Packing", 
    "SBSI + Packing",        
    "SBSI + DCRP"   
]

final_stats = {}
for m in methods: 
    final_stats[m] = run_official_ablation_benchmark(m)

print("\n" + "="*145)
print(f"🏆 [Ablation Study] 6단계 종합 성능 검증 결과 (Total {NUM_TEST_IMAGES} Images) 🏆")
print("="*145)
print(f"{'Method':<32} | {'Type':<4} | {'Img':<4} | {'mAP':<6} | {'AP50':<6} | {'APs':<6} | {'APm':<6} | {'APl':<6} | {'Inf Cnt':<7} | {'Inf Time':<9} | {'Tot Time':<9}")
print("-" * 145)

for m in methods:
    groups = final_stats[m]
    for g in ['ALL', 'HR', 'LR']:
        s = groups[g]
        if s['Img_Cnt'] == 0: continue
        
        mAP  = s.get('AP50:95', 0.0)
        ap50 = s.get('AP50', 0.0)
        aps  = s.get('AP_small', 0.0)
        apm  = s.get('AP_medium', 0.0)
        apl  = s.get('AP_large', 0.0)
        
        print(f"{m if g == 'ALL' else '':<32} | {g:<4} | {s['Img_Cnt']:<4} | {mAP:.4f} | {ap50:.4f} | {aps:.4f} | {apm:.4f} | {apl:.4f} | {s['Avg_Inf_Cnt']:4.1f} /i | {s['Avg_Inf_Time']:5.1f} ms | {s['Avg_Tot_Time']:5.1f} ms")
    
    print(f"{'':<32} > Peak VRAM: {groups.get('Peak_VRAM', 0.0):.1f} MB")
    print("-" * 145)


# =========================================================
# 📊 [논문용 시각화] 6단계 성능 증명 종합 그래프 렌더링
# =========================================================
print("\n🎨 [Thesis Visual] 6단계 성능 증명 그래프 렌더링 시작...")
plt.rcParams['font.family'] = 'sans-serif'
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

# 논문 스타일 마커 및 색상
colors = ['#7f7f7f', '#17becf', '#e377c2', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 'v', '^', 's', 'D', '*']

# 1. Precision-Recall Curve (IoU=0.5)
recall_vals = np.linspace(0.0, 1.0, 101)
for idx, m in enumerate(methods):
    # 💡 딕셔너리 구조 복구에 따른 'ALL' 키 접근
    pr = final_stats[m]['ALL'].get('pr_curve') 
    if pr is not None:
        ax1.plot(recall_vals, pr, label=m, color=colors[idx], linewidth=2.5)

ax1.set_title('(a) Precision-Recall Curve @ IoU=0.50', fontsize=16, fontweight='bold')
ax1.set_xlabel('Recall', fontsize=14)
ax1.set_ylabel('Precision', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower left', fontsize=11)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])

# 2. 핵심 지표 (AP50:95, APs, APl) 막대 그래프 비교
metrics_to_plot = ['AP50:95', 'AP_small', 'AP_large']
x = np.arange(len(metrics_to_plot))
width = 0.12

for idx, m in enumerate(methods):
    vals = [final_stats[m]['ALL'][metric] for metric in metrics_to_plot]
    bars = ax2.bar(x + idx*width - (width*5/2), vals, width, label=m, color=colors[idx], alpha=0.9)

ax2.set_title('(b) Average Precision (AP) Comparison', fontsize=16, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['mAP (50:95)', 'AP (Small)', 'AP (Large)'], fontsize=14)
ax2.set_ylabel('Score', fontsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(loc='upper left', fontsize=10)

# 3. 효율성 vs 성능 (Inf Time vs mAP) 분산 그래프
for idx, m in enumerate(methods):
    inf_time = final_stats[m]['ALL']['Avg_Inf_Time']
    map_score = final_stats[m]['ALL']['AP50:95']
    ax3.scatter(inf_time, map_score, color=colors[idx], s=300, marker=markers[idx], label=m, edgecolor='black', zorder=5)
    
ax3.set_title('(c) Efficiency vs. Accuracy Trade-off', fontsize=16, fontweight='bold')
ax3.set_xlabel('Average Inference Time (ms)', fontsize=14)
ax3.set_ylabel('mAP (50:95)', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.show()

# =========================================================
# 💡 [NEW Thesis Visual] SCE 적용 전/후 캔버스 비교 시각화
# =========================================================
if len(sce_vis_samples) > 0:
    print("\n" + "="*145)
    print("📸 [Thesis Visual] Spatial Context Extension (SCE) 효과 증명 (Before vs After) 샘플")
    print("="*145)
    
    sample_count = len(sce_vis_samples)
    fig, axes = plt.subplots(sample_count, 2, figsize=(20, 10 * sample_count))
    plt.rcParams['font.family'] = 'sans-serif'
    
    if sample_count == 1: axes = [axes]

    for i, (img_name, canvas_pre, canvas_post) in enumerate(sce_vis_samples):
        axes[i][0].imshow(canvas_pre)
        axes[i][0].set_title(f"Sample {i+1} (a): Pure Tetris Packing (Baseline)\nSource: {img_name}", fontsize=14)
        axes[i][0].axis('off')
        
        axes[i][1].imshow(canvas_post)
        axes[i][1].set_title(f"Sample {i+1} (b): Our Omnidirectional SCE Canvas\n(Seamless Real-pixel Context Restored)", fontsize=14, fontweight='bold')
        axes[i][1].axis('off')
        
        for ax in axes[i]:
             for spine in ax.spines.values(): spine.set_visible(True)

    plt.tight_layout(pad=3.0)
    plt.show()

# =========================================================
# 📊 [NEW Thesis Visual] 연산 효율성 (속도 & VRAM) 비교 막대그래프
# =========================================================
print("\n🎨 [Thesis Visual] 연산 효율성(VRAM & 속도) 막대그래프 렌더링 시작...")

fig4, (ax_time, ax_vram) = plt.subplots(1, 2, figsize=(18, 7))
plt.rcParams['font.family'] = 'sans-serif'

x = np.arange(len(methods))
width = 0.35

# 긴 Method 이름을 그래프 X축에 맞게 축약
short_names = [m.replace("Ours (", "").replace(")", "").replace("2x2 Uniform Crop", "UC 2x2") for m in methods]

# ---------------------------------------------------------
# 1. 속도 비교 막대그래프 (Inf Time vs Total Time)
# ---------------------------------------------------------
inf_times = [final_stats[m]['ALL']['Avg_Inf_Time'] for m in methods]
tot_times = [final_stats[m]['ALL']['Avg_Tot_Time'] for m in methods]

# 막대 그리기
bars_inf = ax_time.bar(x - width/2, inf_times, width, label='Inference Time', color='#1f77b4', edgecolor='black', alpha=0.85)
bars_tot = ax_time.bar(x + width/2, tot_times, width, label='Total Pipeline Time', color='#aec7e8', edgecolor='black', alpha=0.85)

ax_time.set_title('(a) Processing Time Overhead Analysis', fontsize=16, fontweight='bold')
ax_time.set_xticks(x)
ax_time.set_xticklabels(short_names, rotation=20, ha='right', fontsize=13)
ax_time.set_ylabel('Time (ms)', fontsize=14)
ax_time.grid(axis='y', linestyle='--', alpha=0.7)
ax_time.legend(loc='upper left', fontsize=12)

# 막대 위에 값(ms) 표기
for bar in bars_tot:
    yval = bar.get_height()
    if yval > 0:
        ax_time.text(bar.get_x() + bar.get_width()/2, yval + (max(tot_times)*0.02), f'{yval:.1f}', 
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

# ---------------------------------------------------------
# 2. Peak VRAM 비교 막대그래프
# ---------------------------------------------------------
vrams = [final_stats[m].get('Peak_VRAM', 0) for m in methods]

# 막대 그리기 (주황색 테마)
bars_vram = ax_vram.bar(x, vrams, width*1.2, color='#ff7f0e', edgecolor='black', alpha=0.85)

ax_vram.set_title('(b) Peak VRAM Consumption', fontsize=16, fontweight='bold')
ax_vram.set_xticks(x)
ax_vram.set_xticklabels(short_names, rotation=20, ha='right', fontsize=13)
ax_vram.set_ylabel('Memory (MB)', fontsize=14)
ax_vram.grid(axis='y', linestyle='--', alpha=0.7)

# 막대 위에 값(MB) 표기
for bar in bars_vram:
    yval = bar.get_height()
    if yval > 0:
        ax_vram.text(bar.get_x() + bar.get_width()/2, yval + (max(vrams)*0.02), f'{yval:.1f}', 
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

# 그래프 간격 조절 및 출력
plt.tight_layout(pad=3.0)
plt.show()