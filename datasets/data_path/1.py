import os
import os.path as osp
import numpy as np
from collections import defaultdict

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_root = '/root/autodl-tmp/data1/datasets/data_path/MOT17/images/train'
label_root = '/root/autodl-tmp/data1/datasets/data_path/MOT17/labels_with_ids/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root) if not s.startswith('.')]

for seq in seqs:
    seq_info_path = osp.join(seq_root, seq, 'seqinfo.ini')
    with open(seq_info_path, 'r') as f:
        seq_info = f.read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    # 按帧分组
    frames_dict = defaultdict(list)
    for row in gt:
        fid, tid, x, y, w, h, mark, _, _ = row
        if mark == 0:
            continue
        frames_dict[int(fid)].append((int(tid), x, y, w, h))

    # 生成 labels_with_ids
    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid in sorted(frames_dict.keys()):
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        lines = []
        for tid, x, y, w, h in frames_dict[fid]:
            # 转为中心点并归一化
            x_c = (x + w / 2) / seq_width
            y_c = (y + h / 2) / seq_height
            w_n = w / seq_width
            h_n = h / seq_height
            lines.append('0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(tid, x_c, y_c, w_n, h_n))
        with open(label_fpath, 'w') as f:
            f.writelines(lines)

print("✅ labels_with_ids 已生成完成")

