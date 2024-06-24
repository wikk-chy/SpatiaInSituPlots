import numpy as np
import pandas as pd
from aicsimageio import imread
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.segmentation import flood, flood_fill
from skimage.color import label2rgb
from skimage.morphology import erosion
from skimage.morphology import square, octagon
import scipy.ndimage as ndi

def get_new_mask(mask, df):
    ccs = regionprops(mask)
    val2cc = {}
    for cc in ccs:
        c_ = list(cc.centroid)
        val = mask[int(c_[0]), int(c_[1])]
        val2cc[val] = cc
    new_mask = np.zeros(mask.shape, dtype=mask.dtype)
    for _, row in df.iterrows():
        tp = row['cluster']
        tp_idx = cell_types.index(tp) + 1
        c_ = [int(float(i)) for i in row['cell'].strip("()").split(",")]
        val = mask[c_[0], c_[1]]
        if val > 0:
            cc = val2cc[val]
            new_mask[cc.coords[:, 0], cc.coords[:, 1]] = tp_idx
        else:
            print(val)
    return new_mask

def hex_to_array(h):
    h = h.lstrip("#")
    l = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    return np.array(l)

def get_border(mask):
    border = mask - erosion(mask, octagon(3, 1))
    #border = mask - erosion(mask)
    border = border > 0
    return border

def draw_type_mask(ax, mask, colors, labels, output_path, bg_color="#000000", back=None, border=None):
    colors_ = [hex_to_array(i) for i in colors]
    bg_color = hex_to_array(bg_color)
    rgb_img = label2rgb(mask, colors=colors_, bg_color=bg_color)
    if border is not None:
        rgb_img[border] = bg_color
    if back is not None:
        back_ratio = 0.4
        back_ = (back ^ back.min()) / (back.max() ^ back.min())
        back_ = np.stack([back_, back_, back_], axis=2)
        img = back_ * 0.8 + rgb_img * 0.8
        ax.imshow(img)
        ax.imshow(back, cmap="gray", alpha=0.5)
        ax.imshow(rgb_img, alpha=0.75)
    else:
        ax.imshow(rgb_img)
    ax.grid(False)
    # Add legend
    patches = [plt.Rectangle((0,0),1,1,fc=color) for color in colors]
    ax.legend(patches, labels, loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    ax.axis('off')  # Do not display x and y axes
    ax.grid(False)
    plt.savefig(output_path, format='pdf')
    plt.show()
