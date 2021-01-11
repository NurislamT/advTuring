import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm
import seaborn as sns

def draw_results(img, dets=None, label=None, ax=None):
    """Draw one image and inference results for it"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    if img is not None:
        ax.imshow(img)
    ax.axis('off')
    if label is not None:
        ax.set_title(label)

    if dets is not None:
        for b in dets:
            text = "{:.4f}".format(b[4])
            bbox = list(map(int, b[:4]))

            left = bbox[0]
            bottom = bbox[1]
            right = bbox[2]
            top = bbox[3]

            # Create a Rectangle patch
            rect = patches.Rectangle(
                #bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1],
                (left, bottom), right - left, top - bottom,
                linewidth=4, edgecolor='r', facecolor='none'
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(
                right, top, text,
                horizontalalignment='right',
                verticalalignment='bottom', 
                color='white',
                backgroundcolor='red',
                fontsize='large'
            )

            landmarks = np.array([(b[i], b[i + 1]) for i in range(5, len(b), 2)])
            colors = cm.rainbow(np.linspace(0, 1, len(landmarks)))
            for l, c in zip(landmarks, colors):
                ax.scatter(l[0], l[1], color=c, marker='o', s=80, edgecolor='black')
          
        
def draw_imgs(imgs, dets=None, labels=None, width=5, show=True):
    """Draw several images with inference results"""
    if labels is None:
        labels = [None] * len(imgs)
    if dets is None:
        dets = [None] * len(imgs)
    
    width = min(len(imgs), width)
    height = (len(imgs) + width - 1) // width
    fig, ax = plt.subplots(height, width, figsize=(width * 5, height * 5))
    if height == 1:
        ax = [ax]
    if width == 1:
        ax = [ax]
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if idx < len(imgs):
                draw_results(imgs[idx], dets[idx], labels[idx], ax=ax[i][j])
            else:
                draw_results(None, None, None, ax=ax[i][j])
    if show:
        plt.show()
    

def draw_attack_stats(rec):
    """Draw attack noise distribution and heatmap"""
    int_attack = rec.astype(np.int)
    plt.hist(int_attack.flatten(), bins=np.unique(int_attack) + 0.5, edgecolor='white')
    plt.xticks(np.unique(int_attack))
    plt.title('Distribution of values')
    plt.show()
    
    n_channels = rec.shape[-1]
    fig, ax = plt.subplots(1, n_channels, figsize=(15, 4))
    for channel in range(n_channels):
        ax[channel].set_title(f'Generated adversarial noise, channel {channel}')
        sns.heatmap(rec[..., channel], ax=ax[channel])
    plt.show()
    
def maps_img(maps):
    if maps.shape[0] == 3:
        maps = np.stack(maps, axis=-1)
    return (maps - maps.min()) / (maps.max() - maps.min())