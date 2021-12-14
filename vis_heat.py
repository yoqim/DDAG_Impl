import os,cv2
import numpy as np


def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def save_gradcam(save_suffix, heatmap, raw_image):
    heatmap = cv2.resize(heatmap, (raw_image.shape[1], raw_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    
    hcam = heatmap + np.float32(raw_image)
    hcam = norm_image(hcam)
    cv2.imwrite(save_suffix + '_heat.jpg', hcam)

    print("saved ", save_suffix)




save_dir = './debug/'
img_ids = set([i.strip('.png').split('_')[0] for i in os.listdir(save_dir) if i.endswith('.png')])

for ind, name in enumerate(img_ids):
    rgb_name = name + '_rgb'
    ir_name = name + '_ir'
    sel_id = np.random.randint(0,2048,10)

    ### RGB
    img = cv2.imread(save_dir + rgb_name + '.png')
    h,w = img.shape[:2]
    img = cv2.resize(img, (w//2,h//2))

    heatmap = np.load(save_dir + rgb_name + '.npy')
    heatmap = heatmap.squeeze()

    for fi in sel_id:
        save_suffix = save_dir + rgb_name + '_fea{}'.format(fi)
        save_gradcam(save_suffix,heatmap[fi,...],img)

    ### IR
    img = cv2.imread(save_dir + ir_name + '.png')
    h,w = img.shape[:2]
    img = cv2.resize(img, (w//2,h//2))

    heatmap = np.load(save_dir + ir_name + '.npy')
    heatmap = heatmap.squeeze()
    
    for fi in sel_id:
        save_suffix = save_dir + ir_name + '_fea{}'.format(fi)
        save_gradcam(save_suffix,heatmap[fi,...],img)