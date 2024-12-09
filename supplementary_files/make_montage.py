from PIL import Image
import numpy as np
import os
import glob
import tqdm 
import os
import ipdb
import cv2

Image.MAX_IMAGE_PIXELS = None # to avoid image size warning
data_name = 'CCCP_data'
imgdir = data_name+ '/segmented70_no_thresh'
origdir = data_name+ "/data"
savedir = data_name+ "/montaged_70_no_thresh"

if not os.path.exists(savedir):
    os.mkdir(savedir)


filelist = next(os.walk(imgdir))[1]
print(len(filelist))
print(filelist[0])


total_file=len(filelist)
print('total_file', total_file)

count=1
file_no=1
stride=192
margin=64
for fil in tqdm.tqdm(filelist):
    sub_dir = fil.split('.')[1].split('-')[1]
    # orig_img = Image.open(os.path.join( origdir , str(fil).replace(sub_dir+'_', sub_dir+'/')+".png"))
    orig_file = os.path.join( origdir , 
        ((str(fil).replace(sub_dir+'_', sub_dir+'/')+".png").replace('-', ' - ')).replace('#', ' #') )
    # ipdb.set_trace()
    orig_img = Image.open(orig_file)
    width, height = orig_img.size
    
    R = int(width/192)
    C = int(height/192)
    if (R*stride+margin) > width : R -=1
    if (C*stride+margin) > height : C -=1

    img=Image.new('L', (R*stride+margin, C*stride+margin)) # r = 7, c = 7
    final_image=np.array(img)


    for r in range(R):
        for c in range(C):
            f=imgdir+'/'+ fil+ '/'+ str(count)+'.png'
            count=count+1
            im = Image.open(f).convert('L')
            cropped=np.array(im)
            final_image[c*stride:c*stride+256, r*stride:r*stride+256]+=cropped      # For segmented output
            # final_image[c*stride:c*stride+256, r*stride:r*stride+256]=cropped      # For input
            
    save_img=Image.fromarray(final_image,'L')
    # ipdb.set_trace()
    save_to= os.path.join(orig_file.replace(origdir, savedir))
    if not os.path.exists(os.path.join(savedir, sub_dir)):
        os.mkdir(os.path.join(savedir, sub_dir)) 
    save_img.save(save_to)
    count=1
    file_no+=1