from PIL import Image
import glob
import os
import cv2
import argparse
import re



parser = argparse.ArgumentParser()
parser.add_argument('--gif','-g', action='store_true',
                            help='Create a gif, default off')
parser.add_argument('--video','-v', action='store_true',
                            help='Create a video, default off')
parser.add_argument('--frame','-f',type=int, default=10,
                            help='Frames per second if you want to specify')
parser.add_argument('--image','-i',type=str, required=True,
                            help='Name of image files to glob.glob(*__*.png) find and read.')
parser.add_argument('--out','-o',type=str,
                            help='Name of output')
parser.add_argument('--opt',action='store_true',
                            help='Optimize parameter for gif, use for smaller file size')
cl_args = parser.parse_args()


# For sorting to deal with no leading zeros
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


img_names = []
for file in glob.glob('*'+cl_args.image + '*.png'):
    img_names.append(file)

img_names.sort(key=natural_sort_key)
print('Using image files: '+img_names[0])

if cl_args.out == None:
    cl_args.out = img_names[0].rsplit('.',1)[0]
print('Output named: '+cl_args.out)


if cl_args.gif:
    print('Making a gif')
    frames =[]
    for img in img_names:
        frames += [Image.open(img)]
    #EACH FRAME LASTS 1000/FPS MILLISECONDS
    frame_duration = 1000/cl_args.frame
    #SAVE PIL.IMAGE SEQUENCE INTO A GIF --> TURN ON OPTIMIZE FOR SMALLER FILE SIZE
    frames[0].save(cl_args.out+'.gif',save_all=True,optimize=cl_args.opt,
                   append_images=frames[1:],loop=0,duration=frame_duration)
    print('GIF made!')

if cl_args.video:
    print('Making a video')
    frames =[]
    for img in img_names:
        frames += [cv2.imread(img)]
    #GET DIMENSIONS OF FRAME
    h,w,c = frames[0].shape
    #FOURCC OBJECT ENCODES THE IMAGE SEQUENCE INTO REQUIRED FORMAT
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #VIDEO WRITER OBJECT TO ACTUALLY WRITE THE VIDEO
    out = cv2.VideoWriter(cl_args.out+'.avi',fourcc,cl_args.frame,(w,h))

    #WRITE FRAMES TO VIDEO FILE
    for frame in frames:
        out.write(frame)

    #RELEASE VIDEO WRITER OBJECT (NOT NECESSARY BUT GOOD PRACTICE)
    out.release()
    print('Video Made!')

quit()
quit()
