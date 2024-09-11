'''Uses command line flags to turn specified image.png files
into either a gif or a video

Returns:
    out.gif or out.avi
'''
from PIL import Image
import glob
import os
import cv2
import argparse
import re
import logging


def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gif','-g', action='store_true',
                                help='Create a gif, default off')
    parser.add_argument('--video','-v', action='store_true',
                                help='Create a video, default off')
    parser.add_argument('--frame','-f',type=int, default=10,
                                help='Frames per second if you want to specify')
    parser.add_argument('--time', '-t', type=float,
                        help='Total time (in seconds) for the GIF or video. Takes priority over --frame if specified.')
    parser.add_argument('--image','-i',type=str, required=True,
                                help='Name of image files to glob.glob(*__*.png) find and read.')
    parser.add_argument('--out','-o',type=str,
                                help='Name of output')
    parser.add_argument('--opt',action='store_true',
                                help='Optimize parameter for gif, use for smaller file size')
    parser.add_argument('--verbose',action='store_true',
                                help='Print verbose output')
    cl_args = parser.parse_args()
    return cl_args


# For sorting to deal with no leading zeros
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    '''Sorts the file names naturally to account for lack of leading zeros
    use this function in listname.sort(key=natural_sort_key)

    Args:
        s: files/iterator
        _nsre: _description_. Defaults to re.compile('([0-9]+)').

    Returns:
        Sorted data
    '''
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def setup_logging(cl_args):
    global pt, verb
    pt = logging.warning
    verb = logging.info

    # Toggle verbose
    if cl_args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')
    verb('Verbose Logging Enabled')



if __name__ == "__main__":
    # print("__main__ Start")
    cl_args = parseArgs()
    setup_logging(cl_args)

    img_names = []
    for file in glob.glob('*'+cl_args.image + '*.png'):
        img_names.append(file)

    img_names.sort(key=natural_sort_key)
    pt('Using image files: '+img_names[0])

    if cl_args.out == None:
        cl_args.out = img_names[0].rsplit('.',1)[0]
        pt('Output named: '+cl_args.out)

    num_images = len(img_names)

    # Calculate FPS based on the --time option if provided, otherwise use --frame
    if cl_args.time:
        fps = num_images / cl_args.time  # frames per second
        verb(f"Time specified: {cl_args.time} seconds. Calculated FPS: {fps}")
    else:
        fps = cl_args.frame
        verb(f"Frame rate specified: {fps} FPS")


    if cl_args.gif:
        verb('Making a gif')
        frames =[]
        for img in img_names:
            frames += [Image.open(img)]
        #EACH FRAME LASTS 1000/FPS MILLISECONDS
        frame_duration = 1000/fps
        #SAVE PIL.IMAGE SEQUENCE INTO A GIF --> TURN ON OPTIMIZE FOR SMALLER FILE SIZE
        frames[0].save(cl_args.out+'.gif',save_all=True,optimize=cl_args.opt,
                    append_images=frames[1:],loop=0,duration=frame_duration)
        verb('GIF made!')

    if cl_args.video:
        verb('Making a video')
        frames =[]
        for img in img_names:
            frames += [cv2.imread(img)]
        #GET DIMENSIONS OF FRAME
        h,w,c = frames[0].shape
        #FOURCC OBJECT ENCODES THE IMAGE SEQUENCE INTO REQUIRED FORMAT
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        #VIDEO WRITER OBJECT TO ACTUALLY WRITE THE VIDEO
        out = cv2.VideoWriter(cl_args.out+'.avi',fourcc,fps,(w,h))

        #WRITE FRAMES TO VIDEO FILE
        for frame in frames:
            out.write(frame)

        #RELEASE VIDEO WRITER OBJECT (NOT NECESSARY BUT GOOD PRACTICE)
        out.release()
        verb('Video Made!')

# quit()
# quit()
