'''Reads the ctrline.csv files for all the timesteps and determines the pinch off time

Returns:
    Calculated pore pinchoff time
'''
from PIL import Image
import glob
import os
import cv2
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def parseArgs():
    '''Parse command line arguments

    Returns:
        cl_args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold','-t', type=float, default=0.01,
                                help='Threshold of center point to define pinched off, default 0.01')
    parser.add_argument('--center','-c', type=float, default=150,
                                help='Coordinate of pinchoff measurement, default 150')
    parser.add_argument('--input','-i',type=str,
                                help='Name of centerline csv files to glob.glob(*__*.csv) find and read.')
    parser.add_argument('--plot','-p',action='store_true',
                                help='Plot the centerline or not, default False')
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


def center_val(csvname,ctr_coord):
    df = pd.read_csv(csvname)
    if df.empty:
        return 0
    df_closest = df.iloc[(df['id']-ctr_coord).abs().argsort()[:1]]
    return df_closest['phi'].iloc[0]

def plot_timestep(csvname):
    df = pd.read_csv(csvname)
    fig, ax = plt.subplots()
    ax.plot(df.id,df.phi)
    ax.set_ylim([0,1])
    ax.set_xlabel('Y Distance')
    ax.set_ylabel('Phi Centerline')

    image = fig2rgb_array(fig)
    # canvas.draw()  # Draw the canvas, cache the renderer

    # image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    # image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    return image

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


if __name__ == "__main__":
    print("__main__ Start")
    cl_args = parseArgs()

    csv_names = []
    for file in glob.glob('*'+cl_args.input + '*.csv'):
        if not 'total' in file:
            csv_names.append(file)

    csv_names.sort(key=natural_sort_key)
    print('Using csv files: '+csv_names[0])
    output = csv_names[0].rsplit('_', 1)[0] + '_total'
    print('Output name: '+output)

    phi = []
    for csv in csv_names:
        phi.append(center_val(csv,cl_args.center))

    df_out = pd.DataFrame(phi,columns=['center_phi'])
    df_out.to_csv(output+'.csv')
    print('Saved to csv')

    if cl_args.plot:
        print('Plotting ')
        frames = []
        for csv in csv_names:
            frames += [plot_timestep(csv)]
        #GET DIMENSIONS OF FRAME
        h,w,c = frames[0].shape
        #FOURCC OBJECT ENCODES THE IMAGE SEQUENCE INTO REQUIRED FORMAT
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        #VIDEO WRITER OBJECT TO ACTUALLY WRITE THE VIDEO
        out = cv2.VideoWriter(output+'.avi',fourcc,10,(w,h))

        #WRITE FRAMES TO VIDEO FILE
        for frame in frames:
            out.write(frame)

        #RELEASE VIDEO WRITER OBJECT (NOT NECESSARY BUT GOOD PRACTICE)
        out.release()
        print('Video Made!')




# quit()
# quit()
