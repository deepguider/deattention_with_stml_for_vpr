import cv2
import os
import argparse
import numpy as np
from moviepy.editor import *
from ipdb import set_trace as bp

def get_opt():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in_videopath1', type=str, default='./new_test_ytn.mp4', help = '') 
    parser.add_argument('--in_videopath2', type=str, default='./new_test_ytn.mp4', help = '') 
    parser.add_argument('--out_videopath', type=str, default='./concat_video.avi', help = '') 
    parser.add_argument('--dim', type=int, default=0, help ='Concatenation dimension. 0 for horizontal, and 1 for veritcal concatenation.') 
    parser.add_argument('--fps', type=float, default=0.0, help = '') 
    parser.add_argument('--skip_frame', type=int, default=1, help = '') 
    parser.add_argument('--width', type=int, default=0, help = '') 
    parser.add_argument('--height', type=int, default=0, help = '') 
    parser.add_argument('--ratio', type=float, default=1.0, help = '') 
    parser.add_argument('--disp', action='store_true', help='')
    parser.add_argument('--write_fname_txt', action='store_true', help='')
    opt = parser.parse_args()
    return opt

def get_cap(videopath, verbose=True):
    cap = cv2.VideoCapture(videopath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if verbose == True:
        print("(width, height, length, fps) : ({}, {}, {}, {}) of [{}]".format(w, h, vid_length, fps, videopath))
    return cap, w, h, vid_length, fps

def concat_videos(in_videopath1 = 'video1.avi', in_videopath2 = 'video2.avi', out_videopath="concat.avi", dim=0, output_wh=(0,0), output_ratio=1.0,  output_fps=0.0, skip_frame=0, write_fname_txt=False, disp=False):  # just resize
    '''
    output_wh : is resolution of single video. (out_w , out_h) is concatenated video's resolution.
                output_wh is set to video1' (wh*output_ratio) (in case, output_ratio > 0), when output_wh=(0,0).
    dim == 0 : np.hconcat(video1, video2)
        == 1 : np.vconcat(video1, video2)
    '''

    print("\n\n********\nConcatenate {} and {} in dim [{}] direction (0:hori, 1:vert)".format(in_videopath1, in_videopath2, dim))


    cap1, w1, h1, vid_length1, fps1 = get_cap(in_videopath1)
    cap2, w2, h2, vid_length2, fps2 = get_cap(in_videopath2)

    ## Caculate w,h for video 1
    if output_wh == (0, 0):
        w, h = w1, h1
        if output_ratio > 0.0:
            w, h = int(w*output_ratio), int(h*output_ratio)
        #print("Output resolution (width, height) is set to ({}, {}), which is the resolution of {} [video1] with {} ratio.".format(w, h, in_videopath1, output_ratio))
    else:
        w, h = output_wh[0], output_wh[1]
        #print("target (width, height) is set to ({}, {}), which is user defined resolution".format(w, h))
        assert (w>0 and h>0), "Width and height should be larger than zero for a custom resolution."

    ## Caculate w,h for video 2
    if dim == 0 :  # horizontal direction
        ratio2 = h / h2   # new_h / h2
    else:  # vertical direction
        ratio2 = w / w2   # new_w / w2

    output_wh1 = (w, h)
    output_wh2 = (int(w2*ratio2), int(h2*ratio2))

    if output_fps == 0.0:
        output_fps = fps1


    skip_frame_cnt = skip_frame
    print("Save image every {} frames".format(skip_frame+1))

    vid_length = min(vid_length1, vid_length2)
    print("The video frame length is set to {}, which is the shorter of the two.".format(vid_length))
    print("The video frame rate is set to {}".format(output_fps))

    avi_out = None

    print("Press ESC to quit")
    for framenum in range(0, vid_length):
        print('{}/{}\r'.format(framenum, vid_length), end='')
        cap1.set(cv2.CAP_PROP_FRAME_COUNT, framenum)
        cap2.set(cv2.CAP_PROP_FRAME_COUNT, framenum)

        ret, frame1 = cap1.read()  # (720, 1280, 3)
        if ret is False:
            break

        ret, frame2 = cap2.read()  # (720, 1280, 3)
        if ret is False:
            break

        if skip_frame_cnt > 0:
            skip_frame_cnt -= 1
            continue
        else:
            skip_frame_cnt = skip_frame
    
        # Resize
        frame1 = cv2.resize(frame1, output_wh1)
        frame2 = cv2.resize(frame2, output_wh2)

        # Water-mark
        if write_fname_txt == True:
            x = (int(output_wh[0]*0.5))
            y = 20
            frame1 = cv2.putText(frame1, in_videopath1.split("/")[-1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)
            framw2 = cv2.putText(frame2, in_videopath2.split("/")[-1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)

        if dim == 0:
            out_frame = cv2.hconcat([frame1, frame2])
        elif dim == 1:
            out_frame = cv2.vconcat([frame1, frame2])
    
        # Write out video
        if avi_out == None:
            out_h, out_w = out_frame.shape[:2]
            avi_out = cv2.VideoWriter(out_videopath, 0x7634706d, output_fps, (out_w, out_h)) # write ad mp4
        avi_out.write(out_frame)

        if disp:
            cv2.imshow('frame', out_frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27: # Escape (ESC)
                break
    print('')
    ## Release input videos
    if cap1 is not None:
        cap1.release()
    if cap2 is not None:
        cap2.release()

    ## Release output video
    if avi_out is not None:
        if avi_out.isOpened() == True:
            avi_out.release()

    ## Show output result
    cap, _,_,_,_ = get_cap(out_videopath, verbose=True)  # To print out video's spec.
    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    skip_frame = 0
    opt = get_opt()
    in_videopath1 = opt.in_videopath1
    in_videopath2 = opt.in_videopath2
    out_videopath = opt.out_videopath
    dim = opt.dim
    fps = opt.fps
    disp = opt.disp
    output_wh = (opt.width, opt.height)
    ratio = opt.ratio  # ratio is valid when it > 0.0 and output_wh = (0,0)
    write_fname_txt = opt.write_fname_txt

    ## using opt
    concat_videos(in_videopath1, in_videopath2, out_videopath, dim=dim, output_wh=(0,0), output_ratio=ratio, output_fps=1, skip_frame=0, write_fname_txt=write_fname_txt, disp=disp)

    ## For test
    #concat_videos("tokyo247_test_Fail_BL.avi", "tokyo247_test_All_crn", "concat_hori.avi", dim=0, output_wh=(0,0), output_ratio=1.0, output_fps=1, skip_frame=0, write_fname_txt=write_fname_txt, disp=True)
