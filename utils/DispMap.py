import cv2
import utm
import imutils
import copy
import os
import folium
from ipdb import set_trace as bp
import yaml
import numpy as np

## Author : ccsmm78@gmail.com

class DispMap:
    ''' DispMap class including utm coordinates of boundary.'''
    def __init__(self, cv_img, ltop=(0, 0), rbottom=(0, 0), incoord='latlon', width=1000, coord_vflip=False, coord_hflip=False):
        '''
        ltop and rbottom are coordinates of GPS in lat/lon or utm.
        '''
        if type(cv_img) is type("img.jpg"):  # string type vs. np.array (cv2 image)
            map_path = cv_img
            if os.path.exists(map_path):
                cv_img = cv2.imread(map_path)
                print("Reading map image : ", map_path)
            else:
                print("               Map image ({}) was Not found to save avi. We will use blank image as map.".format(map_path))    
                cv_img = np.zeros((480,640,3))
                
        cv_img = imutils.resize(cv_img, width=width)
        self.cv_img_backup = copy.copy(cv_img)
        self.cv_img = cv_img
        self.coord_vflip = coord_vflip
        self.coord_hflip = coord_hflip
        ltop, rbottom = self.flip_coordinate(ltop, rbottom, coord_vflip, coord_hflip)
        self.ltop = ltop
        self.rbottom = rbottom
        self.incoord = incoord
        (self.h, self.w, _) = self.cv_img.shape
        self.x1, self.y1 = self.get_utm(ltop, incoord)
        self.x2, self.y2 = self.get_utm(rbottom, incoord)
        self.utm_ltop = (self.x1, self.y1)
        self.utm_rbottom = (self.x2, self.y2)
        self.init_homography(incoord)

    def flip_coordinate(self, ltop, rbottom, coord_vflip=False, coord_hflip=False):
        ltop_x, ltop_y = ltop
        rbottom_x, rbottom_y = rbottom

        ## Normal coordinates increase from left-top to right-bottom.
        '''pts1 .... pts2
                ....
           pts3 .... pts4
        '''
        tmp_pts1 = [ltop_x, ltop_y]
        tmp_pts2 = [rbottom_x, ltop_y]
        tmp_pts3 = [ltop_x,    rbottom_y]
        tmp_pts4 = [rbottom_x, rbottom_y]
        if coord_vflip == True and coord_hflip==False:  # y-axis flip
            pts1 = tmp_pts3
            pts2 = tmp_pts4
            pts3 = tmp_pts1
            pts4 = tmp_pts2
        elif coord_vflip == False and coord_hflip==True:  # h-axis flip
            pts1 = tmp_pts2
            pts2 = tmp_pts1
            pts3 = tmp_pts4
            pts4 = tmp_pts3
        elif coord_vflip == True and coord_hflip==True:  # v-axis and h-axis flip
            pts1 = tmp_pts4
            pts2 = tmp_pts3
            pts3 = tmp_pts2
            pts4 = tmp_pts1
        else:  # normal
            pts1 = tmp_pts1
            pts2 = tmp_pts2
            pts3 = tmp_pts3
            pts4 = tmp_pts4
        ret_ltop, ret_rbottom = pts1, pts4
        return ret_ltop, ret_rbottom

    def get_4points(self, ltop, rbottom, coord_vflip=False, coord_hflip=False ):
        '''
        ltop_x, ltop_y = 0, 0
        rbottom_x, rbottom_y = self.w-1, self.h-1
        '''
        ltop, rbottom = self.flip_coordinate(ltop, rbottom, coord_vflip, coord_hflip)
        ltop_x, ltop_y = ltop
        rbottom_x, rbottom_y = rbottom
        ## Normal coordinates increase from left-top to right-bottom.
        '''pts1 .... pts2
                ....
           pts3 .... pts4
        '''
        pts1 = [ltop_x, ltop_y]
        pts2 = [rbottom_x, ltop_y]
        pts3 = [ltop_x,    rbottom_y]
        pts4 = [rbottom_x, rbottom_y]
        return np.array([pts1,pts2,pts3,pts4])

    def set_utm_zone(self, zone_num=52, zone_char='S'):
        self.utm_zone_num = zone_num
        self.utm_zone_char = zone_char

    def get_utm(self, xy=(0, 0), incoord='latlon'):
        if incoord.lower() == 'latlon':
            x, y, self.utm_zone_num, self.utm_zone_char = utm.from_latlon(xy[0], xy[1])
        else:  # 'utm'
            x, y = xy
        return x, y

    def init_homography(self, incoord='latlon'):
        ''' utm coord increase in up-direction(vflip) while img coord increase in down-direction(normal).'''
        pts_img = self.get_4points((0,0), (self.w-1, self.h-1))
        pts_utm = self.get_4points(self.utm_ltop, self.utm_rbottom, coord_vflip=self.coord_vflip)

        self.H_utm2img, _ = cv2.findHomography(pts_utm, pts_img, cv2.RANSAC)
        self.H_img2utm, _ = cv2.findHomography(pts_img, pts_utm, cv2.RANSAC)

    def get_img_coord(self, xy=(0, 0), incoord='latlon'):
        x, y = self.get_utm(xy, incoord)  # convert to utm_x, utm_y
        new_coord = np.dot(self.H_utm2img, np.array([x,y,1])).astype(np.int)
        new_x, new_y = new_coord[0], new_coord[1]
        return new_x, new_y
    
    def get_map_coord(self, xy=(0, 0), outcoord='latlon'):  # convert image coordinate to map coordinate(utm)
        x, y = xy
        new_coord = np.dot(self.H_img2utm, np.array([x,y,1])).astype(np.int)
        new_x, new_y = new_coord[0], new_coord[1]
        if outcoord.lower() == 'latlon':
            new_x, new_y = utm.to_latlon(new_x, new_y, self.utm_zone_num, self.utm_zone_char)
        return new_x, new_y

    def get_img_coord_old(self, xy=(0, 0), incoord='latlon'):
        ''' utm_y increase toward up direction while image_y increase toward down direction.
        '''
        x, y = self.get_utm(xy, incoord)
        x = self.y_is_ax_b(self.x1, self.x2, self.w, x)
        y = self.y_is_ax_b(self.y1, self.y2, self.h, y, direction=1)
        return x, y

    def get_map_coord_old(self, xy=(0, 0), outcoord='latlon'):  # convert image coordinate to map coordinate(utm)
        ''' utm_y increase toward up direction while image_y increase toward down direction.
        '''
        (x, y) = xy
        x = self.inv_y_is_ax_b(self.x1, self.x2, self.w, x)
        y = self.inv_y_is_ax_b(self.y1, self.y2, self.h, y, direction=1)
        if outcoord.lower() == 'latlon':
            x, y = utm.to_latlon(x, y, self.utm_zone_num, self.utm_zone_char)
        return x, y

    def loopback_img_coord_conversion(self, xy=(0, 0), incoord='latlon'):  # For verification
        return self.get_img_coord(self.get_map_coord(xy, incoord), incoord)

    def loopback_utm_coord_conversion(self, xy=(0, 0), incoord='latlon'):  # For verification
        return self.get_map_coord(self.get_img_coord(xy, incoord), incoord)

    def y_is_ax_b(self, x1, x2, w, x, direction=0):
        a = w/(x2 - x1)  # a = 1279/(x2-x1)
        b = -1*a*x1
        y = a*x + b
        if direction is not 0:
            y = w - y
        return int(y)

    def inv_y_is_ax_b(self, x1, x2, w, y, direction=0):
        a = w/(x2 - x1)  # a = 1279/(x2-x1)
        b = -1*a*x1
        x = (y-b) / a
        if direction is not 0:  # To do : it this true? I didn't prove this code
            x = w - x
        return int(x)

    def draw_point_on_map(self, xy=(0, 0), incoord='latlon',
            radius=1, color=(255, 0, 0),  # BGR
            thickness=-1  #  -1 : fill, positive: thick
            ):
        # (584735.1354362543, 4477045.784565611, 17, 'T')  # utm.from_latlon(40.4397456, -80.0008864)
        # (585298.0976780445, 4476587.395387753, 17, 'T')  # utm.from_latlon(40.435559, -79.994311)
        x, y = self.get_img_coord(xy, incoord)
        x, y = int(x), int(y)
        cv2.circle(self.cv_img, (x, y), radius, color, thickness)
        return self.cv_img

    def set_img(self, cv_img):
        self.cv_img = cv_img

    def get_img(self):
        return self.cv_img

    def refresh_img(self):
        self.cv_img = copy.copy(self.cv_img_backup)

def write_pose_to_txt(Xs, Ys):
    f = open("poses_latlon.py", 'w')
    data = "    pts = [\n"
    f.write(data)
    for xy in zip(Xs, Ys):
        x, y = mMap.get_map_coord(xy)
        data = "        ({}, {}),".format(x, y)
        data = "%d번째 줄입니다.\n" % i
        f.write(data)
    data = "    ]"
    f.write(data)
    f.close()

def get_latlon_from_line(line, sep=','):
    # line : 
    # 000000, 36.3798158583, 127.367339298
    try:
        line = line.split(sep)
        lat = float(line[1])
        lon = float(line[2])
    except:
        bp()
    return lat, lon

def get_latlon_from_txtfile(poseFile, sep=','):
    txt_lines = open(poseFile,'r').readlines()
    latlon = []
    for i, line in enumerate(txt_lines):
        # line : 
        # 000000, 36.3798158583, 127.367339298
        lat, lon = get_latlon_from_line(line, sep=sep)
        latlon.append((lat, lon))
    return latlon

def get_xy_from_line(line, sep=','):
    return get_latlon_from_line(line, sep)

def get_xyArray_from_txtfile(poseFile, sep=','):
    return get_latlon_from_txtfile(poseFile, sep)

def draw_pose_txtfile(poseFile, myMap, radius = 1, color = 'blue', fill = True, sep=','):
    txt_lines = open(poseFile,'r').readlines()
    # file content :
        # 000000, 36.3798158583, 127.367339298
        # 000000, 36.3798158583, 127.367339298
        # ...


    for i, line in enumerate(txt_lines):
        lat, lon = get_latlon_from_line(line, sep)

        folium.CircleMarker(
                location = [lat, lon],
                radius = radius,
                color = color,
                fill = fill
                ).add_to(myMap)
        if False:
            folium.Marker(
                    location = [lat, lon],
                    popup ="{}".format(line.split(sep)[0]),
                    icon = folium.Icon(color='red', icon='star')
                    ).add_to(myMap)
    return myMap

if __name__ == '__main__':
    ## You can crop map image from naver or google map.
    ## At that time, write down latitude,logitude of left-top, and right-bottom points
    map_path='../img/etrib12.png'
    map_ltop = (36.38079, 127.36720)
    map_rbottom = (36.37979, 127.36828)
    #map_ltop_utm = utm.from_latlon(map_ltop[0], map_ltop[1])[0:2]

    print("Please wait. It takes seconds to load a map...")

    pts = [ 
            (36.38026738466971,127.36765671058822),
            (36.38010326694693,127.3677371768586)
    ]   

    map_img0 = cv2.imread(map_path)
    map_img0 = imutils.resize(map_img0, width=1000)
    map_img = copy.copy(map_img0)
    
    mMap = DispMap(map_path, ltop=map_ltop, rbottom=map_rbottom, width=1000)
    #mMap = DispMap(map_img, ltop=map_ltop, rbottom=map_rbottom, width=1000)

    ## Display circle of points on the map
    for p in pts:
        map_img = mMap.draw_point_on_map(xy=p, radius=10)
        ix, iy = mMap.get_img_coord(p)
        mx, my = mMap.get_map_coord((ix,iy))
        print("map pose <==> img pose (x,y) : {} <==> {}".format((mx,my), (ix,iy)))
    
    cv2.imshow('Map', map_img)
    cv2.waitKey(0)
    

