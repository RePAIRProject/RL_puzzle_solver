import os
import math
import json
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage
from glob import glob
import pdb
import cv2
import shapely

from puzzle_utils.puzzle_gen import utils
from puzzle_utils.puzzle_gen.vector import Vector
from puzzle_utils.shape_utils import get_cm, get_polygon

class PuzzleGenerator:

    def __init__(self, img, img_name):

        self.img = img
        self.img_size = self.img.shape[:2] # Height, Width, Channel
        self.aspect_ratio = self.img_size[0] / self.img_size[1]
        self.erosion_kernel_size = 7
        self.dilation_kernel_size = 51

        # name of the file without extension
        self.name = img_name

        self.raw_regions = 'data/raw_regions/'
        if not os.path.exists(self.raw_regions):
            os.mkdir(self.raw_regions)

        self.puzzle_folder = 'data/puzzles/'
        if not os.path.exists(self.puzzle_folder):
            os.mkdir(self.puzzle_folder)

    def get_smooth_curve(self, x_len, x_pt_n, x_offset, y_offset, x_step):

        x_arr = []
        y_arr = []

        for i in range(x_pt_n+1):

            if i == 0:
                x = 0
            elif i == x_pt_n:
                x = x_len - 1
            else:
                x = round(x_step * i + random.uniform(-x_offset, x_offset))
            y = round(random.uniform(-y_offset, y_offset))

            x_arr.append(x)
            y_arr.append(y)

        x_arr = list(set(x_arr))
        y_arr = y_arr[:len(x_arr)]
        x_arr.sort()

        #pdb.set_trace()
        if self.smooth_flag:
            if len(x_arr) >= 4:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='cubic')
            elif len(x_arr) == 3:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='quadratic')
            elif len(x_arr) == 2:
                smooth_func = interpolate.interp1d(x_arr, y_arr, kind='slinear')
            else:
                raise ValueError("The length of cutting points in x_arr must be larger than 0.")


        else:
            smooth_func = interpolate.interp1d(x_arr, y_arr, kind='linear')

        x_arr = np.arange(0, x_len, dtype=np.int32)
        y_arr = smooth_func(x_arr).astype(np.int32)
        # plt.plot(x_arr, y_arr, 'r')
        # plt.plot(x_arr_s, y_arr_s, 'b')
        # plt.show()

        return x_arr, y_arr


    def get_mask(self, offset_rate_h, offset_rate_w):

        piece_h = self.img_size[0] / self.h_n
        piece_w = self.img_size[1] / self.w_n

        offset_h = piece_h * offset_rate_h
        offset_w = piece_w * offset_rate_w

        self.mask = utils.new_array(self.img_size, 0)

        # Vertical cuts
        for i in range(1, self.w_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[0], self.h_n, offset_h, offset_w, piece_h)
            y_arr = y_arr + round(i * piece_w)
            y_arr = np.clip(y_arr, 0, self.img_size[1] - 1)

            for j in range(self.img_size[0]):
                self.mask[x_arr[j]][y_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[x_arr[j]][k] = 255

        # Horizontal cuts
        for i in range(1, self.h_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[1], self.w_n, offset_w, offset_h, piece_w)
            y_arr = y_arr + round(i * piece_h)
            y_arr = np.clip(y_arr, 0, self.img_size[0] - 1)

            for j in range(self.img_size[1]):
                self.mask[y_arr[j]][x_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[k][x_arr[j]] = 255

        cv2.imwrite('tmp/mask_init.png', np.array(self.mask, dtype=np.uint8))
        # cv2.imshow('mask', self.mask)
        # cv2.waitKey()

    def get_regions(self):

        dirs = [Vector(0,-1), Vector(0, 1), Vector(-1, 0), Vector(1, 0)] # (x, y)
        small_region_area_limit = self.small_region_area_ratio * \
            self.img_size[0] * self.img_size[1] / (self.w_n * self.h_n)

        mask = np.invert(np.array(self.mask, dtype=np.uint8))

        self.region_cnt, self.region_mat, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
        stats = stats.tolist()

        # Remap region idx
        region_idx_map = -1 * np.ones(self.region_cnt, dtype=np.int32)
        region_new_cnt = 0

        for i in range(1, self.region_cnt):
            if stats[i][4] < small_region_area_limit:
                region_idx_map[i] = -1
            else:
                region_idx_map[i] = region_new_cnt
                region_new_cnt += 1

        self.region_mat = region_idx_map[self.region_mat]
        #print('\tRegion cnt final (raw): %d (%d)' % (region_new_cnt, self.region_cnt - 1))
        self.region_cnt = region_new_cnt

        if self.erosion == 0:
            # Expand valid region to fill out the canvas
            bg_pts = np.transpose(np.nonzero(self.region_mat == -1)).tolist()
            # self.region_mat = self.region_mat.tolist()
            self.region_list = self.region_mat.tolist()
            que = []

            for bg_pt in bg_pts:
                cur_p = Vector(bg_pt[1], bg_pt[0])
                for dir in dirs:
                    next_p = cur_p + dir
                    if utils.check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                        self.region_list[next_p.y][next_p.x] == -1:
                        continue
                    que.append(next_p)

            while len(que) > 0:
                cur_p = que.pop(0)
                for dir in dirs:
                    next_p = cur_p + dir
                    if utils.check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                        self.region_list[next_p.y][next_p.x] != -1:
                        continue
                    self.region_list[next_p.y][next_p.x] = self.region_list[cur_p.y][cur_p.x]
                    que.append(next_p)

            # Check the region mat
            unlabel_pts = np.transpose(np.nonzero(np.ma.masked_equal(self.region_list, -1).mask))
            assert(unlabel_pts.size == 0)

        else: #if self.erosion > 0:
            # pdb.set_trace()
            eroded_region_mat = np.ones_like(self.region_mat) * -1
            for reg_val in range(self.region_cnt): # in np.unique(self.region_mat):
                cur_reg = self.region_mat == reg_val
                # plt.subplot(121)
                # plt.imshow(cur_reg)
                erosion_kernel = np.random.rand(self.erosion_kernel_size, self.erosion_kernel_size)
                eroded_reg = cv2.erode(cur_reg.astype(np.uint8), erosion_kernel, iterations=1)
                eroded_region_mat += eroded_reg * (reg_val+1) # +1 because we start from -1 (see line 188)
                # plt.subplot(122)
                # plt.imshow(eroded_reg)
                # plt.show()
                # pdb.set_trace()
            self.region_mat = eroded_region_mat
            self.region_list = self.region_mat.tolist()
            # TODO
            # if self.erosion == 1:
            #     #

            # elif self.erosion == 2:
            #     #
            # elif self.erosion == 3:
            #     #
            # else:
            #     print('not done yet')


        # for i in range(self.region_cnt):
        #     mask = np.ma.masked_equal(self.region_mat, i).mask.astype(np.uint8)
        #     mask = mask * 255
        #     cv2.imwrite('tmp/' + str(i) + '.png', mask)
        #     cv2.imshow('tmp', mask)
        #     cv2.waitKey(0)


    def save_raw_regions(self, iter):

        file_path = os.path.join(self.raw_regions, '%d.npy' % iter)
        file_path_mat = os.path.join(self.raw_regions, '%d_mat.npy' % iter)
        np.save(file_path, np.array(self.region_list, dtype=np.int32))
        np.save(file_path_mat, self.region_mat)

        f = open(file_path[:-3] + 'txt', 'w')
        f.write(str(self.region_cnt))
        f.close()
        print('\tSave to %s & %d.txt' % (file_path, iter))

    def save_extrapolated_regions(self, iter):

        extrap_folder = os.path.join(self.puzzle_folder, str(iter), 'extrapolated')
        if not os.path.exists(extrap_folder):
            os.mkdir(extrap_folder)
        #pdb.set_trace()
        #region_mat_np = np.array(self.region_mat, np.uint32)
        for reg_val in range(self.region_cnt): # in np.unique(self.region_mat):
            cur_reg = self.region_mat == reg_val
            dilation_kernel = np.random.rand(self.dilation_kernel_size, self.dilation_kernel_size)
            dilated_reg = cv2.dilate(cur_reg.astype(np.uint8), dilation_kernel, iterations=1)
            #dilated_frag = self.img * np.dstack((dilated_reg,dilated_reg,dilated_reg))
            rgba_ex = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)
            rgba_ex[:, :, 3] = 255*(dilated_reg)
            rgba = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = 255*(cur_reg)
            cv2.imwrite(os.path.join(extrap_folder, f'piece-{reg_val}_ex.png'), rgba_ex)
            cv2.imwrite(os.path.join(extrap_folder, f'piece-{reg_val}.png'), rgba)

    def save_jpg_regions(self, folder_path):
        cv2.imwrite(os.path.join(folder_path, 'regions_uint8.png'), self.region_mat)
        # change to cmap='gray' for grayscale color coding
        plt.imsave(os.path.join(folder_path, 'regions_col_coded.jpg'), self.region_mat, cmap='jet')

    def get_pieces_from_puzzle_v2(self):
        pieces = []
        bg_mat = np.zeros_like(self.img)
        h_max = 0
        w_max = 0
        padding = np.min(self.img.shape[:2]) // 30
        for i in range(self.region_cnt):
            mask_i = self.region_mat == i
            if len(self.img.shape) > 2: 
                image_i = self.img * np.repeat(mask_i, self.img.shape[2]).reshape(self.img.shape)
            else:
                image_i = np.where(mask_i, self.img, bg_mat)
            poly_i = get_polygon(mask_i)
            cm_i = get_cm(mask_i)[::-1]
            coords = np.argwhere(mask_i)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            h_i = y1-y0 
            w_i = x1-x0 
            if h_i > h_max:
                h_max = h_i 
            if w_i > w_max:
                w_max = w_i     
            ## centering
            centered_img = np.zeros_like(self.img)
            centered_mask = np.zeros_like(mask_i)
            center_i = np.asarray([self.img.shape[0] / 2, self.img.shape[1] / 2])
            shift2center = (center_i - cm_i)#[::1]
            x0c = np.round(x0+shift2center[1]).astype(int)
            x1c = np.round(x0c + w_i).astype(int)
            y0c = np.round(y0+shift2center[0]).astype(int)
            y1c = np.round(y0c + h_i).astype(int)
            centered_img[y0c:y1c, x0c:x1c] = image_i[y0:y1, x0:x1]
            centered_mask[y0c:y1c, x0c:x1c] = mask_i[y0:y1, x0:x1]
            centered_poly = get_polygon(centered_mask)
            pieces.append({
                'mask': mask_i,
                'centered_mask': centered_mask,
                'image': image_i,
                'centered_image': centered_img,
                'polygon': poly_i,
                'centered_polygon': centered_poly,
                'center_of_mass': cm_i,
                'height': h_i,
                'width': w_i,
                'shift2center': shift2center
            })

        # put pieces inside a square 
        sq_size = max(h_max, w_max) + padding
        hsq = sq_size // 2
        #pdb.set_trace()
        # remember center ordering!
        from_idx = np.round(center_i-hsq).astype(int)
        to_idx = np.round(center_i+hsq).astype(int)
        for i in range(len(pieces)):
            squared_img = np.zeros((sq_size, sq_size, 3))
            squared_img = pieces[i]['centered_image'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            squared_mask = pieces[i]['centered_mask'][from_idx[0]:to_idx[0], from_idx[1]:to_idx[1]]
            # we remove the offset in the centered polygon to get it aligned
            xoffset = (self.img.shape[1]-sq_size) / 2   # half of the distance from the square to the shape of the image!
            yoffset = (self.img.shape[0]-sq_size) / 2
            squared_poly = shapely.affinity.translate(pieces[i]['centered_polygon'], xoff=xoffset, yoff=yoffset)
            pieces[i]['squared_image'] = squared_img
            pieces[i]['squared_mask'] = squared_mask
            pieces[i]['squared_polygon'] = squared_poly
        #pdb.set_trace()

        return pieces, sq_size

    def get_pieces_from_puzzle(self):

        pieces = []
        pieces_shapes = []
        pieces_shapes_cropped = []
        w_max = 0
        h_max = 0
        bg_mat = np.zeros_like(self.img)

        # Compute maximum boundary
        for i in range(self.region_cnt):

            region_i = self.region_mat == i
            #region_map3 = np.repeat(region_map, 3).reshape(self.img.shape)
            image_region_i = np.where(region_i, self.img, bg_mat)
            coords = np.argwhere(region_i)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            image_region_i_cropped = image_region_i[y0:y1, x0:x1]
            region_i_cropped = region_i[y0:y1, x0:x1]
            pieces_shapes_cropped.append(image_region_i_cropped)
            pieces_shapes.append(image_region_i)

            #cm_i = get_cm(region_i)
            cm_i_cropped = get_cm(region_i_cropped)
            #shift2center = cm_i_cropped - ( np.asarray(region_i_cropped.shape[:2]) / 2 )
            pieces.append({
                'mask_orig': region_i,
                'mask_cropped': region_i_cropped,
                # 'mask_center': cm_i,
                # 'mask_cropped_center': cm_i_cropped,
                # 'shift2center_cropped': shift2center,
                'img_cropped': self.img[y0:y1, x0:x1],
                'corner_x':x0,
                'corner_y':y0,
                'height':y1-y0,
                'width':x1-x0
            })
            w_max = max(w_max, x1 - x0)
            h_max = max(h_max, y1 - y0)

        diam = int(math.sqrt(w_max ** 2 + h_max ** 2) + 50)
        # now we create the images
        for i in range(self.region_cnt):

            piece_padded = np.zeros((diam, diam))
            piece_padded[0:pieces[i]['height'], 0:pieces[i]['width']] = pieces_shapes_cropped[i]
            cm = get_cm(piece_padded)
            center = np.asarray([diam / 2, diam / 2])
            shift2center_frag = (center - cm)[::-1]
            cm_i = get_cm(region_i)
            center_i = np.asarray([self.img.shape[0] / 2, self.img.shape[1] / 2])
            shift2center_img = (center_i - cm_i)[::-1]
            piece_centered = np.zeros((diam, diam, 4))
            mask_centered = np.zeros((diam, diam))
            y0c = np.round(shift2center_frag[0]).astype(int)
            y1c = np.round(shift2center_frag[0]+pieces[i]['height']).astype(int)
            x0c = np.round(shift2center_frag[1]).astype(int)
            x1c = np.round(shift2center_frag[1]+pieces[i]['width']).astype(int)
            
            # y0c = np.round((center[0]+shift2center[0]-pieces[i]['height']/2)).astype(int)
            # y1c = np.round((center[0]+shift2center[0]+pieces[i]['height']/2)).astype(int)
            # x0c = np.round((center[1]+shift2center[1]-pieces[i]['width']/2)).astype(int)
            # x1c = np.round((center[1]+shift2center[1]+pieces[i]['width']/2)).astype(int)
            # print('debug values')
            # print(y0c, y1c, x0c, x1c, cm, center, shift2center, pieces[i]['height'], pieces[i]['width'])
            # pdb.set_trace()

            for h in range(3):
                piece_centered[y0c:y1c, x0c:x1c, h] = pieces[i]['img_cropped']
            
            mask_centered[y0c:y1c, x0c:x1c] = pieces[i]['mask_cropped']
            piece_centered[y0c:y1c, x0c:x1c, 3] = pieces[i]['mask_cropped'] * 255
            pieces[i]['img'] = piece_centered
            pieces[i]['mask'] = mask_centered
            pieces[i]['shift2center_frag'] = shift2center_frag
            pieces[i]['shift2center_img'] = shift2center_img
            poly_orig = get_polygon(pieces[i]['mask_orig'])
            pieces[i]['poly_orig'] = poly_orig
            poly = get_polygon(mask_centered)
            pieces[i]['poly_centered'] = poly

        return pieces

            # pad_top = (diam - pieces_shapes_cropped[i].shape[0]) // 2
            # pad_left = (diam - pieces_shapes_cropped[i].shape[1]) // 2
            # pad_bottom = diam - pieces_shapes_cropped[i].shape[0] - pad_top
            # pad_right = diam - pieces_shapes_cropped[i].shape[1] - pad_left

            # print('debug values')
            # print(pad_top, pad_left, pad_bottom, pad_right, pieces_shapes_cropped[i].shape, diam)
            # piece_padded = np.zeros((diam, diam))
            # shift_y, shift_x = np.round(pieces[i]['shift2center_cropped']).astype(int)
            # piece_padded[pad_bottom+shift_y:diam - pad_top+shift_y, pad_left+shift_x:diam - pad_right+shift_x] = pieces_shapes_cropped[i]
            # piece_padded2 = np.zeros((diam, diam))
            # shift_y, shift_x = - np.round(pieces[i]['shift2center_cropped']).astype(int)
            # piece_padded2[pad_bottom+shift_y:diam - pad_top+shift_y, pad_left+shift_x:diam - pad_right+shift_x] = pieces_shapes_cropped[i]
            # piece_padded3 = np.zeros((diam, diam))
            # piece_padded3[pad_bottom:diam - pad_top, pad_left:diam - pad_right] = pieces_shapes_cropped[i]
            # plt.subplot(221)
            # plt.title(f"shift + {get_cm(piece_padded)} // {(diam / 2):.02f}")
            # plt.imshow(piece_padded)
            # plt.subplot(222)
            # plt.title(f"shift - {get_cm(piece_padded2)} // {(diam / 2):.02f}")
            # plt.imshow(piece_padded2)
            # plt.subplot(223)
            # plt.title(f"no shift {get_cm(piece_padded3)} // {(diam / 2):.02f}")
            # plt.imshow(piece_padded3)
            # plt.subplot(224)
            # plt.title(f"new method {get_cm(mask_centered)} // {(diam / 2):.02f}")
            # plt.imshow(mask_centered)
            # plt.show()
            # pdb.set_trace()

            # piece_padded = cv2.copyMakeBorder(pieces_shapes_cropped[i],
            #     pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=bg_color)
            # pieces[i]['img'] = piece_padded
            # pieces[i]['padding'] = [pad_top, pad_left, pad_bottom, pad_right]

        
    # def compute_maximum_boundaries(self, bg_mat):
        
    #     # Compute maximum boundary
    #     for i in range(self.region_cnt):

    #         region_map = self.region_mat == i
    #         region_map3 = np.repeat(region_map, 3).reshape(self.img.shape)
    #         rgb = np.where(region_map3, self.img, bg_mat)

    #         coords = np.argwhere(region_map)
    #         y0, x0 = coords.min(axis=0)
    #         y1, x1 = coords.max(axis=0) + 1

    #         self.w_max = max(w_max, x1 - x0)
    #         self.h_max = max(h_max, y1 - y0)

    def save_puzzle(self, iter, bg_color, save_regions=False):

        pdb.set_trace()
        bg_mat = np.full(self.img.shape, bg_color, np.uint8)
        #region_mat_np = np.array(self.region_mat, np.uint32)

        region_rgbs = []
        w_max = 0
        h_max = 0
        groundtruth = []

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        os.mkdir(puzzle_path)

        pdb.set_trace()
        if save_regions:
            #pdb.set_trace()
            cv2.imwrite(os.path.join(puzzle_path, 'regions_uint8.png'), self.region_mat)
            # change to cmap='gray' for grayscale color coding
            plt.imsave(os.path.join(puzzle_path, 'regions_col_coded.jpg'), self.region_mat, cmap='jet')

        # Compute maximum boundary
        for i in range(self.region_cnt):

            region_map = self.region_mat == i
            region_map3 = np.repeat(region_map, 3).reshape(self.img.shape)
            rgb = np.where(region_map3, self.img, bg_mat)

            coords = np.argwhere(region_map)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            region_rgb = rgb[y0:y1, x0:x1]
            region_rgbs.append(region_rgb)
            groundtruth.append({
                'id': i,
                'dx': int(x0),
                'dy': int(y0)
            })

            w_max = max(w_max, x1 - x0)
            h_max = max(h_max, y1 - y0)
        r = int(math.sqrt(w_max ** 2 + h_max ** 2) + 5)

        groundtruth_path = os.path.join(puzzle_path, 'groundtruth.txt')
        outfile = open(groundtruth_path, 'w')
        
        pdb.set_trace()
        # Compute groundtruth
        # Save groundtruth in txt
        #pdb.set_trace()
        for i in range(self.region_cnt):

            pad_top = (r - region_rgbs[i].shape[0]) // 2
            pad_left = (r - region_rgbs[i].shape[1]) // 2
            pad_bottom = r - region_rgbs[i].shape[0] - pad_top
            pad_right = r - region_rgbs[i].shape[1] - pad_left

            region_pad = cv2.copyMakeBorder(region_rgbs[i],
                pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=bg_color)

            #pdb.set_trace()
            degree = random.uniform(-self.rot_range, self.rot_range)
            # region_rot = ndimage.rotate(region_pad, degree, reshape=False, cval=bg_color)
            rotation_mat = cv2.getRotationMatrix2D((region_pad.shape[1]/2, region_pad.shape[0]/2), degree, 1)
            region_rot = cv2.warpAffine(region_pad, rotation_mat, (region_pad.shape[1], region_pad.shape[0]),
                borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
            if self.alpha_channel:
                rgba = cv2.cvtColor(region_rot, cv2.COLOR_RGB2RGBA)
                rgba[:, :, 3] = 255*(1 - (region_rot[:,:] == bg_color)[:,:,0])
                region_rot = rgba

            cv2.imwrite(os.path.join(puzzle_path, 'piece-%d.png' % i), region_rot)

            groundtruth[i]['dx'] -= pad_left
            groundtruth[i]['dy'] -= pad_top
            groundtruth[i]['dx_region_to_img'] = (bg_mat.shape[1] - r) // 2
            groundtruth[i]['dy_region_to_img'] = (bg_mat.shape[0] - r) // 2
            groundtruth[i]['dx_full'] = groundtruth[i]['dx'] - groundtruth[i]['dx_region_to_img']
            groundtruth[i]['dy_full'] = groundtruth[i]['dy'] - groundtruth[i]['dy_region_to_img']
            groundtruth[i]['rotation'] = degree / 180 * math.pi
            groundtruth[i]['rotation_deg'] = degree

            outfile.write('%d %d %.3f\n' % (groundtruth[i]['dx'], groundtruth[i]['dy'], groundtruth[i]['rotation']))
            # rgb = np.ma.masked_equal(self.region_mat == i, self.img)
            # cv2.imshow('region_rgb', region_rgbs[i])
            # cv2.imshow('region_pad', region_pad)
            # cv2.imshow('region_rot', region_rot)
            # cv2.waitKey()
            # print(rgb)
            # break

        outfile.close()

        #pdb.set_trace()
        # general information about the generation process
        # this will be the starting point to solve the puzzle
        general_info = {
            'name': self.name,
            'orig_img_w': bg_mat.shape[1],
            'orig_img_h': bg_mat.shape[0],
            'region_side': r,
            'ref_fragment': groundtruth[0],
            'regions': self.region_cnt,
            'alpha_channel': self.alpha_channel,
            'rot_range': self.rot_range,
            'small_region_area_ratio': self.small_region_area_ratio,
            'num_of_missing_fragments': int(self.num_of_missing_fragments),
            'missing_indices': [int(ind) for ind in self.missing_indices]
        }

        gt = {
            'info': general_info,
            'fragments': groundtruth
        }
        # Save groundtruth in json
        groundtruth_path = os.path.join(puzzle_path, 'groundtruth.json')
        outfile = open(groundtruth_path, 'w')
        json.dump(groundtruth, outfile, indent=3)
        outfile.close()

        #for gk in general_info.keys(): print(gk, type(general_info[gk]))
        groundtruth_path = os.path.join(puzzle_path, 'groundtruth_extended.json')
        outfile = open(groundtruth_path, 'w')
        json.dump(gt, outfile, indent=3)
        outfile.close()

        # Save config file
        config_path = os.path.join(puzzle_path, 'config.txt')
        outfile = open(config_path, 'w')

        outfile.write('piece-\n') # Prefix
        outfile.write('%d\n' % self.region_cnt) # Piece number
        outfile.write('%d %d %d\n' % (bg_color[0], bg_color[1], bg_color[2])) # bg color in BGR

        outfile.close()

        # save a file for the challenge
        challenge_path = os.path.join(puzzle_path, 'challenge.json')
        outfile = open(challenge_path, 'w')
        json.dump(general_info, outfile, indent=3)
        outfile.close()

    def save_zip(self, iter):

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        zip_path = os.path.join(puzzle_path, 'puzzle-%d.zip' % iter)

        zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        for i in range(self.region_cnt):
            piece_name = 'piece-%d.png' % i
            zipf.write(os.path.join(puzzle_path, piece_name), piece_name)
        zipf.write(os.path.join(puzzle_path, 'config.txt'), 'config.txt')
        zipf.close()

    def save_challenge_zip(self, iter):

        puzzle_path = os.path.join(self.puzzle_folder, str(iter))
        zip_path = os.path.join(puzzle_path, f'challenge-{self.name}-%d.zip' % iter)

        zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        for i in range(self.region_cnt):
            if i not in self.missing_indices:
                piece_name = 'piece-%d.png' % i
                zipf.write(os.path.join(puzzle_path, piece_name), piece_name)
        zipf.write(os.path.join(puzzle_path, 'challenge.json'), 'challenge.json')
        zipf.close()


    def run(self, piece_n, offset_rate_h=0.2, offset_rate_w=0.2, small_region_area_ratio=0.25, rot_range=180,
            smooth_flag=False, alpha_channel=True, perc_missing_fragments=0, erosion=0, borders=False):

        self.rot_range = rot_range
        self.piece_n = piece_n
        self.w_n = math.floor(math.sqrt(piece_n / self.aspect_ratio))
        self.h_n = math.floor(self.w_n * self.aspect_ratio)
        self.smooth_flag = smooth_flag
        self.alpha_channel = alpha_channel
        self.small_region_area_ratio = small_region_area_ratio
        self.missing_indices = []
        self.erosion = erosion
        self.borders = borders

        # print('\tInitial block in hori: %d, in vert: %d' % (self.w_n, self.h_n))
        # print('\tOffset rate h: %.2f, w: %.2f, small region: %.2f, rot: %.2f' %
        #     (offset_rate_h, offset_rate_w, small_region_area_ratio, rot_range))

        self.get_mask(offset_rate_h, offset_rate_w)
        self.get_regions()

        self.num_of_missing_fragments = np.floor(self.region_cnt * perc_missing_fragments / 100).astype(int)
        if self.num_of_missing_fragments > 0:
            self.missing_indices = random.sample(set(np.arange(1, self.region_cnt)), self.num_of_missing_fragments)
            self.missing_indices = np.sort([int(ind) for ind in self.missing_indices])

    def save(self, bg_color=(0,0,0), save_regions=False):

        exist_data_len = len(glob(os.path.join(self.raw_regions, '*.npy')))
        self.save_raw_regions(exist_data_len)
        self.save_puzzle(exist_data_len, bg_color, save_regions)
        if self.borders:
            self.save_extrapolated_regions(exist_data_len)
        self.save_zip(exist_data_len)
        self.save_challenge_zip(exist_data_len)
