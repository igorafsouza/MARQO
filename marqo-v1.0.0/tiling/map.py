
# whole namespaces
import os
import cv2
import yaml
import json
import large_image
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


# specific functions form namespaces
from typing import List

# pipeline classes and functions

class TileMap:
    """
    """

    def __init__(self):
        pass


    def set_parameters(self, config_file):
        """
        Read and fetch the paramaters and names needed to execute the masking.

        Paramters:
            - 

        """
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        self.output_resolution = config_data['parameters']['image']['output_resolution']
        self.scale_factor = config_data['parameters']['image']['scale_factor']
        self.tile_dimension = config_data['parameters']['image']['tile_dimension']        
        
        self.marker_index_maskforclipping = config_data['parameters']['tiling']['marker_index_maskforclipping']
        self.tile_analysis_thresh = config_data['parameters']['tiling']['tile_analysis_thresh']
        self.erosion_buffer_microns = config_data['parameters']['tiling']['erosion_buffer_microns']
        
        self.raw_images_path = config_data['sample']['raw_images_full_path']
        self.sample_name = config_data['sample']['name']


        self.marco_output_path = config_data['directories']['marco_output']


    def get_tiling_map_(self, registered_masks: List, tissue_masks: List):
        """
        
        in:
            <-
        
        out:
            -> None

        """
    
        #init of tiling for all samples
        tile_height, tile_width = self.tile_dimension, self.tile_dimension
        
        temp_dict = {'coordinates': [], 'tissue_percentages': []}
        total_tiles_w_tissue = 0

        #acquire original tissue mask from chosen marker index
        mother_ymin = registered_masks[self.marker_index_maskforclipping][2]
        mother_xmin = registered_masks[self.marker_index_maskforclipping][0]
        mother_ymax = registered_masks[self.marker_index_maskforclipping][3]
        mother_xmax = registered_masks[self.marker_index_maskforclipping][1]
        mother_thumbnail_mask = tissue_masks[self.marker_index_maskforclipping][4]

        #create eroded tissue mask
        erosion_kernel = np.ones((3,3), np.uint8)
        num_iterations = int(np.floor((self.erosion_buffer_microns * 1.25 / self.output_resolution) / self.scale_factor))
        eroded_thumbnail_mask_forclipping = (cv2.erode(mother_thumbnail_mask.astype(np.uint8), erosion_kernel, iterations = num_iterations)).astype(bool)

        print(f'mother_ymin: {mother_ymin}')
        print(f'mother_xmin: {mother_xmin}')
        print(f'mother_ymax: {mother_ymax}')
        print(f'mother_xmax: {mother_xmax}')

        #accrue info per tile above tissue % thresh
        numtiles_height = int(np.ceil((mother_ymax - mother_ymin) / tile_height))
        numtiles_width = int(np.ceil((mother_xmax - mother_xmin) / tile_width))
        for tile_yindex in range(numtiles_height):
            for tile_xindex in range(numtiles_width):
                current_ymin = int(mother_ymin + tile_yindex * tile_height)
                current_ymax = current_ymin + tile_height
                current_xmin = int(mother_xmin + tile_xindex * tile_width)
                current_xmax = current_xmin + tile_width
                current_ymin_thumbnail = current_ymin*(1.25 / self.output_resolution)
                current_ymax_thumbnail = current_ymax*(1.25 / self.output_resolution)
                current_xmin_thumbnail = current_xmin*(1.25 / self.output_resolution)
                current_xmax_thumbnail = current_xmax*(1.25 / self.output_resolution)
                
                eroded_thumbnail_mask_tile = eroded_thumbnail_mask_forclipping[int(np.floor(current_ymin_thumbnail)): int(np.ceil(current_ymax_thumbnail)), 
                                                                               int(np.floor(current_xmin_thumbnail)): int(np.ceil(current_xmax_thumbnail))]
                
                amount_of_tissue_per_tile = np.sum(eroded_thumbnail_mask_tile)/(tile_height*1.25 / self.output_resolution * tile_width*1.25 / self.output_resolution) * 100
                if amount_of_tissue_per_tile > 100:
                    amount_of_tissue_per_tile = 100 #correct rounding error; percentages cannot be above 100

                current_array_coordinates = [current_ymin, current_xmin]
                if amount_of_tissue_per_tile > self.tile_analysis_thresh:
                    temp_dict['coordinates'].append(current_array_coordinates)
                    temp_dict['tissue_percentages'].append(amount_of_tissue_per_tile)
                    total_tiles_w_tissue +=1

        #acquire image and mask
        a = large_image.getTileSource(self.raw_images_path[self.marker_index_maskforclipping])
        data = a.getMetadata()
        OG_width = data['sizeX']
        OG_height = data['sizeY']
        b = a.getRegionAtAnotherScale(sourceRegion=dict(left=0, top=0, width=OG_width, height=OG_height, units='base_pixels'), targetScale=dict(magnification=1.25), format=large_image.tilesource.TILE_FORMAT_NUMPY)
        c = np.asarray(b[0])
        open_cv_image = c[:, :, :].copy()

        #plot original tissue mask
        pos_ys, pos_xs = np.where(mother_thumbnail_mask==True)
        for pos_pixel_index in range(len(pos_ys)):
            cv2.circle(open_cv_image, (pos_xs[pos_pixel_index], pos_ys[pos_pixel_index]), radius=0, color=(0, 255, 255,200), thickness = -1)

        #plot eroded tissue mask
        pos_ys, pos_xs = np.where(eroded_thumbnail_mask_forclipping==True)
        for pos_pixel_index in range(len(pos_ys)):
            cv2.circle(open_cv_image, (pos_xs[pos_pixel_index], pos_ys[pos_pixel_index]), radius = 0, color = (255, 0, 0,200), thickness = -1)
        
        alpha = 0.3
        image_updated = cv2.addWeighted(open_cv_image, alpha, c, 1 - alpha, 0)

        #plot ROIs
        print(f'Tile map original: {temp_dict["coordinates"]}')
        for count, item in enumerate(temp_dict['coordinates']): #draw ROIs
            starting_coor = (int(item[1]*(1.25 / self.output_resolution)), int(item[0] * (1.25 / self.output_resolution)))

            ending_coor = (int((item[1] * 1.25 / self.output_resolution) + (tile_width * 1.25 / self.output_resolution)), 
                           int((item[0] * 1.25 / self.output_resolution) + (tile_height * 1.25 / self.output_resolution)))
            
            cv2.rectangle(image_updated, (starting_coor), (ending_coor), (150,150,150,230), 2)
            cv2.putText(image_updated, str(count), starting_coor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 200), 2)

        #visualize both
        fig = plt.figure(figsize=(20, 20))
        graphs = fig.subplots(1,1)
        graphs.imshow(image_updated.astype(np.uint8))
        graphs.axis('off')
        graphs.set_title(str(total_tiles_w_tissue) + ' total tiles for ' + str(self.sample_name))
        
        #export as ome.tiff to compress
        save_path = os.path.join(self.marco_output_path, f'{self.sample_name}_tilemap.ome.tif')
        with tiff.TiffWriter(save_path) as tif:
            tif.write(image_updated.astype(np.uint8))

        #translate tile coords back to first marker index for analysus_per_tile fxn
        print(f'Registered mask for tilemap (shift): {registered_masks[self.marker_index_maskforclipping][4]}')
        for tile_index, nth_tile_coords in enumerate(temp_dict['coordinates']):
            first_marker_ycoord = nth_tile_coords[0] - registered_masks[self.marker_index_maskforclipping][4][0]
            first_marker_xcoord = nth_tile_coords[1] - registered_masks[self.marker_index_maskforclipping][4][1]

            print(f'Ycoord: {first_marker_ycoord}')
            print(f'Xcoord: {first_marker_xcoord}')
            
            # TODO: need to check why for same cases it is generating -1 y coords
            if first_marker_ycoord < 0:
                first_marker_ycoord = 0
            
            temp_dict['coordinates'][tile_index] = [first_marker_ycoord, first_marker_xcoord]

        save_path = os.path.join(self.marco_output_path, f'{self.sample_name}_tilemap.json')
        with open(save_path, 'w+') as fout:
            json.dump(temp_dict, fout)

        return temp_dict


    
    def get_tiling_map(self, registered_masks: List, tissue_masks: List):
        """
        
        in:
            <-
        
        out:
            -> None

        """
    
        #init of tiling for all samples
        tile_height, tile_width = self.tile_dimension, self.tile_dimension
        
        temp_dict = {'coordinates': [], 'tissue_percentages': []}
        total_tiles_w_tissue = 0

        #acquire original tissue mask from chosen marker index
        mother_ymin = registered_masks[self.marker_index_maskforclipping][2]
        mother_xmin = registered_masks[self.marker_index_maskforclipping][0]
        mother_ymax = registered_masks[self.marker_index_maskforclipping][3]
        mother_xmax = registered_masks[self.marker_index_maskforclipping][1]
        mother_thumbnail_mask = tissue_masks[self.marker_index_maskforclipping][4]

        #create eroded tissue mask
        erosion_kernel = np.ones((3,3), np.uint8)
        num_iterations = int(np.floor((self.erosion_buffer_microns * 1.25 / self.output_resolution) / self.scale_factor))
        eroded_thumbnail_mask_forclipping = (cv2.erode(mother_thumbnail_mask.astype(np.uint8), erosion_kernel, iterations = num_iterations)).astype(bool)


        #acquire image and mask
        a = large_image.getTileSource(self.raw_images_path[self.marker_index_maskforclipping])
        data = a.getMetadata()
        OG_width = data['sizeX']
        OG_height = data['sizeY']
        b = a.getRegionAtAnotherScale(sourceRegion=dict(left=0, top=0, width=OG_width, height=OG_height, units='base_pixels'), targetScale=dict(magnification=1.25), format=large_image.tilesource.TILE_FORMAT_NUMPY)
        c = np.asarray(b[0])
        open_cv_image = c[:, :, :].copy()

        #plot original tissue mask
        pos_ys, pos_xs = np.where(mother_thumbnail_mask==True)
        for pos_pixel_index in range(len(pos_ys)):
            cv2.circle(open_cv_image, (pos_xs[pos_pixel_index], pos_ys[pos_pixel_index]), radius=0, color=(0, 255, 255,200), thickness = -1)

        # get the most mask cartesian points
        x_min_ref = min(pos_xs) * (self.output_resolution / 1.25)
        y_min_ref = min(pos_ys) * (self.output_resolution / 1.25)
        x_max_ref = max(pos_xs) * (self.output_resolution / 1.25)
        y_max_ref = max(pos_ys) * (self.output_resolution / 1.25)

        #plot eroded tissue mask
        pos_ys, pos_xs = np.where(eroded_thumbnail_mask_forclipping==True)
        for pos_pixel_index in range(len(pos_ys)):
            cv2.circle(open_cv_image, (pos_xs[pos_pixel_index], pos_ys[pos_pixel_index]), radius = 0, color = (255, 0, 0,200), thickness = -1)

        alpha = 0.3
        image_updated = cv2.addWeighted(open_cv_image, alpha, c, 1 - alpha, 0)

        x_axis = x_max_ref - x_min_ref
        y_axis = y_max_ref - y_min_ref

        # TODO: add block to deal with images smaller than 1000 x 1000px

        if (x_axis % tile_width) != 0:
            remainder = x_axis % tile_width 
            x_axis += remainder
            x_max_ref += remainder 

        if (y_axis % tile_height) != 0:
            remainder = y_axis % tile_height
            y_axis += remainder
            y_max_ref += remainder 

        #accrue info per tile above tissue % thresh
        numtiles_height = int(np.ceil(y_axis / tile_height))
        numtiles_width = int(np.ceil(x_axis / tile_width))

        for tile_yindex in range(numtiles_height):
            for tile_xindex in range(numtiles_width):
                current_ymin = int(y_min_ref + tile_yindex * tile_height)
                current_ymax = current_ymin + tile_height
                current_xmin = int(x_min_ref + tile_xindex * tile_width)
                current_xmax = current_xmin + tile_width
                current_ymin_thumbnail = current_ymin*(1.25 / self.output_resolution)
                current_ymax_thumbnail = current_ymax*(1.25 / self.output_resolution)
                current_xmin_thumbnail = current_xmin*(1.25 / self.output_resolution)
                current_xmax_thumbnail = current_xmax*(1.25 / self.output_resolution)
                
                eroded_thumbnail_mask_tile = eroded_thumbnail_mask_forclipping[int(np.floor(current_ymin_thumbnail)): int(np.ceil(current_ymax_thumbnail)), 
                                                                               int(np.floor(current_xmin_thumbnail)): int(np.ceil(current_xmax_thumbnail))]
                
                amount_of_tissue_per_tile = np.sum(eroded_thumbnail_mask_tile)/(tile_height*1.25 / self.output_resolution * tile_width*1.25 / self.output_resolution) * 100
                if amount_of_tissue_per_tile > 100:
                    amount_of_tissue_per_tile = 100 #correct rounding error; percentages cannot be above 100

                current_array_coordinates = [current_ymin, current_xmin]
                if amount_of_tissue_per_tile > self.tile_analysis_thresh:
                    temp_dict['coordinates'].append(current_array_coordinates)
                    temp_dict['tissue_percentages'].append(amount_of_tissue_per_tile)
                    total_tiles_w_tissue +=1
        
        #plot ROIs
        for count, item in enumerate(temp_dict['coordinates']): #draw ROIs
            starting_coor = (int(item[1]*(1.25 / self.output_resolution)), int(item[0] * (1.25 / self.output_resolution)))

            ending_coor = (int((item[1] * 1.25 / self.output_resolution) + (tile_width * 1.25 / self.output_resolution)), 
                           int((item[0] * 1.25 / self.output_resolution) + (tile_height * 1.25 / self.output_resolution)))
            
            cv2.rectangle(image_updated, (starting_coor), (ending_coor), (150,150,150,230), 2)
            cv2.putText(image_updated, str(count), starting_coor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 200), 2)


        #visualize both
        fig = plt.figure(figsize=(20, 20))
        graphs = fig.subplots(1,1)
        graphs.imshow(image_updated.astype(np.uint8))
        graphs.axis('off')
        graphs.set_title(str(total_tiles_w_tissue) + ' total tiles for ' + str(self.sample_name))
        
        #export as ome.tiff to compress
        save_path = os.path.join(self.marco_output_path, f'{self.sample_name}_tilemap.ome.tif')
        with tiff.TiffWriter(save_path) as tif:
            tif.write(image_updated.astype(np.uint8))

        #translate tile coords back to first marker index 
        for tile_index, nth_tile_coords in enumerate(temp_dict['coordinates']):
            first_marker_ycoord = nth_tile_coords[0] - registered_masks[self.marker_index_maskforclipping][4][0]
            first_marker_xcoord = nth_tile_coords[1] - registered_masks[self.marker_index_maskforclipping][4][1]

            
            # TODO: need to check why for same cases it is generating -1 y coords
            if first_marker_ycoord < 0:
                first_marker_ycoord = 0
            
            temp_dict['coordinates'][tile_index] = [first_marker_ycoord, first_marker_xcoord]

        save_path = os.path.join(self.marco_output_path, f'{self.sample_name}_tilemap.json')
        with open(save_path, 'w+') as fout:
            json.dump(temp_dict, fout)

        return temp_dict, image_updated
