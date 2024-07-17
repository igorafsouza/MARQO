
import yaml
import math 
import numpy as np
import pandas as pd
import traceback
import functools
import operator

from KDEpy import FFTKDE
import histomicstk as ts

from utils.image import ImageHandler
from deconvolution.color import ColorMatrix


class Signal:
    """
    Class defines methods and attributes to extract stain specif signals in single cell resolution
    """

    def __init__(self):
        pass


    def set_parameters(self, config_file):
        self.config_file = config_file
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        try:
            self.num_pixels_thresh_for_peak = config_data['parameters']['quantification']['num_pixels_thresh_for_peak']
            self.hema_thresh_deemedRBC = config_data['parameters']['quantification']['hema_thresh_deemedRBC']

            self.bw_parameter = config_data['parameters']['classification']['bw']

            self.segmentation_buffer = config_data['parameters']['segmentation']['segmentation_buffer']

            self.scale_factor = config_data['parameters']['image']['scale_factor']

            self.marker_name_list = config_data['sample']['markers']
            self.visual_per_tile = config_data['parameters']['image']['visual_per_tile']

            self.tiff_dir = config_data['directories']['tiff']
            self.sample_name = config_data['sample']['name']
            self.technology = config_data['sample']['technology']

            if self.technology == 'cyif':
                self.cycles_metadata = config_data['cycles_metadata']


        except:
            self.num_pixels_thresh_for_peak = config_data['parameters']['num_pixels_thresh_for_peak']
            self.hema_thresh_deemedRBC = config_data['parameters']['hema_thresh_deemedRBC']

            self.bw_parameter = config_data['parameters']['bw_parameter']

            self.segmentation_buffer = config_data['parameters']['segmentation_buffer']

            self.scale_factor = config_data['parameters']['SF']

            self.marker_name_list = config_data['sample']['markers']
            self.visual_per_tile = config_data['parameters']['visual_per_tile']

            self.tiff_dir = config_data['directories']['product_directories']['tiff']
            self.sample_name = config_data['sample']['sample_name']
            self.technology = 'micsss'


    def _write_header(self):
        header_row = ['Tile index','ROI-relative cell y_coord (pixels)', 'ROI-relative cell x_coord (pixels)', 'Nuc peri (microns)', 'Nuc area (microns)', 'Nuc circularity', 'Nuc major axis (microns)', 'Nuc minor axis (microns)']

        for marker_name in self.marker_name_list:
            header_row = header_row + [(marker_name + ' y_coor (pixels)'),
                                       (marker_name + ' x_coor (pixels)'),
                                       (marker_name + ' cyto peri (microns)'),
                                       (marker_name + ' cyto area (microns)'),
                                       (marker_name + ' cyto circularity'),
                                       (marker_name + ' cyto major axis (microns)'),
                                       (marker_name + ' cyto minor (microns)'),
                                       (marker_name + ' nuc hema min'),
                                       (marker_name + ' nuc hema max'),
                                       (marker_name + ' nuc hema SD'),
                                       (marker_name + ' nuc hema mean'),
                                       (marker_name + ' nuc hema IQR'),
                                       (marker_name + ' nuc hema p1'),
                                       (marker_name + ' nuc hema p2'),
                                       (marker_name + ' nuc hema p3'),
                                       (marker_name + ' nuc hema p4'),
                                       (marker_name + ' nuc hema p5'),
                                       (marker_name + ' nuc hema p6'),
                                       (marker_name + ' nuc hema p7'),
                                       (marker_name + ' nuc hema p8'),
                                       (marker_name + ' nuc hema p9'),
                                       (marker_name + ' nuc hema num_peaks'),
                                       (marker_name + ' cyto AEC min'),
                                       (marker_name + ' cyto AEC max'),
                                       (marker_name + ' cyto AEC SD'),
                                       (marker_name + ' cyto AEC mean'),
                                       (marker_name + ' cyto AEC IQR'),
                                       (marker_name + ' cyto AEC p1'),
                                       (marker_name + ' cyto AEC p2'),
                                       (marker_name + ' cyto AEC p3'),
                                       (marker_name + ' cyto AEC p4'),
                                       (marker_name + ' cyto AEC p5'),
                                       (marker_name + ' cyto AEC p6'),
                                       (marker_name + ' cyto AEC p7'),
                                       (marker_name + ' cyto AEC p8'),
                                       (marker_name + ' cyto AEC p9'),
                                       (marker_name + ' cyto AEC num_peaks'),
                                       (marker_name + ' nuc AEC min'),
                                       (marker_name + ' nuc AEC max'),
                                       (marker_name + ' nuc AEC SD'),
                                       (marker_name + ' nuc AEC mean'),
                                       (marker_name + ' nuc AEC IQR'),
                                       (marker_name + ' nuc AEC p1'),
                                       (marker_name + ' nuc AEC p2'),
                                       (marker_name + ' nuc AEC p3'),
                                       (marker_name + ' nuc AEC p4'),
                                       (marker_name + ' nuc AEC p5'),
                                       (marker_name + ' nuc AEC p6'),
                                       (marker_name + ' nuc AEC p7'),
                                       (marker_name + ' nuc AEC p8'),
                                       (marker_name + ' nuc AEC p9'),
                                       (marker_name + ' nuc AEC num_peaks'),
                                       (marker_name + ' mem AEC min'),
                                       (marker_name + ' mem AEC max'),
                                       (marker_name + ' mem AEC SD'),
                                       (marker_name + ' mem AEC mean'),
                                       (marker_name + ' mem AEC IQR'),
                                       (marker_name + ' mem AEC p1'),
                                       (marker_name + ' mem AEC p2'),
                                       (marker_name + ' mem AEC p3'),
                                       (marker_name + ' mem AEC p4'),
                                       (marker_name + ' mem AEC p5'),
                                       (marker_name + ' mem AEC p6'),
                                       (marker_name + ' mem AEC p7'),
                                       (marker_name + ' mem AEC p8'),
                                       (marker_name + ' mem AEC p9'),
                                      (marker_name + ' mem AEC num_peaks')]

        return header_row


    def _find_morph_features(self, y_all, x_all, pixel_to_micron_SF):
    
        if len(y_all) > 1:    
            #calculate complete edge coordinates
            y_shift = np.min(y_all) - 2 #add buffer for edge cells
            x_shift = np.min(x_all) - 2
            y_all = y_all - y_shift #decrease size of temp array for faster processing 
            x_all = x_all - x_shift #decrease size of temp array for faster processing
            y_max = np.max(y_all) + 2
            x_max = np.max(x_all) + 2
            temp_cell_array = np.zeros((y_max, x_max))
            temp_cell_array[y_all,x_all] = True
            edge_indices = set(list(np.where(temp_cell_array[y_all-1, x_all]==0)[0]) + list(np.where(temp_cell_array[y_all+1, x_all]==0)[0]) + list(np.where(temp_cell_array[y_all, x_all-1]==0)[0]) + list(np.where(temp_cell_array[y_all, x_all+1]==0)[0]))
            y_edge = y_all[list(edge_indices)]
            x_edge = x_all[list(edge_indices)]    

            #calculate each spec
            area = len(y_all)
            perimeter = len(y_edge)
            y_mean = np.mean(y_edge)
            x_mean = np.mean(x_edge)
            distances_to_center = np.zeros((len(y_edge),1))
            for edge_index in range(len(y_edge)):
                distances_to_center[edge_index] = np.sqrt((y_edge[edge_index] - y_mean)**2 + (x_edge[edge_index] - x_mean)**2)
            ma = (np.min(distances_to_center))*2 #minor axis assumed to be twice the smallest radius 
            MA = (np.max(distances_to_center))*2 #major axis assumed to be twice the largest radius 
        
        #catch statement for any 1-pixel artifacts picked up
        else:
            area = 1
            perimeter = 1
            ma = 1
            MA = 1
                
        #convert morph features from pixels to microns
        ma_microns = ma*pixel_to_micron_SF #1D
        MA_microns = MA*pixel_to_micron_SF #1D
        perimeter_microns = perimeter*pixel_to_micron_SF #1D
        area_microns = area*(pixel_to_micron_SF**2) #2D    
        circularity = (4.0*math.pi*area_microns)/(perimeter_microns**2.0)
        if circularity>1: #constrict range 
            circularity=1    
        
        return(perimeter_microns, area_microns, circularity, MA_microns, ma_microns)


    def _update_geojsons_with_valid_cells(self, allmarkers_geojson_list, valid_cells_list):
        updated_allmarkers_geojson_list = []
        valid_cells_indicies = (np.where(np.array(valid_cells_list)==True))
        
        for geojson in allmarkers_geojson_list:
            updated_json_list_for_marker = list(np.array(geojson)[valid_cells_indicies[0]])
            updated_allmarkers_geojson_list.append(updated_json_list_for_marker)

        return updated_allmarkers_geojson_list

    def quantify(self, roi_index, registered_rois, linear_shifts, mother_coordinates, registered_masks, tile_tissue_percentage, visualization_figs, **kwargs):
      
        final_centroids = kwargs['final_centroids']
        final_nuc_edgecoords = kwargs['final_nuc_edgecoords']
        final_nuc_allcoords = kwargs['final_nuc_allcoords']
        final_cyto_edgecoords = kwargs['final_cyto_edgecoords']
        final_cyto_allcoords = kwargs['final_cyto_allcoords']
        allmarkers_geojson_list = kwargs['allmarkers_geojson_list']

        visual_raw_AEC = []
        visual_allcells_AEC = []
        ROI_pixel_metrics = []
        final_data_fortile = []

        if final_centroids == []:
            #rawcellmetrics_fortile_df = pd.DataFrame()
            header_row = self._write_header()
            rawcellmetrics_fortile_df = pd.DataFrame([], columns = header_row)
            results = {
                'data_for_tile': (rawcellmetrics_fortile_df, visualization_figs, ROI_pixel_metrics),
                'geojson_data': allmarkers_geojson_list
            }
            
            return results

        #init for quantification
        height, width, null = registered_rois[0].shape  # grabbing the dimensions of the tiles
        num_cells_perstain = len(final_cyto_allcoords[0])
        num_stains = len(registered_rois)
        num_values_per_cell = 15 #min, max, SD, mean, IQR, p1, p2, p3, p4, p5, p6, p7, p8, p9, num_peaks
        type_to_evaluate = 3 #AEC channel will evaluate values for nucleus, cytoplasm, and membrane
        AEC_initial_values_array = np.zeros((num_cells_perstain, num_stains, num_values_per_cell, type_to_evaluate)) #init numpy array for AEC values instead of dict for faster processing
        hema_initial_values_array = np.zeros((num_cells_perstain, num_stains, num_values_per_cell)) #init numpy array for hema values instead of dict for faster processing
        multitiff_output_final = np.zeros((int(num_stains+2), height, width)).astype(np.uint16)

        color_matrix = ColorMatrix()
        color_matrix.set_parameters(self.config_file)

        for stain_index, roi in enumerate(registered_rois):
            color_deconv_matrix = color_matrix.roi_evaluation(roi)
            imDeconvolved = ts.preprocessing.color_deconvolution.color_deconvolution(roi, color_deconv_matrix)
            Hemo_image_array = imDeconvolved.Stains[:, :, 0] #extract hemotoxylin channel
            Hemo_image_array[Hemo_image_array>255] = 255
            Hemo_image_array = np.invert(Hemo_image_array)
            AEC_tile = imDeconvolved.Stains[:, :, 1] #extract intensity of staining based on AEC expression
            AEC_tile[AEC_tile>255] = 255
            AEC_image_array = np.invert(AEC_tile)
            visual_raw_AEC.append(AEC_image_array)
            AEC_mask_allcells = np.zeros((height,width,3)).astype(np.uint8)

            #determine area values for tile
            tissue_area_fortile = tile_tissue_percentage / 100 * height * width * self.scale_factor**2
            ROI_pixel_metrics.append([tile_tissue_percentage, tissue_area_fortile])

            #update multitiff final
            multitiff_output_final[stain_index, :, :] = AEC_image_array
            if stain_index == 0:
                res_image_array = imDeconvolved.Stains[:, :, 2] #extract residual channel
                res_image_array[res_image_array>255] = 255
                res_image_array = np.invert(res_image_array)
                multitiff_output_final[-2, :, :] = Hemo_image_array
                multitiff_output_final[-1, :, :] = res_image_array

            for cell_index, cell_cyto_allcoords in enumerate(final_cyto_allcoords[stain_index]):

                #acquire respective pixel values
                cyto_AEC_values = AEC_image_array[cell_cyto_allcoords[0], cell_cyto_allcoords[1]]
                nuc_AEC_values = AEC_image_array[final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]]
                mem_AEC_values = AEC_image_array[final_cyto_edgecoords[stain_index][cell_index][0], final_cyto_edgecoords[stain_index][cell_index][1]]
                nuc_hema_values = Hemo_image_array[final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]]

                # getting hema values for the other morphological structures
                mem_hema_values = Hemo_image_array[final_cyto_edgecoords[stain_index][cell_index][0], final_cyto_edgecoords[stain_index][cell_index][1]]
                cyto_hema_values = Hemo_image_array[cell_cyto_allcoords[0], cell_cyto_allcoords[1]]

                # gettin residual channel values
                cyto_res_values = res_image_array[cell_cyto_allcoords[0], cell_cyto_allcoords[1]]
                nuc_res_values = res_image_array[final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]]
                mem_res_values = res_image_array[final_cyto_edgecoords[stain_index][cell_index][0], final_cyto_edgecoords[stain_index][cell_index][1]]

                #for visualization
                AEC_mask_allcells[cell_cyto_allcoords[0],cell_cyto_allcoords[1]] = registered_rois[stain_index][cell_cyto_allcoords[0], cell_cyto_allcoords[1]]
                AEC_mask_allcells[final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]] = registered_rois[stain_index][final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]]

                # adding measurements for the cell properties
                allmarkers_geojson_list[stain_index][cell_index].calculate_measurements(nuc_hema_values, cyto_hema_values, mem_hema_values, 
                                                                                        nuc_AEC_values, cyto_AEC_values, mem_AEC_values, 
                                                                                        nuc_res_values, cyto_res_values, mem_res_values)

                #save pixel values into numpy init values arrays
                for type_index, pixels_to_evaluate in enumerate([cyto_AEC_values, nuc_AEC_values, mem_AEC_values, nuc_hema_values]):
                    if not(type_index == 3): #for indicies 0-2
                        AEC_initial_values_array[cell_index][stain_index][0][type_index] = np.min(pixels_to_evaluate)/255
                        AEC_initial_values_array[cell_index][stain_index][1][type_index] = np.max(pixels_to_evaluate)/255
                        AEC_initial_values_array[cell_index][stain_index][2][type_index] = np.std(pixels_to_evaluate)/255
                        AEC_initial_values_array[cell_index][stain_index][3][type_index] = np.mean(pixels_to_evaluate)/255
                        p1,p2,p25,p3,p4,p5,p6,p7,p75,p8,p9 = np.percentile(pixels_to_evaluate,[10,20,25,30,40,50,60,70,75,80,90])/255
                        AEC_initial_values_array[cell_index][stain_index][4][type_index] = p75-p25 #iqr
                        for percentile_index, percentile_value in enumerate([p1,p2,p3,p4,p5,p6,p7,p8,p9]):
                            AEC_initial_values_array[cell_index][stain_index][int(5+percentile_index)][type_index] = percentile_value #percentiles

                        #determine num_peaks per graph
                        x1, y1 = FFTKDE(bw = self.bw_parameter).fit(pixels_to_evaluate).evaluate()
                        d1 = np.gradient(y1)
                        a = np.diff(np.sign(d1))
                        infls_index_d1_negative = np.where(a<0)[0]
                        y_infls_d1_negative = [y1[i] for i in infls_index_d1_negative]
                        num_peaks_forpixels = np.sum(np.array(y_infls_d1_negative) > self.num_pixels_thresh_for_peak)
                        if num_peaks_forpixels==0: #zero peaks is unrealistic and means there is one broader and shorter peak
                            num_peaks_forpixels=1
                        AEC_initial_values_array[cell_index][stain_index][14][type_index] = num_peaks_forpixels   

                    else: #when type_index==3, aka hema values
                        hema_initial_values_array[cell_index][stain_index][0] = np.min(pixels_to_evaluate)/255
                        hema_initial_values_array[cell_index][stain_index][1] = np.max(pixels_to_evaluate)/255
                        hema_initial_values_array[cell_index][stain_index][2] = np.std(pixels_to_evaluate)/255
                        hema_initial_values_array[cell_index][stain_index][3] = np.mean(pixels_to_evaluate)/255
                        p1,p2,p25,p3,p4,p5,p6,p7,p75,p8,p9 = np.percentile(pixels_to_evaluate,[10,20,25,30,40,50,60,70,75,80,90])/255
                        hema_initial_values_array[cell_index][stain_index][4] = p75-p25 #iqr
                        hema_initial_values_array[cell_index][stain_index][5:14] = p1,p2,p3,p4,p5,p6,p7,p8,p9 #percentiles

                        #determine num_peaks per graph
                        x1, y1 = FFTKDE(bw = self.bw_parameter).fit(pixels_to_evaluate).evaluate()
                        d1 = np.gradient(y1)
                        a = np.diff(np.sign(d1))
                        infls_index_d1_negative = np.where(a<0)[0]
                        y_infls_d1_negative = [y1[i] for i in infls_index_d1_negative]
                        num_peaks_forpixels = np.sum(np.array(y_infls_d1_negative) > self.num_pixels_thresh_for_peak)
                        if num_peaks_forpixels==0: #zero peaks is unrealistic and means there is one broader and shorter peak
                            num_peaks_forpixels=1
                        hema_initial_values_array[cell_index][stain_index][14] = num_peaks_forpixels 

            visual_allcells_AEC.append(AEC_mask_allcells)

        #eliminate cells that are below median hema threshold -- we assume these are RBCs
        valid_cells_list = np.zeros((num_cells_perstain, 1), bool)
        for cell_index in range(num_cells_perstain):
            #2 relative ROI coords + 5 nucleus morph values + num_stains*(2 coords + 5 cell morph values + 15 hema values + 15*3 AEC values)
            temp_cell_row = np.zeros((int(2 + 5 + num_stains*(2+5+num_values_per_cell+type_to_evaluate*num_values_per_cell)),1))

            if (np.median(hema_initial_values_array[cell_index][:,9]) > self.hema_thresh_deemedRBC): #9th index is the median channel; take median of medians of nuclear hema values

                #tally which cells are valid
                valid_cells_list[cell_index] = True

                #2 relative ROI coords + 5 nucleus morph values + num_stains*(2 coords + 5 cell morph values + 15 hema values + 15*3 AEC values)
                #temp_cell_row = np.zeros((int(2 + 5 + num_stains*(2+5+num_values_per_cell+type_to_evaluate*num_values_per_cell)),1))

                #acquire relative ROI cell coords
                temp_cell_row[:2,0] = final_centroids[cell_index][0] - self.segmentation_buffer, final_centroids[cell_index][1] - self.segmentation_buffer

                #find morph values for nuc, which stays the same across all stains
                ny_all,nx_all = final_nuc_allcoords[cell_index]
                nuc_peri, nuc_area, nuc_circ, nuc_MA, nuc_ma = self._find_morph_features(ny_all, nx_all, self.scale_factor)  #determine morph features for nucleus in microns
                temp_cell_row[2:7,0] = (nuc_peri, nuc_area, nuc_circ, nuc_MA, nuc_ma)

                #iterate for unique values across each stain
                for stain_index in range(num_stains):

                    #determine coordinates of each cell respective to all stained images
                    current_ycoor = int(mother_coordinates[0] + registered_masks[stain_index][4][0] + final_centroids[cell_index][0] - self.segmentation_buffer - linear_shifts[stain_index][0])
                    current_ycoor = np.round(current_ycoor)

                    current_xcoor = int(mother_coordinates[1] + registered_masks[stain_index][4][1] + final_centroids[cell_index][1] - self.segmentation_buffer - linear_shifts[stain_index][1])
                    current_xcoor = np.round(current_xcoor)

                    temp_cell_row[int(7 + 67 * stain_index): int(9 + 67 * stain_index), 0] = current_ycoor,current_xcoor

                    #determine morphological features of the cell
                    cy_all, cx_all = final_cyto_allcoords[stain_index][cell_index]
                    cyto_peri, cyto_area, cyto_circ, cyto_MA, cyto_ma = self._find_morph_features(cy_all, cx_all, self.scale_factor)  #determine morph features for whole cell in microns
                    temp_cell_row[int(9+67*stain_index):int(14+67*stain_index), 0] = np.round(cyto_peri), np.round(cyto_area), np.round(cyto_circ), np.round(cyto_MA), np.round(cyto_ma)

                    #acquire hema values
                    temp_cell_row[int(14+67*stain_index):int(29+67*stain_index), 0] = hema_initial_values_array[cell_index][stain_index]

                    #acquire AEC values
                    temp_cell_row[int(29+67*stain_index):int(44+67*stain_index), 0] = AEC_initial_values_array[cell_index][stain_index][:, 0]
                    temp_cell_row[int(44+67*stain_index):int(59+67*stain_index), 0] = AEC_initial_values_array[cell_index][stain_index][:, 1]
                    temp_cell_row[int(59+67*stain_index):int(74+67*stain_index), 0] = AEC_initial_values_array[cell_index][stain_index][:, 2]

                #append all values to final_data_matrix
                temp_cell_row = np.round(temp_cell_row, 3)
                final_data_fortile.append(temp_cell_row)

        # update geojsons with valid cells
        allmarkers_geojson_list = self._update_geojsons_with_valid_cells(allmarkers_geojson_list, valid_cells_list)

        header_row = self._write_header()

        #convert list of numpy arrays to pandas dataframe
        num_rows_fortile = len(final_data_fortile)
        num_columns_fortile = len(temp_cell_row)
        rawcellmetrics_fortile_numpy = np.zeros((num_rows_fortile, num_columns_fortile))
        for current_row_index in range(num_rows_fortile):
            rawcellmetrics_fortile_numpy[current_row_index,:] = final_data_fortile[current_row_index].T
        
        tileindex_column = np.zeros((num_rows_fortile,1))
        tileindex_column[:] = roi_index
        rawcellmetrics_fortile_numpy = np.append(tileindex_column, rawcellmetrics_fortile_numpy, axis=1)
        rawcellmetrics_fortile_df = pd.DataFrame(rawcellmetrics_fortile_numpy, columns = header_row)

        #for visualization
        for visual_index in range(len(visual_raw_AEC)):
            visualization_figs['quantification'].append([visual_raw_AEC[visual_index], visual_allcells_AEC[visual_index]])
        
        multitiff_output_final = multitiff_output_final[:, self.segmentation_buffer: -self.segmentation_buffer, self.segmentation_buffer: -self.segmentation_buffer]

        if self.visual_per_tile:
            resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
            metadata = {'axes': 'CYX', 
                        'Channel': {'Name':(self.marker_name_list + ['Hematoxylin for ' + self.marker_name_list[0]] + ['3rd channel for ' + self.marker_name_list[0]])}, 
                        'PhysicalSizeX': self.scale_factor, 
                        'PhysicalSizeY': self.scale_factor}
            image_name = f'{self.sample_name}_multichanneloutput_ROI#{roi_index}.ome.tif'
            ImageHandler.save_tiff(multitiff_output_final, image_name, self.tiff_dir, resolution, metadata)

        results = {
            'data_for_tile': (rawcellmetrics_fortile_df, visualization_figs, ROI_pixel_metrics),
            'geojson_data': allmarkers_geojson_list
        }
        return results



    def quantify_if_channels(self, roi_index, registered_rois, linear_shifts, mother_coordinates, registered_masks, tile_tissue_percentage, visualization_figs, **kwargs):
      
        final_centroids = kwargs['final_centroids']
        final_nuc_edgecoords = kwargs['final_nuc_edgecoords']
        final_nuc_allcoords = kwargs['final_nuc_allcoords']
        final_cyto_edgecoords = kwargs['final_cyto_edgecoords']
        final_cyto_allcoords = kwargs['final_cyto_allcoords']
        allmarkers_geojson_list = kwargs['allmarkers_geojson_list']
        dapi_index = 0

        #init for quantification
        final_data_fortile = []
        dapi_image_array = registered_rois[0]
        #dapi_image_array[dapi_image_array>255] = 255
        height = registered_rois[0].shape[0]  # grabbing the dimensions of the tiles
        width = registered_rois[0].shape[1]  # grabbing the dimensions of the tiles
        num_cells_perstain = len(final_cyto_allcoords[0])
        num_stains = len(registered_rois)

        if hasattr(self, 'cycles_metadata'):
            self.marker_name_list = []
            marker_list_across_channels = [v['marker_name_list'] for k, v in self.cycles_metadata.items()]
            marker_list_across_channels = functools.reduce(operator.iconcat, marker_list_across_channels, [])  # flatten the list of lists
            cycles = [f'cycle_{i}' for i, v in self.cycles_metadata.items() for _ in range(len(v['marker_name_list']))]
            for marker, cycle in zip(marker_list_across_channels, cycles):
                marker_name = f'{cycle}_{marker}'
                self.marker_name_list.append(marker_name)


        num_values_per_cell = 15 #min, max, SD, mean, IQR, p1, p2, p3, p4, p5, p6, p7, p8, p9, num_peaks
        type_to_evaluate = 3 #AEC channel will evaluate values for nucleus, cytoplasm, and membrane
        chromogen_initial_values_array = np.zeros((num_cells_perstain, num_stains, num_values_per_cell, type_to_evaluate)) #init numpy array for AEC values instead of dict for faster processing
        dapi_initial_values_array = np.zeros((num_cells_perstain, num_stains, num_values_per_cell)) #init numpy array for hema values instead of dict for faster processing
        visual_raw_AEC = []
        visual_allcells_AEC = []
        ROI_pixel_metrics = []
        multitiff_output_final = np.zeros((int(num_stains), height, width))

        for stain_index, roi in enumerate(registered_rois):
            chromogen_tile = roi  #extract intensity of staining based on chromogen expression
            chromogen_image_array = chromogen_tile
            visual_raw_AEC.append(chromogen_image_array)
            chromogen_mask_allcells = np.zeros((height,width))

            if roi.dtype == np.uint16:
                max_pixel_value = 65535            

            elif roi.dtype == np.uint8:
                max_pixel_value = 255

            #determine area values for tile
            tissue_area_fortile = tile_tissue_percentage / 100 * height * width * self.scale_factor**2
            ROI_pixel_metrics.append([tile_tissue_percentage, tissue_area_fortile])

            #update multitiff final
            multitiff_output_final[stain_index, :, :] = chromogen_image_array[:, :]

            for cell_index, cell_cyto_allcoords in enumerate(final_cyto_allcoords[0]):

                #acquire respective pixel values
                cyto_chromogen_values = chromogen_image_array[cell_cyto_allcoords[0], cell_cyto_allcoords[1]]
                nuc_chromogen_values = chromogen_image_array[final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]]
                mem_chromogen_values = chromogen_image_array[final_cyto_edgecoords[dapi_index][cell_index][0], final_cyto_edgecoords[dapi_index][cell_index][1]]

                #get values for dapi channel
                nuc_dapi_values = dapi_image_array[final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]]
                mem_dapi_values = dapi_image_array[final_cyto_edgecoords[dapi_index][cell_index][0], final_cyto_edgecoords[dapi_index][cell_index][1]]
                cyto_dapi_values = dapi_image_array[cell_cyto_allcoords[0], cell_cyto_allcoords[1]]

                #for visualization
                chromogen_mask_allcells[cell_cyto_allcoords[0],cell_cyto_allcoords[1]] = registered_rois[stain_index][cell_cyto_allcoords[0], cell_cyto_allcoords[1]]
                chromogen_mask_allcells[final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]] = registered_rois[stain_index][final_nuc_allcoords[cell_index][0], final_nuc_allcoords[cell_index][1]]

                # adding measurements for the cell properties
                allmarkers_geojson_list[dapi_index][cell_index].calculate_measurements_if(nuc_chromogen_values, cyto_chromogen_values, mem_chromogen_values,
                                                                                           nuc_dapi_values, mem_dapi_values, cyto_dapi_values, max_pixel_value)

                #save pixel values into numpy init values arrays
                for type_index, pixels_to_evaluate in enumerate([cyto_chromogen_values, nuc_chromogen_values, mem_chromogen_values, nuc_dapi_values]):
                    if not(type_index == 3): #for indicies 0-2
                        chromogen_initial_values_array[cell_index][stain_index][0][type_index] = np.min(pixels_to_evaluate)
                        chromogen_initial_values_array[cell_index][stain_index][1][type_index] = np.max(pixels_to_evaluate)
                        chromogen_initial_values_array[cell_index][stain_index][2][type_index] = np.std(pixels_to_evaluate)
                        chromogen_initial_values_array[cell_index][stain_index][3][type_index] = np.mean(pixels_to_evaluate)
                        p1,p2,p25,p3,p4,p5,p6,p7,p75,p8,p9 = np.percentile(pixels_to_evaluate,[10,20,25,30,40,50,60,70,75,80,90])
                        chromogen_initial_values_array[cell_index][stain_index][4][type_index] = p75-p25 #iqr
                        for percentile_index, percentile_value in enumerate([p1,p2,p3,p4,p5,p6,p7,p8,p9]):
                            chromogen_initial_values_array[cell_index][stain_index][int(5+percentile_index)][type_index] = percentile_value #percentiles

                        #determine num_peaks per graph
                        x1, y1 = FFTKDE(bw = self.bw_parameter).fit(pixels_to_evaluate).evaluate()
                        d1 = np.gradient(y1)
                        a = np.diff(np.sign(d1))
                        infls_index_d1_negative = np.where(a<0)[0]
                        y_infls_d1_negative = [y1[i] for i in infls_index_d1_negative]
                        num_peaks_forpixels = np.sum(np.array(y_infls_d1_negative) > self.num_pixels_thresh_for_peak)
                        
                        if num_peaks_forpixels==0: #zero peaks is unrealistic and means there is one broader and shorter peak
                            num_peaks_forpixels=1
                        
                        chromogen_initial_values_array[cell_index][stain_index][14][type_index] = num_peaks_forpixels   

                    else: #when type_index==3, aka hema values
                        dapi_initial_values_array[cell_index][stain_index][0] = np.min(pixels_to_evaluate)
                        dapi_initial_values_array[cell_index][stain_index][1] = np.max(pixels_to_evaluate)
                        dapi_initial_values_array[cell_index][stain_index][2] = np.std(pixels_to_evaluate)
                        dapi_initial_values_array[cell_index][stain_index][3] = np.mean(pixels_to_evaluate)
                        p1,p2,p25,p3,p4,p5,p6,p7,p75,p8,p9 = np.percentile(pixels_to_evaluate,[10,20,25,30,40,50,60,70,75,80,90])
                        dapi_initial_values_array[cell_index][stain_index][4] = p75-p25 #iqr
                        dapi_initial_values_array[cell_index][stain_index][5:14] = p1,p2,p3,p4,p5,p6,p7,p8,p9 #percentiles

                        #determine num_peaks per graph
                        x1, y1 = FFTKDE(bw = self.bw_parameter).fit(pixels_to_evaluate).evaluate()
                        d1 = np.gradient(y1)
                        a = np.diff(np.sign(d1))
                        infls_index_d1_negative = np.where(a<0)[0]
                        y_infls_d1_negative = [y1[i] for i in infls_index_d1_negative]
                        num_peaks_forpixels = np.sum(np.array(y_infls_d1_negative) > self.num_pixels_thresh_for_peak)
                        if num_peaks_forpixels==0: #zero peaks is unrealistic and means there is one broader and shorter peak
                            num_peaks_forpixels=1

                        dapi_initial_values_array[cell_index][stain_index][14] = num_peaks_forpixels 

            visual_allcells_AEC.append(chromogen_mask_allcells)

        #eliminate cells that are below median hema threshold -- we assume these are RBCs
        valid_cells_list = np.zeros((num_cells_perstain, 1), bool)
        for cell_index in range(num_cells_perstain):

            #tally which cells are valid
            valid_cells_list[cell_index] = True

            #2 relative ROI coords + 5 nucleus morph values + num_stains*(2 coords + 5 cell morph values + 15 hema values + 15*3 AEC values)
            temp_cell_row = np.zeros((int(2 + 5 + num_stains*(2+5+num_values_per_cell+type_to_evaluate*num_values_per_cell)),1))

            #acquire relative ROI cell coords
            temp_cell_row[:2,0] = final_centroids[cell_index][0] - self.segmentation_buffer, final_centroids[cell_index][1] - self.segmentation_buffer

            #find morph values for nuc, which stays the same across all stains
            ny_all,nx_all = final_nuc_allcoords[cell_index]
            nuc_peri, nuc_area, nuc_circ, nuc_MA, nuc_ma = self._find_morph_features(ny_all, nx_all, self.scale_factor)  #determine morph features for nucleus in microns
            temp_cell_row[2:7,0] = (nuc_peri, nuc_area, nuc_circ, nuc_MA, nuc_ma)

            #iterate for unique values across each stain
            for stain_index in range(num_stains):

                #determine coordinates of each cell respective to all stained images
                current_ycoor = int(mother_coordinates[0] + registered_masks[dapi_index][4][0] + final_centroids[cell_index][0] - self.segmentation_buffer - linear_shifts[dapi_index][0])
                current_xcoor = int(mother_coordinates[1] + registered_masks[dapi_index][4][1] + final_centroids[cell_index][1] - self.segmentation_buffer - linear_shifts[dapi_index][1])
                temp_cell_row[int(7 + 67 * stain_index): int(9 + 67 * stain_index), 0] = current_ycoor,current_xcoor

                #determine morphological features of the cell
                cy_all,cx_all = final_cyto_allcoords[0][cell_index]
                cyto_peri, cyto_area, cyto_circ, cyto_MA, cyto_ma = self._find_morph_features(cy_all, cx_all, self.scale_factor)  #determine morph features for whole cell in microns
                temp_cell_row[int(9+67*stain_index):int(14+67*stain_index), 0] = cyto_peri, cyto_area, cyto_circ, cyto_MA, cyto_ma

                #acquire hema values
                temp_cell_row[int(14+67*stain_index):int(29+67*stain_index), 0] = dapi_initial_values_array[cell_index][stain_index]

                #acquire AEC values
                temp_cell_row[int(29+67*stain_index):int(44+67*stain_index), 0] = chromogen_initial_values_array[cell_index][stain_index][:, 0]
                temp_cell_row[int(44+67*stain_index):int(59+67*stain_index), 0] = chromogen_initial_values_array[cell_index][stain_index][:, 1]
                temp_cell_row[int(59+67*stain_index):int(74+67*stain_index), 0] = chromogen_initial_values_array[cell_index][stain_index][:, 2]

            #append all values to final_data_matrix
            temp_cell_row = np.round(temp_cell_row, 3)
            final_data_fortile.append(temp_cell_row)

        # update geojsons with valid cells
        allmarkers_geojson_list = self._update_geojsons_with_valid_cells(allmarkers_geojson_list, valid_cells_list)

        header_row = ['Tile index','ROI-relative cell y_coord (pixels)', 'ROI-relative cell x_coord (pixels)', 'Nuc peri (microns)', 'Nuc area (microns)', 'Nuc circularity', 'Nuc major axis (microns)', 'Nuc minor axis (microns)']
        for marker_name in self.marker_name_list:
            header_row = header_row + [(marker_name + ' y_coor (pixels)'),
                                       (marker_name + ' x_coor (pixels)'),
                                       (marker_name + ' cyto peri (microns)'),
                                       (marker_name + ' cyto area (microns)'),
                                       (marker_name + ' cyto circularity'),
                                       (marker_name + ' cyto major axis (microns)'),
                                       (marker_name + ' cyto minor (microns)'),
                                       (marker_name + ' nuc dapi min'),
                                       (marker_name + ' nuc dapi max'),
                                       (marker_name + ' nuc dapi SD'),
                                       (marker_name + ' nuc dapi mean'),
                                       (marker_name + ' nuc dapi IQR'),
                                       (marker_name + ' nuc dapi p1'),
                                       (marker_name + ' nuc dapi p2'),
                                       (marker_name + ' nuc dapi p3'),
                                       (marker_name + ' nuc dapi p4'),
                                       (marker_name + ' nuc dapi p5'),
                                       (marker_name + ' nuc dapi p6'),
                                       (marker_name + ' nuc dapi p7'),
                                       (marker_name + ' nuc dapi p8'),
                                       (marker_name + ' nuc dapi p9'),
                                       (marker_name + ' nuc dapi num_peaks'),
                                       (marker_name + ' cyto chromogen min'),
                                       (marker_name + ' cyto chromogen max'),
                                       (marker_name + ' cyto chromogen SD'),
                                       (marker_name + ' cyto chromogen mean'),
                                       (marker_name + ' cyto chromogen IQR'),
                                       (marker_name + ' cyto chromogen p1'),
                                       (marker_name + ' cyto chromogen p2'),
                                       (marker_name + ' cyto chromogen p3'),
                                       (marker_name + ' cyto chromogen p4'),
                                       (marker_name + ' cyto chromogen p5'),
                                       (marker_name + ' cyto chromogen p6'),
                                       (marker_name + ' cyto chromogen p7'),
                                       (marker_name + ' cyto chromogen p8'),
                                       (marker_name + ' cyto chromogen p9'),
                                       (marker_name + ' cyto chromogen num_peaks'),
                                       (marker_name + ' nuc chromogen min'),
                                       (marker_name + ' nuc chromogen max'),
                                       (marker_name + ' nuc chromogen SD'),
                                       (marker_name + ' nuc chromogen mean'),
                                       (marker_name + ' nuc chromogen IQR'),
                                       (marker_name + ' nuc chromogen p1'),
                                       (marker_name + ' nuc chromogen p2'),
                                       (marker_name + ' nuc chromogen p3'),
                                       (marker_name + ' nuc chromogen p4'),
                                       (marker_name + ' nuc chromogen p5'),
                                       (marker_name + ' nuc chromogen p6'),
                                       (marker_name + ' nuc chromogen p7'),
                                       (marker_name + ' nuc chromogen p8'),
                                       (marker_name + ' nuc chromogen p9'),
                                       (marker_name + ' nuc chromogen num_peaks'),
                                       (marker_name + ' mem chromogen min'),
                                       (marker_name + ' mem chromogen max'),
                                       (marker_name + ' mem chromogen SD'),
                                       (marker_name + ' mem chromogen mean'),
                                       (marker_name + ' mem chromogen IQR'),
                                       (marker_name + ' mem chromogen p1'),
                                       (marker_name + ' mem chromogen p2'),
                                       (marker_name + ' mem chromogen p3'),
                                       (marker_name + ' mem chromogen p4'),
                                       (marker_name + ' mem chromogen p5'),
                                       (marker_name + ' mem chromogen p6'),
                                       (marker_name + ' mem chromogen p7'),
                                       (marker_name + ' mem chromogen p8'),
                                       (marker_name + ' mem chromogen p9'),
                                       (marker_name + ' mem chromogen num_peaks')
                                       ]

        #convert list of numpy arrays to pandas dataframe
        num_rows_fortile = len(final_data_fortile)
        num_columns_fortile = len(temp_cell_row)
        rawcellmetrics_fortile_numpy = np.zeros((num_rows_fortile, num_columns_fortile))
        for current_row_index in range(num_rows_fortile):
            rawcellmetrics_fortile_numpy[current_row_index,:] = final_data_fortile[current_row_index].T
        
        tileindex_column = np.zeros((num_rows_fortile,1))
        tileindex_column[:] = roi_index
        rawcellmetrics_fortile_numpy = np.append(tileindex_column, rawcellmetrics_fortile_numpy, axis=1)
        rawcellmetrics_fortile_df = pd.DataFrame(rawcellmetrics_fortile_numpy, columns = header_row)

        #for visualization
        for visual_index in range(len(visual_raw_AEC)):
            visualization_figs['quantification'].append([visual_raw_AEC[visual_index], visual_allcells_AEC[visual_index]])
        
        multitiff_output_final = multitiff_output_final[:, self.segmentation_buffer: -self.segmentation_buffer, self.segmentation_buffer: -self.segmentation_buffer]

        if self.visual_per_tile:
            resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
            metadata = {'axes': 'CYX', 
                        'Channel': {'Name':(self.marker_name_list)}, 
                        'PhysicalSizeX': self.scale_factor, 
                        'PhysicalSizeY': self.scale_factor}
            image_name = f'{self.sample_name}_multichanneloutput_ROI#{roi_index}.ome.tif'
            ImageHandler.save_tiff(multitiff_output_final, image_name, self.tiff_dir, resolution, metadata)

        results = {
            'data_for_tile': (rawcellmetrics_fortile_df, visualization_figs, ROI_pixel_metrics),
            'geojson_data': allmarkers_geojson_list
        }
        return results

