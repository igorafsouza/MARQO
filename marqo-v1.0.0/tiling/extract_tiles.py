import yaml
import cv2
import numpy as np

from utils.image import ImageHandler

class TileExtraction:
    """
    
    """

    def __init__(self):
        self.fixed_roi = None
        self.moving_roi = None
        self.registered_rois = []
        self.linear_shifts = []
        self.visualization_figs = {'registration':[],'segmentation':[],'quantification':[]}


    def set_parameters(self, config_file):
        """
        Read and fetch the paramaters and names needed to execute the masking.

        Paramters:
            - high_sens:
            - n_thresholding_steps:
            - sigma:
            - min_size:
            - crop_perc:

        """
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)


        self.markers = config_data['sample']['markers']
        self.marco_output_path = config_data['directories']['marco_output']

        self.sample_name = config_data['sample']['name']
        self.technology = config_data['sample']['technology']

        self.output_resolution = config_data['parameters']['image']['output_resolution']

        self.raw_images_path = config_data['sample']['raw_images_full_path']
        self.scale_factor_to_output = int(self.output_resolution / 1.25) 

        self.tile_height = config_data['parameters']['image']['tile_dimension']
        self.tile_width = config_data['parameters']['image']['tile_dimension']
        self.output_resolution = config_data['parameters']['image']['output_resolution']
        self.scale_factor = config_data['parameters']['image']['scale_factor']

        self.registration_buffer = config_data['parameters']['registration']['registration_buffer']
        self.segmentation_buffer = config_data['parameters']['segmentation']['segmentation_buffer']

        self.rgb_dir = config_data['directories']['rgb']

        if self.technology == 'if':
            self.af_channels = config_data['sample']['af_channels']
            self.dna_marker_channel = config_data['sample']['dna_marker_channel']


    def get_fixed_tile(self, image_path, mother_coordinates, channel_index=0):
        Height = int(self.tile_height + 2 * self.registration_buffer)
        Width = int(self.tile_width + 2 * self.registration_buffer)

        y_min = int(mother_coordinates[0] - self.registration_buffer)
        x_min = int(mother_coordinates[1] - self.registration_buffer)

        self.fixed_roi = ImageHandler.acquire_image(image_path, y_min, x_min, Height, Width, self.output_resolution, channel_index = channel_index)


    def extract_rois_all_channels(self, mother_coordinates, roi_index):
        difference_buffer = int(self.registration_buffer - self.segmentation_buffer)

        # There is only one IF image
        image_path = self.raw_images_path[0]
        
        fixed_tiles = []
        for channel_index, channel in enumerate(self.markers):
            self.get_fixed_tile(image_path, mother_coordinates, channel_index=channel_index)
            fixed_tiles.append(self.fixed_roi)

        for index, roi in enumerate(fixed_tiles):
            marker_name = self.markers[index]

            if self.af_channels:
                if index != self.dna_marker_channel and index not in self.af_channels:
                    roi = ImageHandler.subtract_af(roi, fixed_tiles, self.af_channels)
                    
            RGB_image_array = roi[difference_buffer: -difference_buffer, difference_buffer: -difference_buffer]
            linear_shift = [0,0]
            
            fixed_ROI_visual = roi[self.registration_buffer: -self.registration_buffer, self.registration_buffer: -self.registration_buffer]
            if fixed_ROI_visual.dtype == np.uint16:
                fixed_ROI_visual = ImageHandler.sharpening(fixed_ROI_visual)
                fixed_ROI_visual = cv2.normalize(fixed_ROI_visual, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

            elif fixed_ROI_visual.dtype == np.uint8:
                fixed_ROI_visual = ImageHandler.sharpening(fixed_ROI_visual)
                fixed_ROI_visual = cv2.normalize(fixed_ROI_visual, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
            metadata = {'PhysicalSizeX': self.scale_factor, 'PhysicalSizeY': self.scale_factor}
            image_name = f'{self.sample_name}_ROI#{roi_index}_{marker_name}.ome.tif'
            ImageHandler.save_tiff(fixed_ROI_visual, image_name, self.rgb_dir, resolution, metadata)
            
            registered_roi = RGB_image_array
            self.registered_rois.append(registered_roi)
            self.linear_shifts.append(linear_shift)

        
        return self.registered_rois, self.linear_shifts, self.visualization_figs


    def roi_extraction(self, mother_coordinates, roi_index: int):
        """
        Input (implicit):
            -> fixed tile
            -> moving tile

        Output:
            <- list with rois
        """

        difference_buffer = int(self.registration_buffer - self.segmentation_buffer)

        for image_index, image_path in enumerate(self.raw_images_path):
            marker_name = self.markers[image_index]

            # get and save first tile
            image_path = self.raw_images_path[image_index]
            self.get_fixed_tile(image_path, mother_coordinates)
            
            RGB_image_array = self.fixed_roi #duplicate to be used in the else
            RGB_image_array = self.fixed_roi[difference_buffer: -difference_buffer, difference_buffer: -difference_buffer, :]
            linear_shift = [0,0]
            
            fixed_ROI_visual = self.fixed_roi[self.registration_buffer: -self.registration_buffer, self.registration_buffer: -self.registration_buffer, :]
            
            resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
            metadata = {'PhysicalSizeX': self.scale_factor, 'PhysicalSizeY': self.scale_factor}
            image_name = f'{self.sample_name}_ROI#{roi_index}_{marker_name}.ome.tif'
            ImageHandler.save_tiff(fixed_ROI_visual, image_name, self.rgb_dir, resolution, metadata)


            registered_roi = RGB_image_array
            self.registered_rois.append(registered_roi)
            self.linear_shifts.append(linear_shift)

        
        return self.registered_rois, self.linear_shifts, self.visualization_figs
