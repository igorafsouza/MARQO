
# whole namespaces
import os
import yaml
import cv2
import large_image
import numpy as np
from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot as plt

# specific functions form namespaces
from typing import List

# pipeline classes and functions
from utils.image import ImageHandler


class RigidRegistration:
    """
    
    """

    def __init__(self):
        self.raw_images_path = ''


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

        sample = config_data['sample']['name']

        self.marker_name_list = config_data['sample']['markers']
        self.marco_output_path = config_data['directories']['marco_output']

        self.sample_name = sample

        output_resolution = config_data['parameters']['image']['output_resolution']

        self.raw_images_path = config_data['sample']['raw_images_full_path']
        self.scale_factor_to_output = int( output_resolution / 1.25) 
        
        self.iterations = config_data['parameters']['registration']['iterations']
        self.resolutions = config_data['parameters']['registration']['resolutions']
        self.spatial_samples = config_data['parameters']['registration']['spatial_samples']
        self.technology = config_data['sample']['technology']
        
        if self.technology == 'if':
            self.dna_marker = config_data['sample']['dna_marker_channel']


    def get_transformation_map(self, image_index: int, tissue_masks: List):
        """
        
        in:
            <-

        out:
            -> transformation_map
        """

        if self.technology == 'if':
            current_marker = self.marker_name_list[self.dna_marker]

        else:
            current_marker = self.marker_name_list[image_index]

        if image_index == 0: #if first stain
            I1_x_min, I1_x_max, I1_y_min, I1_y_max, _, _ = tissue_masks[image_index]

            window_height = I1_y_max - I1_y_min
            window_width = I1_x_max - I1_x_min

            #linear shift is zero for base registered image
            linear_shift = [0, 0]

            first_stain_name = self.raw_images_path[0]

            img_first = ImageHandler.acquire_image(first_stain_name, I1_y_min, I1_x_min, window_height, window_width, 1.25)

            #apply scale factor to store as metadata 
            I1_y_max = int(round(I1_y_max * self.scale_factor_to_output))
            I1_y_min = int(round(I1_y_min * self.scale_factor_to_output))
            I1_x_max = int(round(I1_x_max * self.scale_factor_to_output))
            I1_x_min = int(round(I1_x_min * self.scale_factor_to_output))

            #save to registered_masks array
            transformation_map = [I1_x_min, I1_x_max, I1_y_min, I1_y_max, linear_shift]

            im_0 = Image.fromarray(img_first)
            im_0_path = os.path.join(self.marco_output_path, f'{self.sample_name}_{current_marker}_thumbnail.png')
            metadata_im_0 = {'I1_x_min': I1_x_min, 'I1_x_max': I1_x_max, 'I1_y_min': I1_y_min, 'I1_y_max': I1_y_max, 'linear_shift': [0, 0]}
            ImageHandler.save_image(im_0, im_0_path, metadata=metadata_im_0)

        else: #other stains

            #find first stain
            first_stain_name = self.raw_images_path[0]

            #grab 1.25x of first-stain image
            a = large_image.getTileSource(first_stain_name)
            data = a.getMetadata()
            OG_width = data['sizeX']
            OG_height = data['sizeY']
            a = a.getRegionAtAnotherScale(sourceRegion=dict(left = 0, top = 0, width=OG_width, height=OG_height, units='base_pixels'), 
                                          targetScale=dict(magnification = 1.25), format=large_image.tilesource.TILE_FORMAT_NUMPY)

            #grab 1.25x of current-stain image
            b = large_image.getTileSource(self.raw_images_path[image_index])
            data = b.getMetadata()
            OG_width = data['sizeX']
            OG_height = data['sizeY']
            b = b.getRegionAtAnotherScale(sourceRegion=dict(left = 0, top = 0, width = OG_width, height = OG_height, units = 'base_pixels'), 
                                          targetScale=dict(magnification = 1.25), format=large_image.tilesource.TILE_FORMAT_NUMPY)
            if self.technology == 'cyif':
                a = np.asarray(a[0])
                img_a = a[:, :, 0].astype(np.uint8)

                b = np.asarray(b[0])
                img_current = b[:, :, 0].astype(np.uint8)

            else:
                a = np.asarray(a[0])
                img_a = a[:, :, 2].astype(np.uint8)

                b = np.asarray(b[0])
                img_current = b[:, :, 2].astype(np.uint8)

            #tissue_mask for both images
            I1_x_min, I1_x_max, I1_y_min, I1_y_max, _, _ = tissue_masks[0]
            I2_x_min, I2_x_max, I2_y_min, I2_y_max, _, _ = tissue_masks[image_index]

            #align images
            window_height = I1_y_max - I1_y_min
            window_width = I1_x_max - I1_x_min
            cropped_image_a = img_a[I1_y_min:I1_y_max,I1_x_min:I1_x_max].astype(np.uint8)
            cropped_image_b = img_current[I2_y_min:I2_y_max,I2_x_min:I2_x_max].astype(np.uint8)

            #perform the registration
            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.LogToConsoleOff()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(cropped_image_a))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(cropped_image_b))
            parameterMapVector = sitk.GetDefaultParameterMap('translation')
            parameterMapVector['MaximumNumberOfIterations'] = [str(self.iterations)]
            parameterMapVector['NumberOfResolutions'] = [str(self.resolutions)]
            parameterMapVector['NumberOfSpatialSamples'] = [str(self.spatial_samples)]
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            reg_map = elastixImageFilter.GetTransformParameterMap()[0]
            [x_shift, y_shift] = [-float(reg_map['TransformParameters'][0]), -float(reg_map['TransformParameters'][1])]
            x_shift_overall = (I2_x_min - I1_x_min) - x_shift
            y_shift_overall = (I2_y_min - I1_y_min) - y_shift

            #redefine image boundaries to align
            I2_x_min = I2_x_min - x_shift
            I2_x_max = I2_x_min + window_width
            I2_y_min = I2_y_min - y_shift
            I2_y_max = I2_y_min + window_height
            img_first = ImageHandler.acquire_image(first_stain_name, I1_y_min, I1_x_min, window_height, window_width, 1.25)
            img_current = ImageHandler.acquire_image(self.raw_images_path[image_index], I2_y_min, I2_x_min, window_height, window_width, 1.25)
            im = Image.fromarray(img_current)

            if self.technology == 'cyif':
                img_first = np.invert(img_first)
                img_first = np.dstack([img_first, img_first, img_first]).astype(np.uint8)
                mask = img_first[:, :, 0] < 250
                blue_color = np.zeros_like(img_first, dtype=np.uint8)
                blue_color[mask] = [0, 0, 255]  # RGB format
                
                img_first = np.where(mask[..., np.newaxis], blue_color, img_first)

                img_current = np.invert(img_current)
                img_current = np.dstack([img_current, img_current, img_current]).astype(np.uint8)
                mask = img_current[:, :, 0] < 250

                green_color = np.zeros_like(img_current, dtype=np.uint8)
                green_color[mask] = [0, 255, 0]  # RGB format
                
                img_current = np.where(mask[..., np.newaxis], green_color, img_current)


            fig = plt.figure(figsize=(15, 5), dpi=100)
            graphs = fig.subplots(1,3)
            graphs[0].imshow(img_first)
            graphs[0].set_title('Image index #0 \n Sample name: ' + str(self.sample_name) + '\n Marker name: ' + str(self.marker_name_list[0]))
            graphs[0].axis('off')
            graphs[1].imshow(img_current)
            graphs[1].set_title('Image index #' + str(image_index) + '\n Sample name: ' + str(self.sample_name) + '\n Marker name: ' + current_marker)
            graphs[1].axis('off')
            graphs[2].imshow(img_first, cmap='jet', alpha=0.5)
            graphs[2].imshow(img_current, cmap='jet', alpha=0.5)
            graphs[2].axis('off')
            graphs[2].set_title('Overlaid images')

            #apply scale factor
            I1_y_max = I1_y_max * self.scale_factor_to_output
            I1_y_min = I1_y_min * self.scale_factor_to_output
            I1_x_max = I1_x_max * self.scale_factor_to_output
            I1_x_min = I1_x_min * self.scale_factor_to_output

            I2_y_max = I2_y_max * self.scale_factor_to_output
            I2_y_min = I2_y_min * self.scale_factor_to_output
            I2_x_max = I2_x_max * self.scale_factor_to_output
            I2_x_min = I2_x_min * self.scale_factor_to_output

            X1 = round(x_shift_overall * self.scale_factor_to_output)
            Y1 = round(y_shift_overall * self.scale_factor_to_output)

            linear_shift = [Y1, X1]
            
            im_path = os.path.join(self.marco_output_path, f'{self.sample_name}_{current_marker}_thumbnail.png')
            metadata = {'I1_x_min': I2_x_min, 'I1_x_max': I2_x_max, 'I1_y_min': I2_y_min, 'I1_y_max': I2_y_max, 'linear_shift': linear_shift}
            ImageHandler.save_image(im, im_path, metadata = metadata)

            #save to registered_masks array
            transformation_map = [I2_x_min, I2_x_max, I2_y_min, I2_y_max, linear_shift]

        return transformation_map
