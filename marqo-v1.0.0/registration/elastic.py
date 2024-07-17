import yaml
import cv2
import numpy as np
import histomicstk as ts
import SimpleITK as sitk
import traceback
from scipy.ndimage import binary_dilation

from utils.image import ImageHandler
from deconvolution.color import ColorMatrix


class ElasticRegistration:
    """
    Class to perform the elastic registration per tile
    """

    def __init__(self):
        """
        One fixed attribute, the anchor tile (from the first image), and one dynamic attribute, corresponding to the 
        subsequent tiles (one from each image). Both attributes are update on the fly.
        """
        self.fixed_roi = None
        self.moving_roi = None
        self.registered_rois = []
        self.linear_shifts = []
        self.visualization_figs = {'registration':[],'segmentation':[],'quantification':[]}

    def set_parameters(self, config_file):
        """
        Read and fetch the paramaters and names needed to execute the masking.

        Paramters:
            - mag_to_analyze_for_matrix:
            - marco_output_path:
            - sample_name:
            - marker_name_list:

        """
        self.config_file = config_file
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        try:
            self.tile_height = config_data['parameters']['image']['tile_dimension']
            self.tile_width = config_data['parameters']['image']['tile_dimension']
            self.output_resolution = config_data['parameters']['image']['output_resolution']
            self.scale_factor = config_data['parameters']['image']['scale_factor']

            self.registration_buffer = config_data['parameters']['registration']['registration_buffer']
            self.segmentation_buffer = config_data['parameters']['segmentation']['segmentation_buffer']

            self.sample_name = config_data['sample']['name']
            self.marker_name_list = config_data['sample']['markers']
            self.raw_images_path = config_data['sample']['raw_images_full_path']

            self.rgb_dir = config_data['directories']['rgb']
            self.technology = config_data['sample']['technology']

            if self.technology == 'cyif':
                self.cycles_metadata = config_data['cycles_metadata']

        except:
            self.tile_height = config_data['parameters']['tile_dimension']
            self.tile_width = config_data['parameters']['tile_dimension']
            self.output_resolution = config_data['parameters']['output_resolution']
            self.scale_factor = config_data['parameters']['SF']

            self.registration_buffer = config_data['parameters']['registration_buffer']
            self.segmentation_buffer = config_data['parameters']['segmentation_buffer']

            self.sample_name = config_data['sample']['sample_name']
            self.marker_name_list = config_data['sample']['markers']
            self.raw_images_path = config_data['sample']['slides_path']

            self.rgb_dir = config_data['directories']['product_directories']['rgb']


    def get_fixed_tile(self, image_path, mother_coordinates, channel_index=0):
        Height = int(self.tile_height + 2 * self.registration_buffer)
        Width = int(self.tile_width + 2 * self.registration_buffer)

        y_min = int(mother_coordinates[0] - self.registration_buffer)
        x_min = int(mother_coordinates[1] - self.registration_buffer)

        self.fixed_roi = ImageHandler.acquire_image(image_path, y_min, x_min, Height, Width, self.output_resolution, channel_index=channel_index)


    def get_moving_tile(self, image_path, mother_coordinates, registered_masks, image_index, channel_index=0):
        Height = int(self.tile_height + 2 * self.registration_buffer)
        Width = int(self.tile_width + 2 * self.registration_buffer)

        y_min = mother_coordinates[0] - registered_masks[0][2] + registered_masks[image_index][2] - self.registration_buffer
        x_min = mother_coordinates[1] - registered_masks[0][0] + registered_masks[image_index][0] - self.registration_buffer

        self.moving_roi = ImageHandler.acquire_image(image_path, y_min, x_min, Height, Width, self.output_resolution, channel_index=channel_index)


    def get_transformed_images(self, mother_coordinates, registered_masks, roi_index: int):
        """
        Input (implicit):
            -> fixed tile
            -> moving tile

        Output:
            <- list with registrated tiles
        """

        difference_buffer = int(self.registration_buffer - self.segmentation_buffer)

        color_matrix = ColorMatrix()
        color_matrix.set_parameters(self.config_file)

        for image_index, image_path in enumerate(self.raw_images_path):
            marker_name = self.marker_name_list[image_index]
            print(f'Registration: ROI {roi_index} | marker {marker_name}')

            if image_index == 0: # first image is the anchor image (reference)
                # get and save first tile
                image_path = self.raw_images_path[image_index]
                self.get_fixed_tile(image_path, mother_coordinates)
                
                # execute colour deconvolution for the current tile
                color_deconv_matrix = color_matrix.roi_evaluation(self.fixed_roi)
                imDeconvolved_fixed = ts.preprocessing.color_deconvolution.color_deconvolution(self.fixed_roi, color_deconv_matrix)
                
                fixed_ROI_hemo = imDeconvolved_fixed.Stains[:, :, 0] #extract hemotoxylin channel for registration of consecutively stained tiles
                fixed_ROI_hemo = np.invert(fixed_ROI_hemo)
                fixed_ROI_hemo[fixed_ROI_hemo>255] = 255

                RGB_image_array = self.fixed_roi #duplicate to be used in the else
                RGB_image_array = self.fixed_roi[difference_buffer: -difference_buffer, difference_buffer: -difference_buffer, :]
                linear_shift = [0,0]
                
                fixed_ROI_visual = self.fixed_roi[self.registration_buffer: -self.registration_buffer, self.registration_buffer :-self.registration_buffer, :]
                
                resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
                metadata = {'PhysicalSizeX': self.scale_factor, 'PhysicalSizeY': self.scale_factor}
                image_name = f'{self.sample_name}_ROI#{roi_index}_{marker_name}.ome.tif'
                ImageHandler.save_tiff(fixed_ROI_visual, image_name, self.rgb_dir, resolution, metadata)


            else:
                self.get_moving_tile(image_path, mother_coordinates, registered_masks, image_index)
                # execute colour deconvolution for the current tile
                color_deconv_matrix = color_matrix.roi_evaluation(self.moving_roi)
                imDeconvolved_moving = ts.preprocessing.color_deconvolution.color_deconvolution(self.moving_roi, color_deconv_matrix)
                moving_ROI_hemo = imDeconvolved_moving.Stains[:, :, 0] #extract hemotoxylin channel
                moving_ROI_hemo = np.invert(moving_ROI_hemo)
                moving_ROI_hemo[moving_ROI_hemo>255] = 255

                elastixImageFilter = sitk.ElastixImageFilter()
                elastixImageFilter.LogToConsoleOff()
                elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixed_ROI_hemo))
                elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(moving_ROI_hemo))
                parameterMapVector = sitk.GetDefaultParameterMap('translation') #basic translation registration per tile
                parameterMapVector['MaximumNumberOfIterations'] = ['1000']
                parameterMapVector['NumberOfResolutions'] = ['5']
                parameterMapVector['NumberOfSpatialSamples'] = ['4000']
                elastixImageFilter.SetParameterMap(parameterMapVector)
                elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine')) #basic geometric warping registration per tile
                parameterMapVector['Metric'] = ['AdvancedKappaStatistic']
                elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('bspline')) #elastic registration per tile

                elastixImageFilter.Execute()

                reg_map = elastixImageFilter.GetTransformParameterMap()[0]
                [x_shift, y_shift] = [-float(reg_map['TransformParameters'][0]), -float(reg_map['TransformParameters'][1])]
                linear_shift = [y_shift, x_shift]

                #apply registered parameter map to all 3 RGB color channels
                reg_map = elastixImageFilter.GetTransformParameterMap()
                transformixImageFilter = sitk.TransformixImageFilter()
                transformixImageFilter.LogToConsoleOff()
                transformixImageFilter.SetTransformParameterMap(reg_map)
                transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(self.moving_roi[:,:,0])) #for RED
                transformixImageFilter.Execute()

                temp = transformixImageFilter.GetResultImage()
                red_current_tile = sitk.GetArrayFromImage(temp)
                red_current_tile[red_current_tile > 255] = 255
                transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(self.moving_roi[:,:,1])) #for GREEN
                transformixImageFilter.Execute()

                temp = transformixImageFilter.GetResultImage()
                green_current_tile = sitk.GetArrayFromImage(temp)
                green_current_tile[green_current_tile > 255] = 255
                transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(self.moving_roi[:,:,2])) #for BLUE
                transformixImageFilter.Execute()

                temp = transformixImageFilter.GetResultImage()
                blue_current_tile = sitk.GetArrayFromImage(temp)
                blue_current_tile[blue_current_tile > 255] = 255
                RGB_image_array = np.dstack((red_current_tile, green_current_tile, blue_current_tile)).astype(np.uint8)
                RGB_image_array[RGB_image_array > 255] = 255


                RGB_image_array_visual = RGB_image_array[self.registration_buffer: -self.registration_buffer, self.registration_buffer: -self.registration_buffer, :]
                RGB_image_array = RGB_image_array[difference_buffer: -difference_buffer, difference_buffer: -difference_buffer, :]
                self.visualization_figs['registration'].append([fixed_ROI_visual, RGB_image_array_visual])

                resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
                metadata = {'PhysicalSizeX': self.scale_factor, 'PhysicalSizeY': self.scale_factor}
                image_name = f'{self.sample_name}_ROI#{roi_index}_{marker_name}.ome.tif'
                ImageHandler.save_tiff(RGB_image_array_visual, image_name, self.rgb_dir, resolution, metadata)


            registered_roi = RGB_image_array
            self.registered_rois.append(registered_roi)
            self.linear_shifts.append(linear_shift)

        
        return self.registered_rois, self.linear_shifts, self.visualization_figs


    def inpaint_white_pixels(self, image_sitk):
        image_np = sitk.GetArrayFromImage(image_sitk).astype(np.float32)  # Ensure it's float32
        
        # Check and scale the image data if necessary
        min_val = np.min(image_np)
        max_val = np.max(image_np)
        if max_val > 1:  # Assuming the data isn't already normalized
            image_np = (image_np - min_val) / (max_val - min_val)  # Normalize to 0.0 - 1.0

        # Convert to uint8
        image_uint8 = (image_np * 255).astype(np.uint8)

        # Create a mask for inpainting
        mask = (image_uint8 >= 250).astype(np.uint8)  # Create mask for the highest values
        #mask = mask.astype(bool)

        # Perform inpainting
        #image_uint8[mask] = 0
        #inpainted_uint8 = image_uint8

        inpainted_uint8 = cv2.inpaint(image_uint8, mask, 3, cv2.INPAINT_NS)

        # Convert back to float32
        inpainted_np = inpainted_uint8.astype(np.float32) / 255.0

        # Convert back to SimpleITK Image
        #inpainted_image_sitk = sitk.GetImageFromArray(inpainted_np)

        return inpainted_np


    def get_registered_dapis(self, mother_coordinates, registered_masks, roi_index: int):
        """
        Input (implicit):
            -> fixed tile
            -> moving tile

        Output:
            <- list with registrated tiles
        """

        difference_buffer = int(self.registration_buffer - self.segmentation_buffer)
        registered_dapis = []
        for image_index, image_path in enumerate(self.raw_images_path):
            cycle_metadata = self.cycles_metadata[image_index]  # grabbing metadata
            cycle_dna_marker = cycle_metadata['dna_marker']
            cycle_markers = cycle_metadata['marker_name_list']
            print(f'Cycle #{image_index}')

            if image_index == 0: # first cycle is used as anchor for further registration
                image_path = self.raw_images_path[image_index]

                # get dapi channel from first cycle
                for channel_index, marker_name in enumerate(cycle_markers):
                    self.get_fixed_tile(image_path, mother_coordinates, channel_index=channel_index)

                    if self.fixed_roi.dtype == np.uint16:
                        self.fixed_roi = ImageHandler.sharpening(self.fixed_roi)
                        self.fixed_roi = cv2.normalize(self.fixed_roi, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

                    elif self.fixed_roi.dtype == np.uint8:
                        self.fixed_roi = ImageHandler.sharpening(self.fixed_roi)
                        self.fixed_roi = cv2.normalize(self.fixed_roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    RGB_image_array = self.fixed_roi[difference_buffer: -difference_buffer, difference_buffer: -difference_buffer]
                    linear_shift = [0,0]

                    if channel_index == 0:
                        fixed_ROI_dapi = self.fixed_roi
                        registered_dapis.append(RGB_image_array)
                    
                    fixed_ROI_visual = self.fixed_roi[self.registration_buffer: -self.registration_buffer, self.registration_buffer :-self.registration_buffer]

                    resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
                    metadata = {'PhysicalSizeX': self.scale_factor, 'PhysicalSizeY': self.scale_factor}
                    image_name = f'{self.sample_name}_ROI#{roi_index}_{self.marker_name_list[image_index]}_{marker_name}.ome.tif'
                    ImageHandler.save_tiff(fixed_ROI_visual, image_name, self.rgb_dir, resolution, metadata)

                    registered_roi = RGB_image_array
                    self.registered_rois.append(registered_roi)
                    self.linear_shifts.append(linear_shift)

            else:
                for channel_index, marker_name in enumerate(cycle_markers):
                    self.get_moving_tile(image_path, mother_coordinates, registered_masks, image_index, channel_index=channel_index)

                    current_dtype = self.moving_roi.dtype
                    if self.moving_roi.dtype == np.uint16:
                        self.moving_roi = ImageHandler.sharpening(self.moving_roi)
                        self.moving_roi = cv2.normalize(self.moving_roi, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

                    elif self.moving_roi.dtype == np.uint8:
                        self.moving_roi = ImageHandler.sharpening(self.moving_roi)
                        self.moving_roi = cv2.normalize(self.moving_roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    if channel_index == 0:
                        # get dapi from every sequential cycle
                        moving_ROI_dapi = self.moving_roi


                        elastix_image_filter = sitk.ElastixImageFilter()
                        elastix_image_filter.LogToConsoleOff()

                        fixed_image = sitk.GetImageFromArray(fixed_ROI_dapi)
                        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

                        moving_image = sitk.GetImageFromArray(moving_ROI_dapi)
                        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
                        
                        # Set the fixed and moving images
                        elastix_image_filter.SetFixedImage(fixed_image)
                        elastix_image_filter.SetMovingImage(moving_image)
                        
                        # Use a B-spline transformation type
                        parameter_map_bspline = sitk.GetDefaultParameterMap('bspline')
                        parameter_map_bspline['MaximumNumberOfIterations'] = ['1000']
                        parameter_map_bspline['NumberOfResolutions'] = ['3']
                        parameter_map_bspline['FinalGridSpacingInPhysicalUnits'] = ['100']  # Increase grid spacing to limit deformation
                        parameter_map_bspline['GridSpacingSchedule'] = ['3.0', '2.0', '1.0']
                        parameter_map_bspline['RegularizationType'] = ['BendingEnergy']
                        parameter_map_bspline['RegularizationWeight'] = ['0.01']  # Adjust based on the stiffness needed

                        
                        elastix_image_filter.SetParameterMap(parameter_map_bspline)
                        elastix_image_filter.Execute()
                        
                        # Get the result image and transformation
                        registered_image = elastix_image_filter.GetResultImage()
                        transform_param_map = elastix_image_filter.GetTransformParameterMap()

                        #print('TRANSFORM MAP')
                        #print(transform_param_map[0]['TransformParameters'])

                        current_tile = self.inpaint_white_pixels(registered_image)
                        #current_tile = sitk.GetArrayFromImage(registered_image_inpainted)

                        '''
                        registration_method = sitk.ImageRegistrationMethod()

                        # Set similarity metric
                        registration_method.SetMetricAsMeanSquares()  # Mean squares is good for similar image content

                        # Set the optimizer
                        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                                                     minStep=1e-4,
                                                                                     numberOfIterations=500,
                                                                                     gradientMagnitudeTolerance=1e-6)

                        # Set the interpolator
                        registration_method.SetInterpolator(sitk.sitkLinear)

                        # Use an identity transformation as a starting point
                        fixed_image = sitk.GetImageFromArray(fixed_ROI_dapi)
                        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
                        initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
                        registration_method.SetInitialTransform(initial_transform)

                        # Execute the registration
                        moving_image = sitk.GetImageFromArray(moving_ROI_dapi)
                        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
                        final_transform = registration_method.Execute(fixed_image, moving_image)


                        # Apply the transformation to the moving image
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(fixed_image)
                        resampler.SetInterpolator(sitk.sitkLinear)
                        resampler.SetDefaultPixelValue(0)  # Set the default pixel value here
                        resampler.SetTransform(final_transform)

                        registered_image = resampler.Execute(moving_image)

                        reg_map = final_transform.GetParameters()
                        [x_shift, y_shift] = [-float(reg_map[0]), -float(reg_map[1])]
                        linear_shift = [y_shift, x_shift]

                        current_tile = sitk.GetArrayFromImage(registered_image)
                        '''
                    else:
                        # apply same transform object to the rest of the channels for the same image
                        moving_channel = sitk.GetImageFromArray(self.moving_roi)
                        moving_channel = sitk.Cast(moving_channel, sitk.sitkFloat32)

                        transformix_image_filter = sitk.TransformixImageFilter()
                        transformix_image_filter.LogToConsoleOff()
                        transformix_image_filter.SetMovingImage(moving_channel)
                        transformix_image_filter.SetTransformParameterMap(transform_param_map)
                        transformix_image_filter.Execute()

                        registered_channel = transformix_image_filter.GetResultImage()

                        current_tile = self.inpaint_white_pixels(registered_channel)
                        #current_tile = sitk.GetArrayFromImage(registered_channel_inpainted)
                        '''
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(fixed_image)
                        resampler.SetInterpolator(sitk.sitkLinear)
                        resampler.SetDefaultPixelValue(0)  # Set the default pixel value here
                        resampler.SetTransform(final_transform)
                        registered_channel = resampler.Execute(moving_channel)

                        current_tile = sitk.GetArrayFromImage(registered_channel)
                        '''
                    if current_dtype == np.uint16:
                        #RGB_image_array = current_tile.astype(np.uint16)
                        RGB_image_array = (current_tile * 65535).astype(np.uint16)

                    elif current_dtype == np.uint8:
                        #RGB_image_array = current_tile.astype(np.uint8)
                        RGB_image_array = (current_tile * 255).astype(np.uint8)

                    RGB_image_array_visual = RGB_image_array[self.registration_buffer: -self.registration_buffer, self.registration_buffer: -self.registration_buffer]
                    RGB_image_array = RGB_image_array[difference_buffer: -difference_buffer, difference_buffer: -difference_buffer]

                    if channel_index == 0: 
                        registered_dapis.append(RGB_image_array)

                    self.visualization_figs['registration'].append([fixed_ROI_visual, RGB_image_array_visual])
                   
                    resolution = (1e4 / self.scale_factor, 1e4 / self.scale_factor, 'CENTIMETER')
                    metadata = {'PhysicalSizeX': self.scale_factor, 'PhysicalSizeY': self.scale_factor}
                    image_name = f'{self.sample_name}_ROI#{roi_index}_{self.marker_name_list[image_index]}_{marker_name}.ome.tif'
                    ImageHandler.save_tiff(RGB_image_array_visual, image_name, self.rgb_dir, resolution, metadata)

                    registered_roi = RGB_image_array
                    self.registered_rois.append(registered_roi)
                    self.linear_shifts.append(linear_shift)

        print('Finished.') 
        return self.registered_rois, self.linear_shifts, self.visualization_figs, registered_dapis 
