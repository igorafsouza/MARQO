"""

"""

# whole namespaces
import cv2
import yaml
import large_image
import numpy as np
import matplotlib.pyplot as plt

# specific functions form namespaces
from histomicstk.saliency.tissue_detection import get_tissue_mask


#
from utils.image import ImageHandler


class Masking:
    """
    Masking Class defines the functions and calls needed to extract the tissue mask from the slide images.
    """

    def __init__(self):
        """
        Parameters are dinamically updated by function 'set_parameters()', since each marker has its individual group of values.
        """
        self.raw_images_path = ''
        self.marker_name = ''
        self.sample_name = ''
        self.high_sens = ''
        self.n_thresholding_steps = ''
        self.sigma = ''
        self.min_size = ''
        self.crop_perc = ''


    def set_parameters(self, config_file, image_index):
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


        try:
            marker = config_data['sample']['markers'][image_index]
            sample = config_data['sample']['name']
            self.raw_images_path = config_data['sample']['raw_images_full_path']

            self.marker_name = marker
            self.sample_name = sample
            
            self.high_sens = config_data['parameters']['masking'][marker]['high_sens_select']
            self.n_thresholding_steps = config_data['parameters']['masking'][marker]['n_thresholding_steps']
            self.sigma = config_data['parameters']['masking'][marker]['sigma']
            self.min_size = config_data['parameters']['masking'][marker]['min_size']
            self.crop_perc = config_data['parameters']['masking'][marker]['crop_perc']
            self.stretch_select = config_data['parameters']['masking'][marker]['stretch_select']
            self.stretch_range = config_data['parameters']['masking'][marker]['range_stretch']
            self.local_select = config_data['parameters']['masking'][marker]['local_select']
            self.disk_size = config_data['parameters']['masking'][marker]['disk_size']

        except:
            marker = config_data['sample']['markers'][image_index]
            sample = config_data['sample']['sample_name']
            self.raw_images_path = config_data['sample']['slides_path']

            self.marker_name = marker
            self.sample_name = sample
        
            label = f'{image_index}_{marker}'

            self.high_sens = config_data['masking_parameters'][label]['high_sens']
            self.n_thresholding_steps = config_data['masking_parameters'][label]['n_thresholding_steps']
            self.sigma = config_data['masking_parameters'][label]['sigma']
            self.min_size = config_data['masking_parameters'][label]['min_size']
            self.crop_perc = config_data['masking_parameters'][label]['crop_perc']

            self.stretch_select = False
            self.stretch_range = '0,255'
            self.local_select = False
            self.disk_size = 1
            
        

    def get_mask(self, image_index, image_type='HE'):
        """
        Description ...

        in:
            <-

        out:
            ->

        """

        a = large_image.getTileSource(self.raw_images_path[image_index])
        data = a.getMetadata()
        OG_width = data['sizeX']
        OG_height = data['sizeY']
        b = a.getRegionAtAnotherScale(sourceRegion=dict(left=0, top=0, width=OG_width, height=OG_height, units='base_pixels'), 
                                      targetScale=dict(magnification=1.25), format=large_image.tilesource.TILE_FORMAT_NUMPY)
        c = np.asarray(b[0])

        if image_type == 'HE':
            deconvolve_first = True

        elif image_type == 'IF':
            deconvolve_first = False
            c = np.invert(c)
            #c = ImageHandler.filter_adaptive_equalization(c)
            c = np.dstack([c, c, c])

        im_original = c[:, :, 0:3].astype(np.uint8)
        im = np.copy(im_original)

        #eliminate edges before tissue mask identification
        new_width = int(im.shape[1])
        new_height = int(im.shape[0])
        crop_perc = self.crop_perc / 100
        y_overlap = int(np.round(crop_perc * new_height))
        x_overlap = int(np.round(crop_perc * new_width))
        im[:y_overlap, :, :] = int(255)
        im[:, :x_overlap, :] = int(255)
        im[int(new_height - y_overlap): new_height, :, :] = int(255)
        im[:, int(new_width - x_overlap): new_width, :] = int(255)


        if self.local_select and not self.high_sens and not self.stretch_select:
            im = im[:, :, 2]
            im = ImageHandler.filter_local_equalization(im, disk_size=self.disk_size)
            im = np.invert(im)

        if self.stretch_select and not self.high_sens and not self.local_select:
            im = im[:, :, 2]
            _min = int(self.stretch_range.split(',')[0])
            _max = int(self.stretch_range.split(',')[1])
            im = ImageHandler.filter_contrast_stretch(im, low=_min, high=_max)
            im = np.invert(im)
        
        if self.high_sens and not self.stretch_select and not self.local_select:
            im = im[:, :, 2]
            im = ImageHandler.filter_adaptive_equalization(im)
            im = np.invert(im)

        #perform masking (replace im2 with im for color utilization -- less sensitivty)
        tissue_mask, biggest_tissue = get_tissue_mask(im, deconvolve_first = deconvolve_first , n_thresholding_steps = self.n_thresholding_steps, 
                                                      sigma = self.sigma, min_size = self.min_size)
        
        tissue_mask_as255 = tissue_mask.astype(np.uint8) * 255
        n_labels, labels, lobe_stats, centroids = cv2.connectedComponentsWithStats(tissue_mask_as255, connectivity = 8)
        tissue_mask = tissue_mask > 0

        #find where mask resides
        result = np.where(tissue_mask==True)
        buffer_zone = 200
        x_min = min(result[1]) - buffer_zone

        if x_min < 0:
            x_min = 0

        x_max = max(result[1]) + buffer_zone
        if x_max > len(im[1, :]):
            x_max = len(im[1, :])

        y_min = min(result[0]) - buffer_zone
        if y_min < 0:
            y_min = 0

        y_max = max(result[0]) + buffer_zone
        if y_max > len(im[:, 1]):
            y_max = len(im[:, 1])

        cropped = im_original[y_min:y_max,x_min:x_max]
        height = y_max - y_min
        width = x_max - x_min

        #overlay binary tissue mask to cropped original image
        open_cv_image = cropped[:, :, :].copy()
        tissue_mask_cropped = np.copy(tissue_mask)
        tissue_mask_cropped = tissue_mask_cropped[y_min:y_max,x_min:x_max]
        pos_ys, pos_xs = np.where(tissue_mask_cropped==True)
        alpha = 0.5

        for pos_pixel_index in range(len(pos_ys)):
            cv2.circle(open_cv_image, (pos_xs[pos_pixel_index], pos_ys[pos_pixel_index]), radius = 0, color = (0, 255, 255,200), thickness=-1)

        image_updated = cv2.addWeighted(open_cv_image, alpha, cropped, 1 - alpha, 0)
        image_data_mask = np.asarray(image_updated[:, :, :3])

        #display outputs
        fig = plt.figure(figsize = (15, 5), dpi = 100)
        graphs = fig.subplots(1, 3)
        graphs[0].imshow(im_original)
        graphs[0].axis('off')
        graphs[0].set_title('Image index #' + str(image_index) + '\n Sample name: ' + str(self.sample_name) + '\n Marker name: ' + self.marker_name)
        graphs[1].imshow(tissue_mask)
        graphs[1].set_title('Mask of tissue')
        graphs[2].imshow(image_data_mask)
        graphs[2].axis('off')
        graphs[2].set_title('Cropped image with mask overlaid')

        mask = [int(x_min), int(x_max), int(y_min), int(y_max), tissue_mask, im_original]

        return mask




        
