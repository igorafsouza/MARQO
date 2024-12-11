
# whole namespaces
import os
import cv2
import json
import numexpr
import large_image
import numpy as np
import tifffile as tiff
import tempfile
import xml.etree.ElementTree as ET

# classes from namespaces
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import skimage.exposure as sk_exposure
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from scipy import ndimage
from skimage import measure, color, io

# functions from namespaces
from tifffile import imread

class ImageHandler:
    """
    ImageHandler is class contaning statich methods to handle extract metadata from images files, to manipualte and convert image arrays.
    """

    @staticmethod
    def np_to_pil(np_img):
        """
        Convert a NumPy array to a PIL Image.

        Args:
            np_img: The image represented as a NumPy array.

        Returns:
            The NumPy array converted to a PIL Image.
        """
        if np_img.dtype == "bool":
            np_img = np_img.astype("uint8") * 255

        elif np_img.dtype == "float64":
            np_img = (np_img * 255).astype("uint8")
            
        return Image.fromarray(np_img) 


    @staticmethod
    def mask_rgb(rgb, mask):
        """
        Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

        Args:
            rgb: RGB image as a NumPy array.
            mask: An image mask to determine which pixels in the original image should be displayed.

        Returns:
            NumPy array representing an RGB image with mask applied.
        """
        result = rgb * np.dstack([mask, mask, mask])

        return result

    @staticmethod
    def filter_histogram_equalization(np_img, nbins=256, output_type="uint8"):
        """
        Filter image (gray or RGB) using histogram equalization to increase contrast in image.

        Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        output_type: Type of array to return (float or uint8).

        Returns:
        NumPy array (float or uint8) with contrast enhanced by histogram equalization.
        """
        # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
        if np_img.dtype == "uint8" and nbins != 256:
            np_img = np_img / 255
        
        hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)

        if output_type == "float":
            pass
        
        else:
            hist_equ = (hist_equ * 255).astype("uint8")

            return hist_equ


    @staticmethod
    def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
        """
        Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
        is enhanced.

        Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        clip_limit: Clipping limit where higher value increases contrast.
        output_type: Type of array to return (float or uint8).

        Returns:
        NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
        """
        adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
        if output_type == "float":
            pass
        else:
            adapt_equ = (adapt_equ * 255).astype("uint8")

        return adapt_equ


    @staticmethod
    def filter_contrast_stretch(np_img, low=40, high=60):
        """
        Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
        a specified range.

        Args:
        np_img: Image as a NumPy array (gray or RGB).
        low: Range low value (0 to 255).
        high: Range high value (0 to 255).

        Returns:
        Image as NumPy array with contrast enhanced.
        """
        low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
        contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))

        return contrast_stretch


    @staticmethod
    def filter_local_equalization(np_img, disk_size=50):
        """
        Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.

        Args:
        np_img: Image as a NumPy array.
        disk_size: Radius of the disk structuring element used for the local histograms

        Returns:
        NumPy array with contrast enhanced using local equalization.
        """
        local_equ = sk_filters.rank.equalize(np_img, footprint=sk_morphology.disk(disk_size))

        return local_equ



    @staticmethod
    def open_image(filename):
        """
        Open an image (*.jpg, *.png, etc).

        Args:
            filename: Name of the image file.

        returns:
            A PIL.Image.Image object representing an image.
        """
        image = Image.open(filename)
        return image


    @staticmethod
    def pil_to_np_rgb(pil_img):
        """
        Convert a PIL Image to a NumPy array.

        Note that RGB PIL (w, h) -> NumPy (h, w, 3).

        Args:
            pil_img: The PIL Image.

        Returns:
            The PIL image converted to a NumPy array.
        """
        rgb = np.asarray(pil_img)
        if len(rgb.shape) == 3:
            tissue_mask = rgb[:, :, 0] > 0

        else:
            tissue_mask = rgb > 0

        return tissue_mask


    @staticmethod
    def open_image_np(filename):
        """
        Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

        Args:
            filename: Name of the image file.

        returns:
            np_img: A NumPy representing an RGB image.
            metadata: dictionary with metadata
        """
        pil_img = ImageHandler().open_image(filename)
        metadata = pil_img.text

        np_img = ImageHandler().pil_to_np_rgb(pil_img)
      
        return np_img, metadata   


    @staticmethod
    def save_image(im, saving_path: str, metadata={}) -> None:
        """
        Method to save PIL image along with its metadata

        Args:
            im: PIL image
            metadata: dict with metadata
            saving_path: complete path along with the name of the file to save it

        returns:
            Saves the image on disk
        """
        info = PngInfo()
        for key, value in metadata.items():
            info.add_text(key, str(value))

        im.save(saving_path, pnginfo=info)


    @staticmethod
    def save_tiff(im, image_name, dir, resolution: int, metadata: dict):
        saving_path = os.path.join(dir, image_name)
        with tiff.TiffWriter(saving_path) as tif:
            tif.write(im, resolution = resolution, metadata = metadata)

    @staticmethod
    def open_json(file_path):
        """
        Args:
            file_path: json file path

        returns:
            dictionary containing the data from the jsonå
        """
        with open(file_path, 'r') as stream:
            json_data = json.load(stream) 

        return json_data


    @staticmethod
    def get_registered_masks(marker_name_list, sample_name, registered_masks_path, technology):
        root, _, files = next(os.walk(registered_masks_path))

        registered_masks = []

        if technology != 'if':
            for _, marker_name in enumerate(marker_name_list):
                mask_name = sample_name + '_' + marker_name + '_thumbnail.png'
                mask_path = os.path.join(root, mask_name)
                _, metadata = ImageHandler().open_image_np(mask_path)
                registered_masks.append([float(value) if key != 'linear_shift' else json.loads(value) for key, value in metadata.items()])

        elif technology == 'if':
            for f in files:
                if '_thumbnail.png' in f:
                    mask_name = f
                    mask_path = os.path.join(root, mask_name)
                    _, metadata = ImageHandler().open_image_np(mask_path)
                    break

            registered_masks.append([float(value) if key != 'linear_shift' else json.loads(value) for key, value in metadata.items()])

        return registered_masks


    def sharpening(roi):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        image_sharp = cv2.filter2D(src=roi, ddepth=-1, kernel=kernel)

        return image_sharp
        

    @staticmethod
    def blur(roi, kernel=(3,3)):

        blur = cv2.GaussianBlur(roi, kernel, 2)
        #lower_white = np.array([50, 50, 50])
        #upper_white = np.array([180, 180, 180])
        #mask = cv2.inRange(blur, lower_white, upper_white)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        masked_img = cv2.bitwise_and(roi, roi, mask=mask)

        return masked_img


    @staticmethod
    def watershed(roi):
        '''
        Detach joint objects
        '''
        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #kernel = np.ones((3,3), np.uint8)
        #erosion = cv2.erode(thresh, kernel, iterations=2)

        cells = roi[:, :, 0]
        pixel_size = 0.5031
        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=10)

        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

        ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret3, markers = cv2.connectedComponents(sure_fg)

        markers = markers+10

        markers[unknown==255] = 0

        markers = cv2.watershed(roi, markers)

        roi[markers == -1] = [0,255,255]  

        img2 = color.label2rgb(markers, bg_label=0)

        return roi


    @staticmethod
    def get_eroded_thumbnail(tissue_mask, erosion_buffer_microns, output_resolution, scale_factor):
        
        #clip cells based on marker index and erosion amount
        thumbnail_mask_forclipping = tissue_mask[4]
        erosion_kernel = np.ones((3, 3), np.uint8)

        num_iterations = int(np.floor((erosion_buffer_microns * 1.25 / output_resolution) / scale_factor))
        eroded_thumbnail_mask_forclipping = (cv2.erode(thumbnail_mask_forclipping.astype(np.uint8), erosion_kernel, iterations=num_iterations)).astype(bool)
        
        return eroded_thumbnail_mask_forclipping


    @staticmethod
    def subtract_af(roi, fixed_tiles, af_channels):
        mask = roi > 0

        subtracted = roi
        for i in af_channels:
            normalized_af_channel = fixed_tiles[i] / np.max(fixed_tiles[i])

            scaled_af_channel = normalized_af_channel * np.max(subtracted)

            subtracted = np.maximum(subtracted - scaled_af_channel, 0)

            subtracted[mask == False] = 0

        subtracted = subtracted.astype(roi.dtype)

        return subtracted


    @staticmethod
    def parse_resolution_and_tile_size_from_ome_xml(ome_xml):
        """
        Parse OME-XML metadata to get resolution in DPI, resolution unit, and tile size.

        Parameters:
            ome_xml (str): The OME-XML metadata as a string.

        Returns:
            tuple: (resolution_x_dpi, resolution_y_dpi, resolution_unit, tile_width, tile_height)
        """
        # Parse the XML to get physical sizes and units
        root = ET.fromstring(ome_xml)
        namespace = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
        pixels = root.find(f".//{namespace}Pixels")

        # Extract physical pixel sizes and units
        physical_size_x = float(pixels.get("PhysicalSizeX", 1.0))
        physical_size_y = float(pixels.get("PhysicalSizeY", 1.0))
        unit_x = pixels.get("PhysicalSizeXUnit", "µm")
        unit_y = pixels.get("PhysicalSizeYUnit", "µm")

        # Extract or set default tile size
        tile_width = int(pixels.get("TileWidth", 1024))  # default tile width
        tile_height = int(pixels.get("TileHeight", 1024))  # default tile height

        # Convert to DPI if unit is µm or mm
        if unit_x == "µm" and unit_y == "µm":
            # 1 inch = 25400 µm
            resolution_x_dpi = int(25400 / physical_size_x)
            resolution_y_dpi = int(25400 / physical_size_y)
            resolution_unit = "INCH"

        elif unit_x == "mm" and unit_y == "mm":
            # 1 inch = 25.4 mm
            resolution_x_dpi = int(25.4 / physical_size_x)
            resolution_y_dpi = int(25.4 / physical_size_y)
            resolution_unit = "INCH"

        else:
            # If units are not in µm or mm, default to None
            resolution_x_dpi = None
            resolution_y_dpi = None
            resolution_unit = "UNKNOWN"

        return resolution_x_dpi, resolution_y_dpi, resolution_unit, tile_width, tile_height


    @staticmethod
    def get_ts_tif_pages(image_path, channel_index):
        with tiff.TiffFile(image_path) as tif:
            # Check if the specified channel index is within bounds
            if channel_index >= len(tif.series[0].pages):
                raise ValueError(f"Channel index {channel_index} is out of bounds.")

            # Read the specified channel (page) as a numpy array
            channel_array = tif.series[0].pages[channel_index].asarray()
            ome_xml = tif.ome_metadata

        resolution_x_dpi, resolution_y_dpi, resolution_unit, tile_width, tile_height = ImageHandler.parse_resolution_and_tile_size_from_ome_xml(ome_xml)

        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=True) as temp_file:
            # Save numpy array to the temporary TIFF file
            #tiff.imwrite(temp_file.name, channel_array)
            with tiff.TiffWriter(temp_file.name, ome=True) as tiff_writer:
                tiff_writer.write(
                    channel_array,
                    tile=(tile_width, tile_height),
                    resolution=(resolution_x_dpi, resolution_y_dpi),  # Set resolution to match physical size
                    resolutionunit=resolution_unit
                )

            # Load the temporary TIFF file using large_image
            ts = large_image.getTileSource(temp_file.name)

        return ts

    @staticmethod
    def acquire_image(image_path, y_min, x_min, Height, Width, mag, channel_index=0, technology='micsss'):

        #ensure input types are correct and rounded   
        Height = int(np.round(Height))
        Width = int(np.round(Width))
        y_min = int(np.round(y_min))
        x_min = int(np.round(x_min))

        Height_ori = Height
        Width_ori = Width

        #grab info
        a = large_image.getTileSource(image_path)
        data = a.getMetadata()
        if technology in ['if', 'cycif'] and not 'frames' in data:
            #print('[X] No frames')
            # Get a new tilesource specific for the channel using tiff pages
            a = ImageHandler.get_ts_tif_pages(image_path, channel_index)

        OG_width = int(np.round(data['sizeX']))
        OG_height = int(np.round(data['sizeY']))
        native_res = int(np.round(data['magnification'])) #accounts for images taken at different base resolutions
        b = a.convertRegionScale(sourceRegion=dict(left=0, top=0, width=OG_width, height=OG_height, units='mag_pixels'), 
                                 sourceScale=dict(magnification=native_res), targetScale=dict(magnification=mag), targetUnits='mag_pixels')

        current_width = int(np.round(b['width']))
        current_height = int(np.round(b['height']))

        # check bounds and append black pixels if region is out of bounds
        y_start_index = 0
        y_end_index = Height
        x_start_index = 0
        x_end_index = Width

        if y_min < 0:
            y_start_index = np.abs(y_min)
            y_min = 0
            Height = int(np.round(Height - y_start_index)) + 1

        if x_min < 0:
            x_start_index = np.abs(x_min)
            x_min = 0
            Width = int(np.round(Width - x_start_index)) + 1

        if int(y_min + Height) > current_height:
            y_end_index = current_height - np.abs(y_min)
            Height = int(np.round(Height - y_start_index))

        if int(x_min + Width) > current_width:
            x_end_index = current_width - np.abs(x_min)
            Width = int(np.round(Width - x_start_index))

        # #grab existing image array
        if technology in ['if', 'cycif'] and not 'frames' in data:
            b = a.getRegion(region = dict(left = int(np.round(x_min)), top = int(np.round(y_min)), width = int(np.round(Width)), height = int(np.round(Height)), 
                                      units='mag_pixels'), scale = dict(magnification=mag), format=large_image.tilesource.TILE_FORMAT_NUMPY)

        else:
            b = a.getRegion(frame = channel_index, region = dict(left = int(np.round(x_min)), top = int(np.round(y_min)), width = int(np.round(Width)), height = int(np.round(Height)), 
                                      units='mag_pixels'), scale = dict(magnification=mag), format=large_image.tilesource.TILE_FORMAT_NUMPY)

        c = np.asarray(b[0])

        if technology in ['if', 'cycif']:
            # IF
            bit_depth = c.dtype
            temp_image_array = np.zeros((Height, Width)).astype(bit_depth)
            empty_array = np.zeros((Height_ori, Width_ori)).astype(bit_depth)
            image_array = c[:(Height), :(Width), 0]

        else:
            # IHC
            bit_depth = c.dtype
            temp_image_array = np.zeros((Height, Width, 3)).astype(bit_depth)
            empty_array = np.zeros((Height_ori, Width_ori, 3)).astype(bit_depth)
            image_array = c[:(Height), :(Width), 0:3].astype(bit_depth)

        #check for rounding errors -- if image is smaller than empty array
        diff_height = Height
        diff_width = Width
        if image_array.shape[0] < Height:
            diff_height = image_array.shape[0] - Height

        if image_array.shape[1] < Width:
            diff_width = image_array.shape[1] - Width
    
        # ALERT: this is the line gettin erro because of negative dimensions
        if len(image_array.shape) == 3:
            temp_image_array[0: diff_height, 0: diff_width, :] = image_array

        else:
            temp_image_array[0: diff_height, 0: diff_width] = image_array

        image_array = temp_image_array

        #check for rounding errors -- if empty array is smaller than image
        if (y_end_index - y_start_index) < (image_array.shape[0]):
            y_end_index = y_start_index +  (image_array.shape[0])
            if y_end_index > empty_array.shape[0]:
                y_amount_over = y_end_index - empty_array.shape[0]
                y_end_index -= y_amount_over
                y_start_index -= y_amount_over

        if (x_end_index - x_start_index) < (image_array.shape[1]):
            x_end_index = x_start_index + (image_array.shape[1])

            if x_end_index > empty_array.shape[1]:
                x_amount_over = x_end_index - empty_array.shape[1]
                x_end_index -= x_amount_over
                x_start_index -= x_amount_over

        # #create output array
        if len(image_array.shape) == 3:
            empty_array[int(np.round(y_start_index)): int(np.round(y_end_index)), int(np.round(x_start_index)): int(np.round(x_end_index)), :] = image_array

        else:
            empty_array[int(np.round(y_start_index)): int(np.round(y_end_index)), int(np.round(x_start_index)): int(np.round(x_end_index))] = image_array

        return empty_array


    @staticmethod
    def get_base_res(image_path: str):
        a = large_image.getTileSource(image_path)
        data = a.getMetadata()
        base_res = data['magnification']

        return base_res


    @staticmethod
    def transform_image(img, axis_norm = (0,1), add_hemo=False, pmin=1, pmax=99.8, clip = True):  #normalizes and transforms image to work better in model preditcion function
       
        #check if img is valid
        errormsg = 'Must pass img as numpy object (np.ndarray) or string that is path to image file'
    
        if type(img) is str:
            assert os.path.isfile(img), errormsg+'\n FILE PATH GIVEN IS INVALID'
            img = np.array(imread(img))

        elif type(img) is np.ndarray:
            pass

        else: 
            assert False, errormsg
        
        n_channel = 1 if img.ndim == 2 else img.shape[-1]
        
        #reduce dimensions, normalize, and hemo_filter image
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[...,:3]

        img_normed = ImageHandler.normalize_image(img, pmin,pmax, axis=axis_norm, clip = clip)
    
        if add_hemo and len(img.shape) == 3 and img.shape[2]==3:
            for page in range(img.shape[2]):
                img_normed[...,page] = img_normed[...,page]*hemofinal[page]
            print('hemo_added')
        return img_normed 
    

    @staticmethod
    def normalize_image(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
        """CUSTOM Percentile-based image normalization."""
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)
        maxmi = [100,100,150]
    
        try:
            mi = [min(mi[0][0][i],maxmi[i]) for i in range(0, len(mi[0][0]))]
        except TypeError:
            assert False, 'Attempted to insert image that\'s not RGB. Image must have 3 dimensions.'
    
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)
        try:
            x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
        except ImportError or ModuleNotFoundError:
            x = (x - mi) / ( ma - mi + eps )    
    
        if clip:
            x = np.clip(x,0,1)
        return x
    

    @staticmethod
    def get_frame_by_index(image_path, frame_index, width, height):
        source = large_image.open(image_path)
        frame = source.getRegionAtAnotherScale(frame=frame_index,sourceRegion = dict(left = 0, top = 0, width = width, height = height, units = 'base_pixels'), 
                                               targetScale = dict(magnification = 1.25), format = large_image.tilesource.TILE_FORMAT_NUMPY)

        m = ImageHandler.mask_rgb(frame[0], frame[0])

        mask = [0, width, 0, height, m]
        return mask

    @staticmethod
    def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
        if dtype is not None:
            x = x.astype(dtype, copy=False)
            mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
            ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
            eps = dtype(eps)

        if numexpr is None:
            x = (x - mi) / (ma - mi + eps)
        else:
            x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")

        if clip:
            x = np.clip(x, 0, 1)

        return x
        
    @staticmethod
    def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
        """Percentile-based image normalization."""
        mi = np.percentile(x, pmin, axis=axis, keepdims=True)
        ma = np.percentile(x, pmax, axis=axis, keepdims=True)
        return ImageHandler.normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)



    @staticmethod
    def _create_figure(self, roi_img, x, y, scale_factor=1.0):
        # Constants
        img_width = roi_img.shape[1]
        img_height = roi_img.shape[0]

        fig = go.FigureWidget()

        fig.add_trace(
            go.Scatter(
                x=np.round((x * scale_factor), decimals=1),
                y=np.round((y * scale_factor), decimals=1),
                mode="markers",
                marker_opacity=0
            )
        )

        fig.data[0].marker.color = ['#36454f'] * len(x)
        fig.data[0].marker.opacity = [0] * len(x)


        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )

        # Overlay ROI img onto scatter plot
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=fromarray(roi_img))
        )

        # Configure other layout
        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            hovermode='closest',
        )

        return fig
        
