import os
import sys
from .appwidgets import *
from .technology_abc import Technology

sys.path.append("..")
from pipelines.singleplex import SinglePlex

class SinglePlexTechnology(Technology):
    def __init__(self):
        super().__init__('singleplex', SinglePlex())


    def get_widgets(self, widgets_func):
        return super().get_widgets(widgets_func)


    def initialization_widgets(self):
        widgets_func = [raw_images_path, output_path, sample_name, marker_name_list, cyto_distances, output_resolution]
        initialize_widgets = self.get_widgets(widgets_func)
        return initialize_widgets


    def masking_widgets(self):
        widgets_func = [sigma, n_thresholding_steps, min_size, crop_perc, high_sens_select, stretch_select, range_stretch, local_select, disk_size]
        masking_widgets = self.get_widgets(widgets_func)
        return masking_widgets


    def tiling_widgets(self):
        widgets_func = [tile_analysis_thresh, erosion_buffer_microns, marker_index_maskforclipping]
        tiling_widgets = self.get_widgets(widgets_func)
        return tiling_widgets


    def initialization(self):
        self.pipeline.initialization(self.image_names, self.sample_name, self.marker_name_list, self.output_resolution, self.raw_images_path, self.output_path, self.cyto_distances, self.segmentation_model)


    def masking(self, **kwargs):
        image_index = kwargs['image_index']
        del kwargs['image_index']  # removing image_index from kwargs

        # basic tranformation from tuple to string
        aux = kwargs['range_stretch']
        kwargs['range_stretch'] = f'{aux[0]},{aux[1]}'

        self.pipeline._add_parameters('masking', image_index, **kwargs)

        mask = self.pipeline.tissue_masking(image_index)

        self.tissue_masks = self._update_list(mask, self.tissue_masks, image_index)

    def prelim_registration(self, image_index):
        registered_mask = self.pipeline.prelim_registration(image_index, self.tissue_masks)

        self.registered_masks = self._update_list(registered_mask, self.registered_masks, image_index)

        super().prelim_registration(image_index)


    def tiling_and_erosion(self, **kwargs):
        marker_index_maskforclipping = kwargs['marker_index_maskforclipping']

        self.pipeline._add_parameters('tiling', marker_index_maskforclipping, **kwargs)

        tilestats, tilemap_image = self.pipeline.tiling(self.registered_masks, self.tissue_masks)

        self.num_tiles = len(tilestats['coordinates'])

        self.tilemap_image = tilemap_image



    def color_deconvolution(self):
        self.pipeline.deconvolution(self.registered_masks)


    def _fetch_name(self, marker, sample_name):
        _, _, fs = next(os.walk(self.raw_images_path))
        for f in fs:
            if (marker.lower() in f.lower()) and (sample_name.lower() in f.lower()) and (f.endswith('.svs') or f.endswith('.ndpi')):
                return f

        return ''


    def _set_attributes(self, **kwargs):
        self.__dict__.update(kwargs)
        self.markers = kwargs['marker_name_list']


    def _update_list(self, element, _list, _idx):
        return super()._update_list(element, _list, _idx)

