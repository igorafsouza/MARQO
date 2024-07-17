import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import json
import numpy as np
import subprocess
import threading
import multiprocessing as mp
import concurrent.futures
from IPython.display import display, clear_output, Markdown
from ipywidgets import Button, HBox, VBox, interact, interactive, fixed, interact_manual, Image, Layout, interactive_output, Output, Button, IntSlider, Box, Text, IntProgress
from ipycanvas import MultiCanvas, hold_canvas, Canvas
from time import time, sleep
import queue
import random
import joblib
import traceback
from tqdm import tqdm
import contextlib

from technologies.appwidgets import technology_button, save_image_name, marker_name_list, dna_marker, cyto_distances 
from technologies.micsss_technology import MICSSSTechnology
from technologies.tma_technology import TMATechnology
from technologies.singleplex_technology import SinglePlexTechnology
from technologies.fluorescence_technology import IFTechnology
from technologies.cyclic_if_technology import CyIFTechnology

sys.path.append("..")
from analysis_per_tile import Pipeline
from clustering.kmeans import KMEANS

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



class Notebook:
    def __init__(self):
        self.analysis_output = Output()
        self.clustering_output = Output()
    
    def grab_instance(self, technology):
        # equivalent to a match/case statemet
        self.technology = {'MICSSS (single sample)': MICSSSTechnology(),
                           'MICSSS (multiple sample)': TMATechnology(),
                           'Singleplex IHC': SinglePlexTechnology(),
                           'mIF': IFTechnology(),
                           'CyIF': CyIFTechnology()}[technology]

        
    def sanity_check(self, **kwargs):
        if 'dna_marker' in kwargs:
            if kwargs['dna_marker'] not in kwargs['marker_name_list']:
                print('Check the marker spelling. DNA or TMA marker should exist in the list of markers.')
                return None

        if 'ref_channels' in kwargs:
            ref_channels = kwargs['ref_channels']
            kwargs['ref_channels'] = ref_channels.replace(' ', '').replace("'","").replace('"','').split(',') if ref_channels else []
            for ref_channel in kwargs['ref_channels']:
                if ref_channel not in kwargs['marker_name_list']:
                    print('Check the reference channel spelling. The channels should exist in the list of markers.')
                    return None
                        
        # Checking if provided paths exist
        if 'raw_images_path' in kwargs:
            raw_images_path = kwargs['raw_images_path']
            if not os.path.exists(raw_images_path):
                print('Kindly ensure path to images is correct.')
                return None
        
        if 'output_path' in kwargs:
            output_path = kwargs['output_path']
            if not os.path.exists(output_path):
                print('Kindly ensure export directory exists.')
                return None
        
        if 'output_path' in kwargs:
            sample_name = kwargs['sample_name']
            if os.path.exists(os.path.join(output_path, sample_name)):
                print(f'The sample directory already exists.\nKindly choose a different project name for {sample_name}\n')
        
        # Generating markers name list
        if 'marker_name_list' in kwargs and 'cyto_distances' in kwargs:
            marker_name_list = kwargs['marker_name_list']
            kwargs['marker_name_list'] = marker_name_list.replace(' ', '').replace("'","").replace('"','').split(',')
        
            # Generating cyto distances list
            cyto_distances = kwargs['cyto_distances']
            cyto_distances = [int(c) for c in cyto_distances.split(',')]
            kwargs['cyto_distances'] = cyto_distances
            
            # Checking len(cyto) == len(markers)
            if len(kwargs['cyto_distances']) != len(kwargs['marker_name_list']):
                print('Please, ensure list of cyto distances and list of markers are equal.')
                return
        
        # Converting output resolution to int
        if 'output_resolution' in kwargs:
            output_resolution = kwargs['output_resolution']
            kwargs['output_resolution'] = int(output_resolution.strip('x'))
        
        return kwargs
    
    
    def save_name(self, **widgs):
        image_name = widgs['image_name']
        marker = widgs['marker']
        full_image_path = os.path.join(self.technology.raw_images_path, image_name)
        if not os.path.exists(full_image_path):
            print('Please, make sure the image file is named correctly.')

        else:
            RED = '\033[91m'
            print('Image saved as: ' + RED + image_name)
            self.technology.image_names[marker] = image_name

    def set_image_names(self):        
        # Make sure we reset image_names, in case the user make modifications on initialization
        # and restart the QC steps
        self.technology.image_names = {}
        for idx, marker in enumerate(self.technology.markers):
            widg = save_image_name(description=f'Provide file name for {marker} :')
            widg.description = f'Provide file name for {marker} :'
            my_interact_manual = interact_manual.options(manual_name="Save name")
            my_interact_manual(self.save_name, **{'image_name': widg, 'marker': fixed(marker)})


    def save_cyif_images(self, **widgs):
        widgs = self.sanity_check(**widgs)
        full_image_path = os.path.join(self.technology.raw_images_path, widgs['image_name'])
        if not os.path.exists(full_image_path):
            print('Please, make sure the image file is named correctly.')
        
        else:
            RED = '\033[91m'
            print('Image saved as: ' + RED + widgs['image_name'])
            self.technology.image_names[widgs['cycle']] = {'image_name': widgs['image_name'],
                                    'marker_name_list': widgs['marker_name_list'],
                                    'dna_marker': widgs['dna_marker'].split(','),
                                    'cyto_distances': widgs['cyto_distances']
                                    }

            
    def set_cyif_images(self):        
        # Make sure we reset image_names, in case the user make modifications on initialization
        # and restart the QC steps
        self.technology.image_names = {}
        for n in range(int(self.technology.n_images)):
            widgs = {'image_name': save_image_name(description=f'Provide filame for cycle #{n + 1}: '),
                     'marker_name_list': marker_name_list(),
                     'dna_marker': dna_marker(),
                     'cyto_distances': cyto_distances(),
                     'cycle': fixed(n)
            }
            my_interact_manual = interact_manual.options(manual_name="Save name")
            my_interact_manual(self.save_cyif_images, **widgs)
            
        
    def set_attributes(self, **kwargs):
        kwargs = self.sanity_check(**kwargs)
        if kwargs is None:
            return

        self.technology._set_attributes(**kwargs)

        if self.technology.__class__.__name__ != 'CyIFTechnology':
            self.set_image_names()
    
        else:
            self.set_cyif_images()
    
    def initialize(self, technology):            
        if technology is not None:
            self.grab_instance(technology)
            widgs = self.technology.initialization_widgets()
            VBox([w for w in widgs.values()])

            my_interact_manual = interact_manual.options(manual_name="Initialize")
            my_interact_manual(self.set_attributes, **widgs)
                

    def manual_masking(self, image_index):
        tissue_img = self.technology.tissue_masks[image_index][5]

        if tissue_img.shape[2] == 3:  # If the image has 3 channels (RGB)
            tissue_img_rgba = np.concatenate([
                tissue_img, np.full((tissue_img.shape[0], tissue_img.shape[1], 1), 255, dtype=np.uint8)], axis=-1)  # Add alpha channel
        else:
            tissue_img_rgba = tissue_img  # If already RGBA

        mask = self.technology.tissue_masks[image_index][4]

        mask_height = tissue_img_rgba.shape[0]
        mask_width = tissue_img_rgba.shape[1]

        canvas = MultiCanvas(2, width=mask_width, height=mask_height)
        image_layer, drawing_layer = canvas[0], canvas[1]

        overlay_color = [7, 0, 245, 128] # Semi-transparent red
        overlay = np.zeros_like(tissue_img_rgba, dtype=np.uint8)  # Initialize the overlay
        overlay[mask] = overlay_color
        blended_image = np.where(overlay, overlay, tissue_img_rgba)

        image_layer.put_image_data(blended_image)

        brush_radius = 10
        y, x = np.ogrid[-brush_radius:brush_radius+1, -brush_radius:brush_radius+1]
        brush_mask = x**2 + y**2 <= brush_radius**2
        #drawing_layer.stroke_style = 'rgba(11, 229, 47, 0.2)'
        drawing_layer.fill_style = 'rgba(224, 88, 0, 0.2)'

        current_points = []
        is_drawing = False
        output = Output()

        brush_preview_canvas = Canvas(width=100, height=100)
        # Function to draw the brush preview
        def draw_brush_preview(size):
            with hold_canvas(brush_preview_canvas):
                brush_preview_canvas.clear()
                brush_preview_canvas.fill_style = 'black'
                brush_preview_canvas.fill_circle(brush_preview_canvas.width / 2,
                                                 brush_preview_canvas.height / 2, size)

        draw_brush_preview(brush_radius)

        brush_size_slider = IntSlider(
            value=brush_radius,
            min=1,
            max=50,
            step=1,
            description='Brush Size:',
            continuous_update=True  # Update only when the slider is released
        )

        def update_brush_size(change):
            nonlocal brush_radius, brush_mask
            brush_radius = change['new']
            y, x = np.ogrid[-brush_radius:brush_radius+1, -brush_radius:brush_radius+1]
            brush_mask = x**2 + y**2 <= brush_radius**2

            draw_brush_preview(brush_radius)

        brush_size_slider.observe(update_brush_size, names='value')

        def draw_circle(x, y):
            with hold_canvas(drawing_layer):
                drawing_layer.fill_circle(x, y, brush_radius)
                #drawing_layer.stroke_circle(x, y, brush_radius)

        def start_drawing(x, y):
            nonlocal is_drawing
            is_drawing = True
            current_points = [(x, y)]
            draw_circle(x, y)

        def keep_drawing(x, y):
            if not is_drawing: return
            current_points.append((x, y))
            draw_circle(x, y)

        def stop_drawing(x, y):
            nonlocal is_drawing
            is_drawing = False

        def clean_canvas(btn):
            nonlocal current_points
            with hold_canvas(drawing_layer):
                current_points = []
                drawing_layer.clear()

        def close_canvas(btn):
            output.clear_output()


        def apply_brush_template(btn):
            nonlocal mask, current_points
            for point in current_points:
                x = int(point[0])
                y = int(point[1])

                # Define the brush template bounds
                x_start = max(x - brush_radius, 0)
                x_end = min(x + brush_radius + 1, mask_width)
                y_start = max(y - brush_radius, 0)
                y_end = min(y + brush_radius + 1, mask_height)

                x_offset_start = max(brush_radius - (x - x_start), 0)
                x_offset_end = min(brush_radius + (x_end - x), 2 * brush_radius + 1)
                y_offset_start = max(brush_radius - (y - y_start), 0)
                y_offset_end = min(brush_radius + (y_end - y), 2 * brush_radius + 1)

                # Get the valid part of the brush mask
                valid_brush_mask = brush_mask[y_offset_start:y_offset_end, x_offset_start:x_offset_end]

                # Update the boolean mask
                if btn.description == 'Add Mask':
                    mask[int(y_start):int(y_end), int(x_start):int(x_end)] |= valid_brush_mask

                else:
                    mask[int(y_start):int(y_end), int(x_start):int(x_end)] &= ~valid_brush_mask

            overlay_color = [50, 0, 224, 128]

            overlay = np.zeros_like(tissue_img_rgba, dtype=np.uint8)  # Initialize the overlay

            overlay[mask] = overlay_color

            blended_image = np.where(overlay, overlay, tissue_img_rgba)
            with hold_canvas(image_layer):
                image_layer.clear()  # Clear existing drawings or images
                drawing_layer.clear()
                current_points = []
                image_layer.put_image_data(blended_image, 0, 0)


        canvas.on_mouse_down(start_drawing)
        canvas.on_mouse_move(keep_drawing)
        canvas.on_mouse_up(stop_drawing)

        clean_btn = Button(description='Clean', layout=Layout(width='105px'))
        clean_btn.on_click(clean_canvas)

        close_btn = Button(description='Close', layout=Layout(width='105px'))
        close_btn.on_click(close_canvas)
    
        pos_mask_btn = Button(description='Add Mask', layout=Layout(width='105px'), button_style='success')
        neg_mask_btn = Button(description='Remove Mask', layout=Layout(width='105px'), button_style='danger')

        pos_mask_btn.on_click(apply_brush_template)
        neg_mask_btn.on_click(apply_brush_template)

        def apply_mask(btn):
            tissue_mask = self.technology.tissue_masks[image_index]
            tissue_mask[4] = mask
            self.technology.manual_masking(image_index=image_index, mask=tissue_mask)

        apply_mask_btn = Button(description='Save Mask', layout=Layout(width='105px'))
        apply_mask_btn.on_click(apply_mask)

        brush_widgets = HBox([brush_size_slider, brush_preview_canvas])

        with output:
            canvas_container = Box(children=[canvas], layout=Layout(width='90%', overflow_x='scroll'))
            hbox = HBox([canvas_container, VBox([pos_mask_btn, neg_mask_btn, clean_btn, apply_mask_btn, close_btn])])

            #scrollable_container = Box(children=[VBox([brush_widgets, hbox])], layout=Layout(overflow_x='scroll', display='flex'))
            container = Box(children=[VBox([brush_widgets, hbox])])

            display(container)

        display(output)

    
    def masking_and_prelimreg(self):
        if self.technology.image_names == []:
            print('Please, provide first the images to proceed.')
            return
        
        # function to stage the directories and config file
        self.technology.initialization()
        
        #button for auto masking
        for image_index, image_name in enumerate(self.technology.image_names):
            widgs = self.technology.masking_widgets()
            widgs['image_index'] = fixed(image_index)
            
            out = interactive_output(self.technology.masking, widgs)
            ui = VBox([HBox([widgs['sigma'], 
                             widgs['n_thresholding_steps'], 
                             widgs['min_size'], 
                             widgs['crop_perc']]), 
                       widgs['high_sens_select'], 
                       widgs['stretch_select'],
                       widgs['range_stretch'],
                       widgs['local_select'],
                       widgs['disk_size']
                       ])

            out.layout.width = '60%'
            display(HBox([out, ui]))

            # button to perform manual masking
            my_interact_manual = interact_manual.options(manual_name="Manual Masking")
            my_interact_manual(self.manual_masking, image_index = fixed(image_index))
            
            #button for prelim registration adjacent to every tissue mask
            my_interact_manual = interact_manual.options(manual_name="Register")
            my_interact_manual(self.technology.prelim_registration, image_index = fixed(image_index))
        

    def annotate_cores(self, maskforclipping_index):
        tilemap_img = self.technology.tilemap_image

        if tilemap_img.ndim == 2:
            tilemap_img = np.invert(tilemap_img)
            tilemap_img_rgba = np.stack((tilemap_img, tilemap_img, tilemap_img, np.ones_like(tilemap_img) * 255), axis=-1)

        elif tilemap_img.ndim == 3 and tilemap_img.shape[2] == 1:
            tilemap_img = np.invert(tilemap_img)
            tilemap_img = np.squeeze(tilemap_img, axis=-1)
            tilemap_img_rgba = np.stack((tilemap_img, tilemap_img, tilemap_img, np.ones_like(tilemap_img) * 255), axis=-1)

        elif tilemap_img.ndim == 3 and tilemap_img.shape[2] == 3:
            alpha_channel = np.ones((tilemap_img.shape[0], tilemap_img.shape[1], 1)) * 255
            tilemap_img_rgba = np.concatenate((tilemap_img, alpha_channel.astype(tilemap_img.dtype)), axis=2)

        elif tilemap_img.ndim == 3 and tilemap_img.shape[2] == 4:
            tilemap_img_rgba = tilemap_img

        tilemap_height = tilemap_img_rgba.shape[0]
        tilemap_width = tilemap_img_rgba.shape[1]

        canvas = MultiCanvas(2, width=tilemap_width, height=tilemap_height)
        image_layer, drawing_layer = canvas[0], canvas[1]

        image_layer.put_image_data(tilemap_img_rgba)

        output = Output()

        start_point = None
        is_drawing = False
        image_layer.font = '30px Arial'
        current_rect = []
        cores_map = {}

        width = 0
        height = 0
        top_left_x = 0
        top_left_y = 0

        def on_mouse_down(x, y):
            nonlocal start_point, is_drawing
            start_point = (x, y)
            is_drawing = True

        def on_mouse_move(x, y):
            nonlocal start_point, is_drawing, current_rect, top_left_x, top_left_y, width, height
            if not is_drawing or not start_point:
                return
            
            width = abs(x - start_point[0])
            height = abs(y - start_point[1])
            top_left_x = min(start_point[0], x)
            top_left_y = min(start_point[1], y)
            
            with hold_canvas(drawing_layer):
                drawing_layer.clear()
                drawing_layer.fill_style = 'rgba(255, 0, 0, 0.5)'  # Semi-transparent red
                drawing_layer.fill_rect(top_left_x, top_left_y, width, height)

        def on_mouse_up(x, y):
            nonlocal start_point, is_drawing, current_rect, top_left_x, top_left_y, width, height
            is_drawing = False
            current_rect.append([top_left_x, top_left_y, width, height])

        drawing_layer.on_mouse_down(on_mouse_down)
        drawing_layer.on_mouse_move(on_mouse_move)
        drawing_layer.on_mouse_up(on_mouse_up)


        def clean_canvas(btn):
            with hold_canvas(drawing_layer):
                drawing_layer.clear()

        clean_btn = Button(description='Clean', layout=Layout(width='105px'))
        clean_btn.on_click(clean_canvas)

        label_box = Text(
            placeholder='Enter core label here',
            description='',
            disabled=False
        )


        def save_core(btn):
            nonlocal current_rect, cores_map
            label = label_box.value
            label_box.value = ''   # Clear the label box
            drawing_layer.clear()
            with hold_canvas(image_layer):
                top_left_x, top_left_y, width, height = current_rect[-1]
                image_layer.fill_style = 'rgba(0, 100, 0, 0.5)'
                image_layer.fill_rect(top_left_x, top_left_y, width, height)
                image_layer.fill_style = 'black'  # Text color
                image_layer.fill_text(label, top_left_x + width / 2, top_left_y + height / 2)

                current_rect = []

                # convert coordinates to create the cores map
                x = max((top_left_x * (self.technology.output_resolution / 1.25)) - self.technology.registered_masks[maskforclipping_index][4][1], 0)
                y = max((top_left_y * (self.technology.output_resolution / 1.25)) - self.technology.registered_masks[maskforclipping_index][4][0], 0)
                width = width * (self.technology.output_resolution / 1.25)
                height = height * (self.technology.output_resolution / 1.25)

                if label not in cores_map:
                    cores_map[label] = [[y, x, width, height]]

                else:
                    cores_map[label].append([y, x, width, height])

        def map_options(btn):
            nonlocal current_rect, cores_map, tilemap_img_rgba
            if btn.description == 'Save Map':
                save_path = os.path.join(self.technology.output_path, self.technology.sample_name, f'{self.technology.sample_name}_cores_map.json')
                with open(save_path, 'w+') as fout:
                    json.dump(cores_map, fout)
             
            else:
                image_layer.clear()
                cores_map = {}
                current_rect = []
                image_layer.put_image_data(tilemap_img_rgba)


        save_core_btn = Button(description='Save Core', layout=Layout(width='105px'))
        save_core_btn.on_click(save_core)

        save_map_btn = Button(description='Save Map', layout=Layout(width='105px'), button_style='success')
        clean_map_btn = Button(description='Restart Map', layout=Layout(width='105px'), button_style='danger')

        save_map_btn.on_click(map_options)
        clean_map_btn.on_click(map_options)

        with output:
            canvas_container = Box(children=[canvas], layout=Layout(width='90%', overflow_x='scroll'))
            hbox = HBox([canvas_container, VBox([label_box, save_core_btn, clean_btn, save_map_btn, clean_map_btn])])

            display(hbox)

        display(output)

            
    def tiling_erosion(self):
        if not hasattr(self, 'technology'):
            print('Please, you need to execute the previous steps first.')
            return
        
        elif self.technology.tissue_masks == []:
            print('Please, execute tissue masking first.')
            return
        
        elif self.technology.registered_masks == [] or len(self.technology.registered_masks) < len(self.technology.tissue_masks):
            print('Please, complete tissue masking and preliminary registration before proceed to tiling.')
            return

        dropdown_marker_list = [(f'{index}. {name}', index) for index, name in enumerate(self.technology.markers)]
        last_marker_key = max(dropdown_marker_list )[1]

        widgs = self.technology.tiling_widgets()

        widgs['marker_index_maskforclipping'].options = dropdown_marker_list
        widgs['marker_index_maskforclipping'].value = last_marker_key
        
        out = interactive_output(self.technology.tiling_and_erosion, widgs)
        ui = VBox([(HBox([widgs['tile_analysis_thresh'], 
                          widgs['erosion_buffer_microns']])), 
                   widgs['marker_index_maskforclipping']])
        
        out.layout.width = '60%'
        display(HBox([out, ui]))

        # button to perform core annotation
        my_interact_manual = interact_manual.options(manual_name="Annotate Cores")
        my_interact_manual(self.annotate_cores, maskforclipping_index = fixed(widgs['marker_index_maskforclipping'].value))


    @staticmethod
    def log_listener(result_queue, log_queue, progress_bar, n_tiles):
        completed_tasks = 0
        while completed_tasks < n_tiles:
            try:
                result = result_queue.get(timeout=0.1)
                log_queue.put(result)
                if "Processed ROI" in result:
                    completed_tasks += 1
                    progress_bar.value = completed_tasks

            except queue.Empty:
                continue

    @staticmethod
    def update_progress(future, progress_bar):
        result = future.result()
        if result and "Processed ROI" in result:
            progress_bar.value += 1
            print(result)


    def analysis(self):
        if not hasattr(self, 'technology'):
            print('Please, you need to execute the previous steps first.')
            return
        
        elif not hasattr(self.technology, 'num_tiles'):
            print('Please, execute tiling and erosion step first.')
            return
        
        with self.analysis_output:
            sample_directory = os.path.join(self.technology.output_path, self.technology.sample_name)

            pipeline = Pipeline(sample_directory, self.technology.name)

            tiles_tissue_percentage = pipeline.allsamples_tilestats['tissue_percentages']
            n_tiles = len(tiles_tissue_percentage)

            tiles_coordinates = pipeline.allsamples_tilestats['coordinates']

            results = []

            num_cores = 2

            try:
                #with joblib.Parallel(n_jobs=num_cores) as parallel:
                parameters_list = [
                    [i, pipeline.registered_masks, tiles_coordinates[i], tiles_tissue_percentage[i]]
                    for i in range(n_tiles)
                ]
                with tqdm_joblib(tqdm(desc="Processing", total=n_tiles)) as progress_bar:
                    joblib.Parallel(n_jobs=num_cores)(joblib.delayed(pipeline.instance.analysis)(params) for params in parameters_list)
                    
                print('All tiles processed.')

                print('Starting reconciliation ...')
                pipeline.cellmetrics_reconciliation(n_tiles)

                pipeline.tilestats_reconciliation(n_tiles)
                print('Finished.')

            except Exception as e:
                print(f'An error occurred: {e}')
                print(traceback.format_exc())

    def clustering(self):
        if not hasattr(self.technology, 'output_path'):
            print('Please, you need to provide the sample inputs before execute the clustering step.')
            return

        with self.clustering_output:
            sample_directory = os.path.join(self.technology.output_path, self.technology.sample_name)
            config_file = os.path.join(sample_directory, 'config.yaml')
            rawcellmetrics_file = os.path.join(sample_directory, f'{self.technology.sample_name}_rawcellmetrics.csv')

            if not os.path.exists(config_file):
                print('The current sample does not have a config file. Please, run all the previous steps.')
                return

            elif not os.path.exists(rawcellmetrics_file):
                print('The current sample does not have a rawcellmetrics file. Please, run the Analysis step first.')
                return

            kmeans = KMEANS()
            kmeans._set_parameters(config_file)

            results = []
            for i in range(len(self.technology.markers)):
                result = kmeans.start(i)
                results.append(result)

            kmeans.cluster_files_reconciliation()
           
            kmeans.write_perc_medians(results) 

        
    def _set_attributes(self, **kwargs):
        self.__dict__.update(kwargs) 
    

class Workflow:

    def initialization_markdown(self):
        markdown_content = '''# <center><i> LAUNCH APPLICATION </t></center>

## <center><i> **Choose technology** </i></center>

<center><b>MICSSS (single sample per slide) —</b> Multiplex immunohistochemical consecutive staining on a single slide (MICSSS) for singular whole-slide resected or biopsied tissues per slide.</center>
<center><b>MICSSS (multiple samples per slide) —</b> Multiplex immunohistochemical consecutive staining on a single slide (MICSSS) for tissue microarrays (TMA) or multiple samples per slide, desired to be clustered independently.</center>
<center><b>Singleplex IHC —</b> Immunohistochemistry for single-stain samples.</center>
<center><b>mIF —</b> Multiplex immunofluorescence staining for platforms including Orion, COMET, and CODEX.</center>
<center><b>CyIF —</b> Cyclic immunofluorescence staining.</center>

'''

        return markdown_content


    def masking_markdown(self):
        markdown_content = '''## *Step 1: Initialization*

Ensure each tissue mask appropriately captures the tissue. The left figure depicts the original thumbnail image, while the center depicts its binary mask and the right depicts the cropped image with the overlaid mask.

Adjust the variables via the widgets to the right to alter the tissue masking. 

<ul> <li> <i> <b> "sigma" </i></b> affects the blur of the mask, usually able to fill in smaller holes. 
<li> <i> <b>"n_thresh" </i></b>is an erosion-like effect that wears away at the boundaries of the mask.
<li> <i> <b>"min_size filter" </i></b>is a size-threshold filter that removes any standalone tissue mask objects with an area in pixels less than that selected.
<li> <i> <b>"% edge excluded" </i></b>removes boundaries from the edges of the original thumbnail image so that they will not be deemed positive tissue mask. It removes the selected percentage from both width and height.
<li>The <i> <b> "Adaptive equalization" </i></b>toggle enhances local contrast by redistributing pixel values based on their local intensity distribution. 
<li>The <i> <b> "Contrast stretch" </i></b>toggle expands the range of pixel values to cover the full intensity spectrum, enhancing overall image contrast. 
<li>The <i> <b> "Local equalization" </i></b>toggle applies a local histogram equalization using a circular disk-shaped neighborhood to enhance contrast in specific regions. 
'''

        return markdown_content


    def tiling_markdown(self):
        markdown_content = '''## *Step 2: Tiling and erosion*

Ensure all tiles appropriately capture the tissue. Tiling may be adjusted via the widgets on the right:

<ul> <li> <i> <b> "min % tile tissue area" </i> </b> is the minimum amount of tissue mask required for a tile to be overlaid. The figure only shows tiles above this threshold.
<li> <i> <b> "erosion amount (microns)" </i> </b> affects how much area is eroded away from the boundaries of the tissue, shown in cyan. This area will not be analyzed.
<li> <i> <b> "Marker index for clipping" </i> </b> is the marker used that determines the tiling. It is recommended to use the last stain, as this one will most likely contain the most tissue degradation.
'''

        return markdown_content


    def analysis_markdown(self):
        markdown_content = '''## *Step 3: Analysis*

Click below to perform the analysis per tile.
'''

        return markdown_content 


    def clustering_markdown(self):
        markdown_content = '''## *Step 4: Clustering*

Lastly, perform the clustering step. Afterwards, review the data to perform cell classification (positive-calling).
'''

        return markdown_content 


    def start(self):
        notebook = Notebook()
        
        display(Markdown(self.initialization_markdown()))

        box_layout = Layout(display='flex', 
                            flex_flow='row', 
                            width='100%', 
                            height='80px', 
                            align_items='center', 
                            justify_content='center')

        ui = HBox([technology_button], layout=box_layout)

        out = interactive_output(notebook.initialize, {'technology': technology_button})
        display(VBox([ui]))
        display(VBox([out]))

        display(Markdown(self.masking_markdown()))
        my_interact_manual = interact_manual.options(manual_name="Tissue masking")
        my_interact_manual(notebook.masking_and_prelimreg)

        display(Markdown(self.tiling_markdown()))
        my_interact_manual = interact_manual.options(manual_name="Tiling and erosion")
        my_interact_manual(notebook.tiling_erosion)

        display(Markdown(self.analysis_markdown()))
        my_interact_manual = interact_manual.options(manual_name="Analysis")
        my_interact_manual(notebook.analysis)

        display(notebook.analysis_output)

        display(Markdown(self.clustering_markdown()))
        my_interact_manual = interact_manual.options(manual_name="Clustering")
        my_interact_manual(notebook.clustering)

        display(notebook.clustering_output)
