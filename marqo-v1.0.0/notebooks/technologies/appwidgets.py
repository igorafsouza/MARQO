import ipywidgets as widgets

# Widgets definition for intialization
technology_button = widgets.ToggleButtons(
    value=None,
    options=['MICSSS (single sample)','MICSSS (multiple sample)', 'Singleplex IHC', 'mIF', 'CyIF'],
    #description='Technology under analysis.',
    #style = {'description_width': 'initial'},
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    #tooltips=['MICSS images', 'Singleplex image', 'TMA using MICSSS', 'Immunufluorescence images'],
    style=dict(
        button_width='200px',
        font_weight='bold',
))

def raw_images_path():
    raw_images_path = widgets.Text(
        value='/mnt/data',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Path to raw images:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return raw_images_path

def output_path():
    output_path = widgets.Text(
        value='/mnt/data',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Path to output folder (must already exist):',
        disabled=False,
        layout = widgets.Layout(width='1500px'))
    
    return output_path 

def sample_name():
    sample_name = widgets.Text(
        value='sample_name',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Provide name of sample:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return sample_name

def marker_name_list():
    marker_name_list = widgets.Text(
        value='marker1, marker2, marker3',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Provide list of markers, separated by commas:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return marker_name_list

def n_images():
    n_images = widgets.Text(
        value='',
        placeholder='# cycles',
        style = {'description_width': 'initial'},
        description='Provide the total number of IF cycles, not markers, performed:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return n_images

def cyto_distances():
    cyto_distances = widgets.Text(
        value='3, 3, 3',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Provide list of cytoplasm expansion values, separated by commas:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return cyto_distances

def n_imaages():
    n_images = widgets.Text(
        value='',
        placeholder='Number of images',
        style = {'description_width': 'initial'},
        description='Provide the number of IF images:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return n_images

def output_resolution():
    output_resolution = widgets.ToggleButtons(
        options=['20x','40x'],
        description='Output resolution',
        style = {'description_width': 'initial'},
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['ROIs output at 20x', 'ROIs output at 40x (only available if original images scanned at 40x)'],
        layout = widgets.Layout(width='1500px'))
    
    return output_resolution

# specific for TMA analysis
def tma_map_marker():
    tma_map_marker = widgets.Text(
        value='marker1',
        placeholder='Type something',
        style = {'description_width': 'initial'},
        description='Indicate which marker was used to create the TMA map file',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return tma_map_marker

#specific for IF analysis
def dna_marker():
    dna_marker = widgets.Text(
        value='marker',
        placeholder='DNA channel',
        style = {'description_width': 'initial'},
        description='Name of DNA-specific staining marker, corresponding to the list of marker names above:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return dna_marker

def ref_channels():
    dna_marker = widgets.Text(
        value='',
        placeholder='Autofluorescence channel',
        style = {'description_width': 'initial'},
        description='Name of autofluorescence channel, if it exists, corresponding to the list of marker names above:',
        disabled=False,
        layout = widgets.Layout(width='1500px'))

    return dna_marker

def save_image_name(description=''):
    save_image_name = widgets.Text(
            value = '',
            description = description,
            placeholder = 'Type something',
            style = {'description_width': 'initial'},
            disabled = False,
            layout = widgets.Layout(width='1500px'))

    return save_image_name


# Widgets for masking -- small workaround, using a function to wrap the widget enabling to create
# duplicates of the widget at js level
def sigma():
    sigma = widgets.IntSlider(
        value=3,
        min=1,
        max=30,
        step=1,
        description='sigma:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d')

    return sigma

def n_thresholding_steps():
    nthresh_select = widgets.IntSlider(
        value=1,
        min=1,
        max=50,
        step=1,
        description='n_thresh:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d')

    return nthresh_select

def min_size():
    min_size = widgets.IntSlider(
        value=2000,
        min=100,
        max=1000000,
        step=100,
        description='min_size filter:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d',
        layout={'width': 'max-content'})

    return min_size

def crop_perc():
    crop_perc = widgets.IntSlider(
        value=0,
        min=0,
        max=20,
        step=1,
        description='% edge excluded:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d',
        layout={'width': 'max-content'})
    
    return crop_perc

def high_sens_select():
    high_sens_select = widgets.Checkbox(
        value=False,
        description='Adaptative equalization',
        disabled=False,
        indent=False)

    return high_sens_select

def stretch_select():
    stretch_select = widgets.Checkbox(
        value=False,
        description='Constrast Stretch',
        disabled=False,
        indent=False)

    return stretch_select

def range_stretch():
    range_stretch = widgets.IntRangeSlider(
        value=[0, 255],
        min=0,
        max=255,
        step=1,
        description='Range:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')

    return range_stretch

def local_select():
    local_select = widgets.Checkbox(
        value=False,
        description='Local Equalization',
        disabled=False,
        indent=False)

    return local_select

def disk_size():
    disk_size = widgets.IntSlider(
        value=1,
        min=1,
        max=100,
        step=1,
        description='Disk Size:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    return disk_size

# Widgets for tiling and erosion
def tile_analysis_thresh():
    tile_analysis_thresh = widgets.IntSlider(
        value=15,
        min=5,
        max=70,
        step=5,
        description='min % tile tissue area:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d',
        layout={'width': 'max-content'})

    return tile_analysis_thresh

def erosion_buffer_microns():
    erosion_buffer_microns = widgets.IntSlider(
        value=20,
        min=0,
        max=100,
        step=1,
        description='erosion amount (microns):',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d',
        layout={'width': 'max-content'})

    return erosion_buffer_microns

def marker_index_maskforclipping():
    marker_index_maskforclipping = widgets.Dropdown(
        description = 'Marker index for clipping:',
        disabled = False,
        style = {'description_width': 'initial'},
        readout=True,
        readout_format='d',
        layout={'width': 'max-content'})

    return marker_index_maskforclipping
