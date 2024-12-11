from abc import ABC, abstractmethod

class Technology(ABC): 
    def __init__(self, name, pipeline):
        self.pipeline = pipeline
        self.name = name
        self.tissue_masks = []
        self.registered_masks = []
        self.image_names = []

    @abstractmethod
    def get_widgets(self, widgets_func):
        return {func.__name__: func() for func in widgets_func}

    @abstractmethod
    def initialization_widgets(self):
        ...

    @abstractmethod
    def masking_widgets(self):
        ...

    @abstractmethod
    def tiling_widgets(self):
        ...

    @abstractmethod
    def initialization(self, **kwargs):
        ...

    @abstractmethod
    def masking(self, **kwargs):
        ...

    @abstractmethod
    def prelim_registration(self, image_index):
        CYAN = '\033[96m'
        if image_index == 0:
            print(CYAN + 'Coordinates of mask registered. If multiple images were provided, the sequential images will be registered against the first one')

        else:
            print(CYAN + 'Preliminary registration performed. Please ensure the image on the right appropriately shows the tissues aligned.')

    @abstractmethod
    def tiling_and_erosion(self, **kwargs):
        ...

    @abstractmethod
    def _set_attributes(self, **kwargs):
        ...

    @abstractmethod
    def _fetch_name(self):
        ...

    @abstractmethod
    def _update_list(self, element, _list, _idx):
        if 0 <= _idx < len(_list):
            _list[_idx] = element
        else:
            _list.append(element)

        return _list


