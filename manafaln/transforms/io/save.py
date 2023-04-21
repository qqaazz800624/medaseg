from monai.transforms import (
    SaveImage as _SaveImage,
    SaveImaged as _SaveImaged
)

class SaveImage(_SaveImage):
    """
    Overide monai.transforms.SaveImage with options.
    """
    def __init__(self,
        init_kwargs=None,
        data_kwargs=None,
        meta_kwargs=None,
        write_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.set_options(
            init_kwargs=init_kwargs,
            data_kwargs=data_kwargs,
            meta_kwargs=meta_kwargs,
            write_kwargs=write_kwargs
        )

class SaveImaged(_SaveImaged):
    """
    Overide monai.transforms.SaveImaged with options.
    """
    def __init__(self,
        init_kwargs=None,
        data_kwargs=None,
        meta_kwargs=None,
        write_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.set_options(
            init_kwargs=init_kwargs,
            data_kwargs=data_kwargs,
            meta_kwargs=meta_kwargs,
            write_kwargs=write_kwargs
        )
