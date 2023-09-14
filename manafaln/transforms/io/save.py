from monai.transforms import SaveImage as _SaveImage
from monai.transforms import SaveImaged as _SaveImaged


class SaveImage(_SaveImage):
    """
    Overrides monai.transforms.SaveImage with options.

    Args:
        init_kwargs (dict, optional): Additional keyword arguments for initializing the SaveImage object.
        data_kwargs (dict, optional): Additional keyword arguments for processing the data.
        meta_kwargs (dict, optional): Additional keyword arguments for processing the metadata.
        write_kwargs (dict, optional): Additional keyword arguments for writing the image.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    """

    def __init__(
        self,
        init_kwargs=None,
        data_kwargs=None,
        meta_kwargs=None,
        write_kwargs=None,
        *args,
        **kwargs
    ):
        """
        Initializes the SaveImage object with the given options.

        Args:
            init_kwargs (dict, optional): Additional keyword arguments for initializing the SaveImage object.
            data_kwargs (dict, optional): Additional keyword arguments for processing the data.
            meta_kwargs (dict, optional): Additional keyword arguments for processing the metadata.
            write_kwargs (dict, optional): Additional keyword arguments for writing the image.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(*args, **kwargs)
        self.set_options(
            init_kwargs=init_kwargs,
            data_kwargs=data_kwargs,
            meta_kwargs=meta_kwargs,
            write_kwargs=write_kwargs,
        )


class SaveImaged(_SaveImaged):
    """
    Overrides monai.transforms.SaveImaged with options.

    Args:
        init_kwargs (dict, optional): Additional keyword arguments for initializing the SaveImaged object.
        data_kwargs (dict, optional): Additional keyword arguments for processing the data.
        meta_kwargs (dict, optional): Additional keyword arguments for processing the metadata.
        write_kwargs (dict, optional): Additional keyword arguments for writing the image.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    """

    def __init__(
        self,
        init_kwargs=None,
        data_kwargs=None,
        meta_kwargs=None,
        write_kwargs=None,
        *args,
        **kwargs
    ):
        """
        Initializes the SaveImaged object with the given options.

        Args:
            init_kwargs (dict, optional): Additional keyword arguments for initializing the SaveImaged object.
            data_kwargs (dict, optional): Additional keyword arguments for processing the data.
            meta_kwargs (dict, optional): Additional keyword arguments for processing the metadata.
            write_kwargs (dict, optional): Additional keyword arguments for writing the image.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(*args, **kwargs)
        self.set_options(
            init_kwargs=init_kwargs,
            data_kwargs=data_kwargs,
            meta_kwargs=meta_kwargs,
            write_kwargs=write_kwargs,
        )
