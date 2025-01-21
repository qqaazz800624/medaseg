from typing import Any, Dict, Hashable, List, Mapping, Tuple

from monai.config import KeysCollection
from monai.transforms import MapTransform, Transform
from monai.utils import ensure_tuple, ensure_tuple_rep

# TODO: Parse penWidth

class ParseXAnnotationSegmentationLabel(Transform):
    """
    Parse segmentation labels in JSON annotations from the XAnnotation tool by EBM.

    Args:
        item_keys (List[str]): a list of item keys to parse from the JSON annotations.

    Returns:
        List[List[List[float]]]: a list of lists, where each sublist corresponds to an item
        key and contains a list of coordinates.
    """
    def __init__(
        self,
        item_keys: List[str],
    ):
        self.item_keys = ensure_tuple(item_keys)

    def get_pts(
        self,
        segments: List[Dict[str, List[float]]],
        size: List[float]
    ) -> List[List[float]]:
        """
        Get list of point from segments, normalized with size.
        Args:
            segments: list of segments, each segment is {"a": [x1, y1], "b": [x2, y2]}
            size: size of original image, [W, H]
        Returns:
            pts: list of [x, y]
        """
        pts = []
        W, H = size
        for segment in segments:
            x, y = segment["a"]
            x = float(x / W)
            y = float(y / H)
            pts.append([x, y])

        x, y = segment["b"]
        x = float(x / W)
        y = float(y / H)
        pts.append([x, y])
        return pts

    def __call__(self, json_obj: Dict) -> List[List[List[float]]]:
        ptss = {}

        size = json_obj["size"]  # [W, H]

        shapes = json_obj["shapes"]
        for shape in shapes:
            label = shape.get("strokeColor")
            if shape["segments"] == []:
                continue
            pts = self.get_pts(shape["segments"], size)   # x, y
            ptss[label] = ptss.get(label, []) + [pts]

        data = [
            ptss.get(key, []) for key in self.item_keys
        ]

        return data

class ParseXAnnotationSegmentationLabeld(MapTransform):
    """
    A transform that parses the JSON annotations from the XAnnotation tool by EBM to get segmentation labels.

    Args:
        keys (KeysCollection): keys to pick data for transformation.
        item_keys (List[str]): a list of item keys to parse from the JSON annotations.
        allow_missing_keys (bool): whether to allow missing keys.

    Returns:
        Dict[Hashable, Any]: a dictionary containing the transformed data.
    """
    def __init__(
        self,
        keys: KeysCollection,
        item_keys: List[str],
        allow_missing_keys: bool=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = ParseXAnnotationSegmentationLabel(item_keys=item_keys)

    def __call__(
        self,
        data: Mapping[Hashable, Dict]
    ) -> Dict[Hashable, Any]:
        """
        Args:
            data (Mapping[Hashable, Dict]): input data to be transformed.

        Returns:
            Dict[Hashable, Any]: transformed data.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d

class ParseXAnnotationDetectionLabel(Transform):
    """
    A transform that parses the JSON from the XAnnotation format by EBM to get bounding boxes
    and their corresponding labels.

    Args:
        spatial_size (Tuple[int, int]): the spatial size (width, height) of the image that the bounding boxes should be normalized to.

    Returns:
        Tuple[List[List[float]], List[str]]: a tuple containing a list of bounding boxes and a list of labels.
    """
    def __init__(self, spatial_size: Tuple[int, int]=None):
        self.spatial_size = spatial_size # W, H

    def __call__(self, json_obj: Dict) -> Tuple[List[List[float]], List[str]]:
        """
        Parses the JSON object and returns a tuple containing a list of bounding boxes and a list of labels.

        Args:
            json_obj (Dict): a JSON object in the XAnnotation format by EBM.

        Returns:
            A tuple containing:
            - boxes: a list of lists where each inner list contains the coordinates of a bounding box
                in the format [x_min, y_min, x_max, y_max], normalized with self.spatial_size
            - labels: a list of strings containing the corresponding labels for each bounding box
        """
        boxes = []
        labels = []

        size = json_obj["size"]  # [W, H]

        shapes = json_obj["shapes"]
        for shape in shapes:
            label = shape.get("strokeColor")
            x_min, y_min = shape["a"]
            x_max, y_max = shape["b"]
            if self.spatial_size is not None:
                x_min = x_min / size[0] * self.spatial_size[0]
                y_min = y_min / size[1] * self.spatial_size[1]
                x_max = x_max / size[0] * self.spatial_size[0]
                y_max = y_max / size[1] * self.spatial_size[1]
            box = [x_min, y_min, x_max, y_max]
            boxes.append(box)
            labels.append(label)

        return boxes, labels

class ParseXAnnotationDetectionLabeld(MapTransform):
    """
    A transform that parses the JSON annotations from the XAnnotation tool by EBM to get detection labels.

    Args:
        keys (KeysCollection): keys to pick data for transformation.
        box_keys (KeysCollection): keys to store the bounding boxes.
        label_keys (KeysCollection): keys to store the labels.
        spatial_size (Tuple[int, int]): the spatial size (width, height) of the image that the bounding boxes should be normalized to.
        allow_missing_keys (bool): whether to allow missing keys.

    Returns:
        Dict[Hashable, Any]: a dictionary containing the transformed data.
    """
    def __init__(self,
        keys: KeysCollection,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        spatial_size: Tuple[int, int]=None,
        allow_missing_keys: bool=False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

        self.box_keys = ensure_tuple_rep(box_keys, len(self.keys))
        self.label_keys = ensure_tuple_rep(label_keys, len(self.keys))

        if not len(self.keys)==len(self.label_keys) == len(self.box_keys):
            raise ValueError("Please make sure len(self.keys)==len(label_keys)==len(box_keys)!")

        self.t = ParseXAnnotationDetectionLabel(spatial_size=spatial_size)

    def __call__(self, data: Mapping[Hashable, Dict]) -> Dict[Hashable, Any]:
        """
        Args:
            data: A list of labels, each label is a list bounding boxes.
            
        Returns:
            Dict[Hashable, Any]: transformed data.
        """
        d = dict(data)
        for key, box_key, label_key in self.key_iterator(d, self.box_keys, self.label_keys):
            d[box_key], d[label_key] = self.t(d[key])
        return d
