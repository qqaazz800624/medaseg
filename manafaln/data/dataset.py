import os
import sys
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pickle import dumps, loads
from typing import Any, Callable, List, Literal, Optional, Sequence
from zlib import compress, decompress

from monai.data import Dataset
from monai.data.utils import pickle_hashing
from monai.transforms import Compose, Transform, RandomizableTrait
from tqdm import tqdm


class DataItemSerializer:
    @staticmethod
    def serialize(data):
        return deepcopy(data)

    @staticmethod
    def deserialize(data):
        return deepcopy(data)

class PickleSerializer(DataItemSerializer):
    @staticmethod
    def serialize(data):
        return dumps(data)

    @staticmethod
    def deserialize(data):
        return loads(data)

class CompressedSerializer(DataItemSerializer):
    @staticmethod
    def serialize(data):
        return compress(dumps(data))

    @staticmethod
    def deserialize(data):
        return loads(decompress(data))

class DataItemStorage:
    __slots__ = ["data"]

    def __init__(self):
        self.data = None

    def save(self, data: Any) -> None:
        self.data = data

    def load(self) -> Any:
        return self.data

    def exists(self) -> bool:
        return False if self.data is None else True

class MemoryStorage(DataItemStorage):
    def __init__(self):
        super().__init__()

class FileSystemStorage(DataItemStorage):
    __slots__ = ["dir", "filename"]

    def __init__(self, dir: os.PathLike, filename: str):
        super().__init__()

        self.dir = Path(dir)
        self.filename = self.dir / filename

    def save(self, data: Any) -> None:
        with open(self.filename, "wb") as f:
            f.write(data)

    def load(self) -> Any:
        with open(self.filename, "rb") as f:
            data = f.read()
        return data

    def exists(self) -> bool:
        return Path(self.filename).exists()

class DataItem(object):
    __slots__ = ["transform_index", "serializer", "storage"]

    def __init__(
        self,
        data: Any,
        transform: Optional[Compose] = None,
        serializer: type(DataItemSerializer) = DataItemSerializer,
        storage: DataItemStorage = DataItemStorage()
    ):
        self.serializer = serializer
        self.storage = storage

        if self.storage.exists():
            self.transform_index = self.get_random_index(transform)
        else:
            self.set_data(data, transform)

    def get_random_index(self, transform: Optional[Compose] = None) -> int:
        if transform is None:
            return 0
        random_index = transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        if random_index is None:
            return 0
        return random_index

    def set_data(self, data: Any, transform: Optional[Compose] = None) -> None:
        self.transform_index = self.get_random_index(transform)
        if transform is not None:
            data = transform(data, end=self.transform_index, threading=True)
        data = self.serializer.serialize(data)
        self.storage.save(data)

    def get_data(self, transform: Optional[Compose] = None) -> Any:
        data = self.storage.load()
        data = self.serializer.deserialize(data)
        if transform is not None:
            data = transform(data, start=self.transform_index)
        return data

class HybridCacheDataset(Dataset):
    def __init__(
        self,
        data: Sequence,
        transform: Optional[Callable] = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        cache_mode: Literal["memory", "hybrid", "persistent"] = "memory",
        cache_dir: Optional[str] = None,
        num_workers: Optional[int] = 1,
        progress: bool = True,
        compress_cache: bool = True
    ):
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)

        cache_rate = min(1.0, max(cache_rate, 0.0))
        self.cache_num = min(cache_num, int(len(self.data) * cache_rate))

        self.cache_mode = cache_mode
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is None:
            if self.cache_mode != "memory":
                raise ValueError("`cache_dir` must be set if using hybrid or persistent mode")
        elif not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        elif not self.cache_dir.is_dir():
            raise ValueError("`cache_dir` must be a directory.")

        self.num_workers = num_workers
        self.progress = progress

        if compress_cache:
            self.default_serializer = CompressedSerializer
        else:
            self.default_serializer = PickleSerializer

        self.setup(self.data)

    def create_data_item(self, index: int) -> Any:
        item = None
        if index < self.cache_num:
            if self.cache_mode != "persistent":
                # Cache on memory
                item = DataItem(
                    data=self.data[index],
                    transform=self.transform,
                    serializer=self.default_serializer,
                    storage=MemoryStorage()
                )
            else:
                # Cache on disk
                item = DataItem(
                    data=self.data[index],
                    transform=self.transform,
                    serializer=self.default_serializer,
                    storage=FileSystemStorage(
                        dir=self.cache_dir,
                        filename=pickle_hashing(self.data[index]).decode("utf-8")
                    )
                )
        else:
            if self.cache_mode != "memory":
                # Cache on disk
                item = DataItem(
                    data=self.data[index],
                    transform=self.transform,
                    serializer=self.default_serializer,
                    storage=FileSystemStorage(
                        dir=self.cache_dir,
                        filename=pickle_hashing(self.data[index]).decode("utf-8")
                    )
                )
            else:
                # No cache
                item = DataItem(
                    data=self.data[index],
                    transform=None,
                    serializer=DataItemSerializer,
                    storage=DataItemStorage()
                )
        return item

    def setup(self, data: Sequence) -> List[Any]:
        self.data = data
        indices = list(range(len(self.data)))

        if self.num_workers == 1:
            if self.progress:
                self.cache = []
                for i in tqdm(indices, desc="Loading dataset"):
                    self.cache.append(self.create_data_item(i))
            else:
                self.cache = [self.create_data_item(i) for i in indices]
            return self.cache

        if self.progress:
            with ThreadPool(self.num_workers) as p:
                self.cache = list(
                    tqdm(
                        p.imap(self.create_data_item, indices),
                        total=len(data),
                        desc="Loading dataset"
                    )
                )
        else:
            with ThreadPool(self.num_workers) as p:
                self.cache = list(p.imap(self.create_data_item, indices))
        return self.cache

    def _transform(self, index: int):
        return self.cache[index].get_data(self.transform)

