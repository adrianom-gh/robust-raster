from dask.distributed import Client, LocalCluster
import dask.array as da
import dask.dataframe as dd
import xarray as xr
from dask_plugins import EEPlugin
import multiprocessing
import psutil
import json
import ee
import os

class DaskClusterManager:
    def __init__(self, dask_client: Client = None) -> None:
        '''
        Initialize the DaskHandler class. Creates a dask_client attribute that will
        be used to store the Dask Client information.
        '''
        self.dask_client = dask_client

    @property
    def get_dask_client(self):
        return self.dask_client
    
    def _bytes_to_gigabytes(self, memory: int) -> int:
        '''
        Private method that takes the system memory in bytes and converts it to gigabytes.

        Parameters:
        - memory (int): The total system memory of the machine in bytes. 

        Returns:
        - int: The system memory converted to gigabytes
        '''

        gigabytes = memory / (1024 ** 3)
        return gigabytes
    
    def create_local_threads(self) -> None:
        '''
        Store dask_client attribute with a Dask Client object configured to use the local threads
        on the machine.
        '''
        self.dask_client = Client(processes=False)

    def create_test_cluster(self, **kwargs) -> None:
        '''
        Store dask_client attribute with a Dask Client object configured to generate a LocalCluster object. 
        LocalCluster configuration is autogenerated by taking the number of CPU cores in the given machine, 
        creating one worker per core (with each worker getting one thread), and allocating memory to each worker 
        based on the system memory divided by the number of cores (or number of workers).

        User can override these default settings by providing n_workers, threads_per_worker, and memory_limit 
        through kwargs.
        '''
        num_cores = multiprocessing.cpu_count()
        total_memory = psutil.virtual_memory().total
        total_memory_gb = self._bytes_to_gigabytes(total_memory)

        # Default values
        default_n_workers = 1
        default_threads_per_worker = 1
        default_memory_limit = f"{int(total_memory_gb)}GB"

        # Use kwargs values if provided, otherwise use default values
        n_workers = kwargs.get('n_workers', default_n_workers)
        threads_per_worker = kwargs.get('threads_per_worker', default_threads_per_worker)
        memory_limit = kwargs.get('memory_limit', default_memory_limit)

        self.dask_client = Client(LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit))

    def create_local_cluster(self, **kwargs) -> None:
        '''
        Store dask_client attribute with a Dask Client object configured to generate a LocalCluster object. 
        LocalCluster configuration is autogenerated by taking the number of CPU cores in the given machine, 
        creating one worker per core (with each worker getting one thread), and allocating memory to each worker 
        based on the system memory divided by the number of cores (or number of workers).

        User can override these default settings by providing n_workers, threads_per_worker, and memory_limit 
        through kwargs.
        '''
        num_cores = multiprocessing.cpu_count()
        total_memory = psutil.virtual_memory().total
        total_memory_gb = self._bytes_to_gigabytes(total_memory)

        # Default values
        default_n_workers = num_cores
        default_threads_per_worker = 1
        default_memory_limit = f"{total_memory_gb / num_cores}GB"

        # Use kwargs values if provided, otherwise use default values
        n_workers = kwargs.get('n_workers', default_n_workers)
        threads_per_worker = kwargs.get('threads_per_worker', default_threads_per_worker)
        memory_limit = kwargs.get('memory_limit', default_memory_limit)

        self.dask_client = Client(LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit))

    def connect_to_cloud_cluster(self, scheduler_address: str):
        '''
        Store dask_client attribute with a Dask Client object configured (WIP) to connect to a Cloud service.
        '''
        self.dask_client = Client(scheduler_address)

    def process_with_dask(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
        '''
        If the dask_client attribute is not None, then chunk the dataset to convert it to a Dask array.

        Parameters:
        - dataset (xr.Dataset): An xarray Dataset object that will be chunked.
        
        Returns:
        - xr.Dataset: An xarray dataset object now configured to be a Dask array.
        '''

        # So the payload size in Earth Engine says its 10MB, but xee found through trial and error 48 MBs.
        # When using ee.data.computePixels (which xee using in the backend), it sends a request object. 
        # This object will also contain the chunk size. To compute the size of the chunk, you can multiple 
        # each dimension and then multiply by the dtype size (if the pixels are float64, then 8 bytes). 
        # This, including the other aspects of the request object (filtering by date, cloud mask, etc.) 
        # would add up to your total payload size. To compute the bytes say filter by date takes up, you 
        # add up the characters, including white space, and multiply it by 1 byte (assuming the characters
        # are UTF-8 encoded).
        default_chunks  = {
            'time': 48,
            'X': 512,
            'Y': 256
        }

        # Extract chunk sizes from kwargs if provided
        chunk_sizes = kwargs.pop('chunks', default_chunks)

        if self.dask_client:
            return dataset.chunk(chunk_sizes, **kwargs)
        return dataset

    def initialize_ee_on_workers(self, json_key=None):
        ee_plugin = EEPlugin(json_key)
        #self.dask_client.register_plugin(ee_plugin)