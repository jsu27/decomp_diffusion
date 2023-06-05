"""
Helpers for distributed training.
"""

import io
import os
import socket
import subprocess

import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist


def setup_dist(backend="nccl"):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    dist.init_process_group(backend=backend, init_method="env://")

    if th.cuda.is_available():  # This clears remaining caches in GPU 0
        th.cuda.set_device(dev())
        th.cuda.empty_cache()


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        if 'LOCAL_RANK' in os.environ:
            return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        else:
            return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            # Create a copy of the parameter tensor to avoid broadcasting failures
            p_copy = p.detach().clone()
            dist.broadcast(p_copy, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    SLURM_VARIABLES = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_TASK_PID",
    ]

    PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
    for name in SLURM_VARIABLES:
        value = os.environ.get(name, None)
        print(PREFIX + "%s: %s" % (name, str(value)))

    # number of nodes / node ID
    params.nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    params.node_rank = int(os.environ["SLURM_NODEID"])

    # define master address and master port
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    params.master_addr = hostnames.split()[0].decode("utf-8")
    print("master address ", params.master_addr)