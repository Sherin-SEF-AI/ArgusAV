"""Minimal TensorRT runner for the Jetson: no CUDA toolkit or pycuda needed.

Uses the CUDA driver library (libcuda, part of the Jetson driver) through ctypes and
unified memory (cuMemAllocManaged), so input and output tensors are plain numpy arrays
the GPU reads and writes directly - on Orin CPU and GPU share DRAM, so nothing is copied.
"""
from __future__ import annotations

import ctypes
import threading

import numpy as np

from .errors import RigError

_lib = None
_lock = threading.Lock()
# Tegra has no concurrent managed access: the CPU must not touch ANY unified-memory buffer
# of this process while a kernel runs. All inference goes through infer(), which holds
# this lock from writing the inputs to copying the outputs (a segfault otherwise).
GPU_LOCK = threading.Lock()
_ctx = ctypes.c_void_p()


def _cu():
    global _lib
    with _lock:
        if _lib is None:
            try:
                lib = ctypes.CDLL("libcuda.so.1")
            except OSError as e:
                raise RigError(f"CUDA driver not available: {e}") from None
            _check(lib.cuInit(0), "cuInit")
            dev = ctypes.c_int()
            _check(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
            _check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(_ctx), dev), "cuDevicePrimaryCtxRetain")
            lib.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint]
            lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
            lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
            _lib = lib
    _check(_lib.cuCtxSetCurrent(_ctx), "cuCtxSetCurrent")      # per calling thread
    return _lib


def _check(rc: int, what: str):
    if rc != 0:
        raise RigError(f"CUDA {what} failed (error {rc})")


def _allocator(trt):
    class Alloc(trt.IOutputAllocator):
        """Outputs whose size depends on the data (e.g. after NMS): one generous unified
        buffer per tensor, reused while it is big enough; TensorRT reports the shape."""

        def __init__(self, cap: int):
            trt.IOutputAllocator.__init__(self)
            self.cap, self.ptr, self.shape = cap, ctypes.c_uint64(), None
            _check(_cu().cuMemAllocManaged(ctypes.byref(self.ptr), cap, 1), "cuMemAllocManaged")

        def reallocate_output(self, name, memory, size, alignment):
            if size > self.cap:
                raise RigError(f"output {name} needs {size} bytes (> {self.cap})")
            return self.ptr.value

        def reallocate_output_async(self, name, memory, size, alignment, stream):
            return self.reallocate_output(name, memory, size, alignment)

        def notify_shape(self, name, shape):
            self.shape = tuple(shape)
    return Alloc


class Engine:
    """A serialized TensorRT engine with its I/O tensors as numpy arrays (unified memory).
    Data-dependent outputs (shape with -1) are returned with their actual shape."""

    DYN_ROWS = 4096

    def __init__(self, path: str, shapes: dict | None = None):
        """shapes: {input name: shape} for dynamic inputs (default: the profile maximum)."""
        import tensorrt as trt
        cu = _cu()
        self._trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RigError(f"cannot load TensorRT engine {path}",
                           "rebuild it for this TensorRT version: avrig models --build")
        self.ctx = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        _check(cu.cuStreamCreate(ctypes.byref(self.stream), 0), "cuStreamCreate")
        self.inputs, self.outputs, self._ptrs, self._dyn = {}, {}, [], {}
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        is_in = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in names}
        for name in names:                   # dynamic inputs (a batch range): the profile's maximum
            if is_in[name] and any(d < 0 for d in self.engine.get_tensor_shape(name)):
                want = (shapes or {}).get(name) or tuple(self.engine.get_tensor_profile_shape(name, 0)[2])
                self.ctx.set_input_shape(name, tuple(want))
        for name in sorted(names, key=lambda n: not is_in[n]):
            shape = tuple(self.ctx.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            if any(d < 0 for d in shape):
                row = int(np.prod([d for d in shape if d > 0])) * dtype.itemsize
                a = _allocator(trt)(self.DYN_ROWS * row)
                self.ctx.set_output_allocator(name, a)
                self._dyn[name] = (a, dtype)
                continue
            nbytes = int(np.prod(shape)) * dtype.itemsize
            ptr = ctypes.c_uint64()
            _check(cu.cuMemAllocManaged(ctypes.byref(ptr), max(nbytes, 16), 1), "cuMemAllocManaged")
            self._ptrs.append(ptr.value)
            buf = (ctypes.c_char * nbytes).from_address(ptr.value)
            arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
            self.ctx.set_tensor_address(name, ptr.value)
            (self.inputs if is_in[name] else self.outputs)[name] = arr

    def infer(self, **inputs) -> dict:
        """Copy inputs in, run, copy outputs out - under GPU_LOCK. Inputs: name ->
        array (broadcast into the input tensor) or a callable(input_array) that fills it."""
        with GPU_LOCK:
            for name, v in inputs.items():
                if callable(v):
                    v(self.inputs[name])
                else:
                    self.inputs[name][...] = v
            out = self.run()
            return {k: np.array(v, copy=True) for k, v in out.items()}

    def run(self) -> dict:
        """Execute with the current inputs. Callers that write inputs / read outputs
        themselves must hold GPU_LOCK; prefer infer()."""
        cu = _cu()
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RigError("TensorRT inference failed")
        _check(cu.cuStreamSynchronize(self.stream), "cuStreamSynchronize")
        for name, (a, dtype) in self._dyn.items():
            shape = a.shape or (0,)
            n = int(np.prod(shape))
            if n == 0:
                self.outputs[name] = np.zeros(shape, dtype)
                continue
            buf = (ctypes.c_char * (n * dtype.itemsize)).from_address(a.ptr.value)
            self.outputs[name] = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return self.outputs

    def close(self):
        if _lib and (self._ptrs or self._dyn):
            for p in self._ptrs + [a.ptr.value for a, _ in self._dyn.values()]:
                _lib.cuMemFree_v2(p)
            self._ptrs, self._dyn = [], {}
            _lib.cuStreamDestroy_v2(self.stream)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
