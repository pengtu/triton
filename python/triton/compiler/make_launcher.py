import hashlib
import os
import tempfile

from ..common import _build
from ..runtime.cache import get_cache_manager
from ..runtime.jit import version_key


def is_hip():
    import torch
    return torch.version.hip is not None

# FIX ME
def is_spirv():
    return os.environ.get("TRITON_TARGET_SPIRV", "0") == "1"

# ----- stub --------

def make_so_cache_key(version_hash, signature, constants):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}{constants}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key

def make_stub(name, signature, constants):
    # name of files that are cached
    so_cache_key = make_so_cache_key(version_key(), signature, constants)
    so_cache_manager = get_cache_manager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher(constants, signature)
            src_path = os.path.join(tmpdir, "main.cpp" if is_spirv() else "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, "rb") as f:
                return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path

# ----- source code generation --------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*" if is_spirv() else "hipDeviceptr_t" if is_hip() else "CUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]

def generate_launcher(constants, signature):
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "void*" if is_spirv() else "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def format_of(ty):
        return {
            "PyObject*": "O",
            "void*" : "K",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    format = "iiiiiKKOOOK" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    if is_spirv():
        src = f"""
    #include <cstddef>
    #include <string>
    #include <iostream>
    #include <iomanip>
    #include <level_zero/ze_api.h>

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <Python.h>
    #include <stdio.h>
    #include <numpy/arrayobject.h>

    static inline void gpuAssert(ze_result_t code, const char *file, int line)
    {{
      if (code != ZE_RESULT_SUCCESS)
      {{
         const char* prefix = "Triton Error [ZE]: ";
         const char* str = std::to_string(code).c_str();
         char err[1024] = {{0}};
         strcat(err, prefix);
         strcat(err, str);
         PyErr_SetString(PyExc_RuntimeError, err);
      }}
    }}

    #define ZE_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

    static void _launch(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_warps, int shared_memory, ze_command_list_handle_t queue, ze_kernel_handle_t function, ze_event_pool_handle_t event_pool, {arg_decls}) {{
      void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};

      if (gridX*gridY*gridZ > 0) {{
        {" ".join(f'zeKernelSetArgumentValue(function, {idx}, sizeof({ty_to_cpp(item)}), params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}
        zeKernelSetGroupSize(function, 32*num_warps, 1, 1);
        ze_group_count_t grpCount = {{gridX, gridY, gridZ}};
        std::cout << "Num_warps is " << num_warps << std::endl;

        ze_event_desc_t eventDesc = {{
            ZE_STRUCTURE_TYPE_EVENT_DESC,
            nullptr,
            0, 
            0, 
            ZE_EVENT_SCOPE_FLAG_HOST 
        }};
        ze_event_handle_t hEvent;
        ZE_CHECK(zeEventCreate(event_pool, &eventDesc, &hEvent));

        // Append a signal of an event into the command list after the kernel executes
        ZE_CHECK(zeCommandListAppendLaunchKernel(queue, function, &grpCount, hEvent, 0, nullptr));
        // Wait on event to complete
        ZE_CHECK(zeEventHostSynchronize(hEvent, std::numeric_limits<uint64_t>::max()));
      }}
    }}

    typedef struct _DevicePtrInfo {{
      void* dev_ptr;
      bool valid;
    }} DevicePtrInfo;

    static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
      DevicePtrInfo ptr_info;
      ptr_info.dev_ptr = 0;
      ptr_info.valid = true;
      std::cout << "!!!Inside getPointer, obj ptr is" << std::endl;
      std::cout << std::hex << obj << std::endl;
      PyTypeObject* obj_type = Py_TYPE(obj);
      std::cout << "Object Type is " << obj_type->tp_name << std::endl;

      if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = (void*) PyLong_AsUnsignedLongLong(obj);
        std::cout << "Passed PyLong_Check and return" << std::endl;
        return ptr_info;
      }}
      if (obj == Py_None) {{
        // valid nullptr
        return ptr_info;
      }}
      PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
      if(ptr){{
        std::cout << "!!!data_ptr is not NULL" << std::endl;
        std::cout << std::hex << ptr << std::endl;
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);
        if (!PyLong_Check(ret)) {{
          PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
          ptr_info.valid = false;
          return ptr_info;
        }}
        ptr_info.dev_ptr = (void*) PyLong_AsUnsignedLongLong(ret);
        if(!ptr_info.dev_ptr) {{
          return ptr_info;        
        }}
        Py_DECREF(ret);  // Thanks ChatGPT!
        std::cout << "!!!dev_ptr" << std::endl;
        std::cout << std::hex << ptr_info.dev_ptr << std::endl;        
        return ptr_info;
      }}
      std::cout << "!!!data_ptr is NULL" << std::endl;
      PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
      return ptr_info;
    }}

    static PyObject* launch(PyObject* self, PyObject* args) {{

      int gridX, gridY, gridZ;
      uint64_t _stream;
      uint64_t _function;
      uint64_t _event_pool;
      int num_warps;
      int shared_memory;
      PyObject *launch_enter_hook = NULL;
      PyObject *launch_exit_hook = NULL;
      PyObject *compiled_kernel = NULL;
      //PyObject *event_pool_handle = NULL;
      std::cout<< "!!!inside launch" << std::endl;
      
      {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if (!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, &_event_pool, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
        return NULL;
      }}
#if 0
      if (PyLong_Check(event_pool_handle)) {{
        _event_pool = PyLong_AsUnsignedLongLong(event_pool_handle);
      }} else {{
        PyTypeObject* obj_type = Py_TYPE(event_pool_handle);
        std::cout << "Object Type is " << obj_type->tp_name << std::endl;
        std::cout << "!!!Event pool handle invalid value" << std::endl;
        return NULL;
      }}
#endif

      if (launch_enter_hook != Py_None) {{
        PyObject_CallObject(launch_enter_hook, args);
      }}

      std::cout<< "!!!before calling _launch" << std::endl;
      // raise exception asap
      // {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
      // std::cout << "!!!done _launch parameter setup" << std::endl;
      _launch(gridX, gridY, gridZ, num_warps, shared_memory, (ze_command_list_handle_t)_stream, (ze_kernel_handle_t)_function, (ze_event_pool_handle_t)_event_pool, {', '.join(f"(void *) _arg{i}" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});
      
      if (launch_exit_hook != Py_None) {{
        PyObject_CallObject(launch_exit_hook, args);
      }}
      if (PyErr_Occurred()) {{
        return NULL;
      }}

      // return None
      Py_INCREF(Py_None);
      return Py_None;
    }}

    static PyMethodDef ModuleMethods[] = {{
      {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
      {{NULL, NULL, 0, NULL}} // sentinel
    }};

    static struct PyModuleDef ModuleDef = {{
      PyModuleDef_HEAD_INIT,
      \"__triton_launcher\",
      NULL, //documentation
      -1, //size
      ModuleMethods
    }};

    PyMODINIT_FUNC PyInit___triton_launcher(void) {{
      PyObject *m = PyModule_Create(&ModuleDef);
      if(m == NULL) {{
        return NULL;
      }}
      PyModule_AddFunctions(m, ModuleMethods);
      return m;
    }}
    """
    elif is_hip():
        src = f"""
    #define __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include <Python.h>
    #include <stdio.h>

    static inline void gpuAssert(hipError_t code, const char *file, int line)
    {{
      if (code != HIP_SUCCESS)
      {{
         const char* prefix = "Triton Error [HIP]: ";
         const char* str = hipGetErrorString(code);
         char err[1024] = {{0}};
         snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
         PyErr_SetString(PyExc_RuntimeError, err);
      }}
    }}

    #define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

    static void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, hipStream_t stream, hipFunction_t function, {arg_decls}) {{
      void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
      if (gridX*gridY*gridZ > 0) {{
          HIP_CHECK(hipModuleLaunchKernel(function, gridX, gridY, gridZ, 64*num_warps, 1, 1, shared_memory, stream, params, 0));
      }}
    }}

    typedef struct _DevicePtrInfo {{
      hipDeviceptr_t dev_ptr;
      bool valid;
    }} DevicePtrInfo;

    static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
      DevicePtrInfo ptr_info;
      ptr_info.dev_ptr = 0;
      ptr_info.valid = true;

      if (PyLong_Check(obj)) {{   
        ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
        return ptr_info;
      }}

      if (obj == Py_None) {{
        // valid nullptr
        return ptr_info;
      }}

      PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");

      if (ptr) {{
        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);

        if (!PyLong_Check(ret)) {{
          PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
          ptr_info.valid = false;
          return ptr_info;
        }}

        ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);

        if (!ptr_info.dev_ptr)
          return ptr_info;

        uint64_t dev_ptr;
        hipError_t status = hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
        if (status == hipErrorInvalidValue) {{
            PyErr_Format(PyExc_ValueError,
                         "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
            ptr_info.valid = false;
        }}

        ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
        return ptr_info;
      }}

      PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
      return ptr_info;
    }}

    static PyObject* launch(PyObject* self, PyObject* args) {{

      int gridX, gridY, gridZ;
      uint64_t _stream;
      uint64_t _function;
      int num_warps;
      int shared_memory;
      PyObject *launch_enter_hook = NULL;
      PyObject *launch_exit_hook = NULL;
      PyObject *compiled_kernel = NULL;

      {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
      if (!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
        return NULL;
      }}

      if (launch_enter_hook != Py_None) {{
        PyObject_CallObject(launch_enter_hook, args);
      }}

      // raise exception asap
      {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
      _launch(gridX, gridY, gridZ, num_warps, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items())});
      if (launch_exit_hook != Py_None) {{
        PyObject_CallObject(launch_exit_hook, args);
      }}
      if (PyErr_Occurred()) {{
        return NULL;
      }}

      // return None
      Py_INCREF(Py_None);
      return Py_None;
    }}

    static PyMethodDef ModuleMethods[] = {{
      {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
      {{NULL, NULL, 0, NULL}} // sentinel
    }};

    static struct PyModuleDef ModuleDef = {{
      PyModuleDef_HEAD_INIT,
      \"__triton_launcher\",
      NULL, //documentation
      -1, //size
      ModuleMethods
    }};

    PyMODINIT_FUNC PyInit___triton_launcher(void) {{
      PyObject *m = PyModule_Create(&ModuleDef);
      if(m == NULL) {{
        return NULL;
      }}
      PyModule_AddFunctions(m, ModuleMethods);
      return m;
    }}
    """
    else:
        src = f"""
#include \"cuda.h\"
#include <stdbool.h>
#include <Python.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, CUstream stream, CUfunction function, {arg_decls}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
  if(gridX*gridY*gridZ > 0){{
    CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject_CallObject(launch_enter_hook, args);
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (launch_exit_hook != Py_None) {{
    PyObject_CallObject(launch_exit_hook, args);
  }}

  if(PyErr_Occurred()) {{
    return NULL;
  }}
  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    print(src)
    return src
