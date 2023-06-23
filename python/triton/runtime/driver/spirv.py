import hashlib
import os
import tempfile
import numpy as np

from ...common.build import _build
from ..cache import get_cache_manager


def get_spirv_utils():
    global _spirv_utils
    if _spirv_utils is None:
        _spirv_utils = SpirvUtils()
    return _spirv_utils


_spirv_utils = None


class SpirvUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SpirvUtils, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def _generate_src():
        return """
        #include <cstddef>
        #include <string>
        #include <vector>
        #include <iostream>
        #include <level_zero/ze_api.h>

        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <Python.h>
        #include <numpy/arrayobject.h>

        static ze_context_handle_t context = {nullptr};
        static ze_driver_handle_t driverHandle = {nullptr};
        static ze_event_pool_handle_t eventPoolHandle = {nullptr};
        
        static std::vector<ze_device_handle_t> devices;
        // Default immediate command list of each device
        static std::vector<ze_command_list_handle_t> queues;

        static inline void gpuAssert(ze_result_t code, const char *file, int line)
        {
           if (code != ZE_RESULT_SUCCESS)
           {
              const char* prefix = "Triton Error [ZE]: ";
              const char* str = std::to_string(code).c_str();
              char err[1024] = {0};
              strcat(err, prefix);
              strcat(err, str);
              PyErr_SetString(PyExc_RuntimeError, err);
           }
        }

        #define ZE_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); if(PyErr_Occurred()) return NULL; }

        static PyObject* getDeviceProperties(PyObject* self, PyObject* args){
            int device_id;
            if(!PyArg_ParseTuple(args, "i", &device_id))
                return NULL;

            if (device_id > devices.size()) {
                std::cout << "Device ID not found: " << device_id << std::endl;
                return NULL;
            }

            // Get device handle
            ze_device_handle_t phDevice = devices[device_id];

            // create a struct to hold device properties
            ze_device_properties_t device_properties = {};
            device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            zeDeviceGetProperties(phDevice, &device_properties);

            int multiprocessor_count = device_properties.numSlices * device_properties.numSubslicesPerSlice;
            int sm_clock_rate = device_properties.coreClockRate;

            ze_device_compute_properties_t compute_properties = {};
            compute_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
            zeDeviceGetComputeProperties(phDevice, &compute_properties);
            int max_shared_mem = compute_properties.maxSharedLocalMemory;

            uint32_t memoryCount = 0;
            zeDeviceGetMemoryProperties(phDevice, &memoryCount, nullptr);
            auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
            for( uint32_t mem = 0; mem < memoryCount; ++mem )
            {
                pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
                pMemoryProperties[mem].pNext = nullptr;
            }
            zeDeviceGetMemoryProperties(phDevice, &memoryCount, pMemoryProperties);
            // for( uint32_t mem = 0; mem < memoryCount; ++mem )
            // {
            //    std::cout << to_string( pMemoryProperties[ mem ] ) << std::endl;
            // }

            int mem_clock_rate = pMemoryProperties[0].maxClockRate;
            int mem_bus_width = pMemoryProperties[0].maxBusWidth;

            delete[] pMemoryProperties;

            return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", max_shared_mem,
                                       "multiprocessor_count", multiprocessor_count,
                                       "sm_clock_rate", sm_clock_rate,
                                       "mem_clock_rate", mem_clock_rate,
                                       "mem_bus_width", mem_bus_width);
        }

        static PyObject* loadBinary(PyObject* self, PyObject* args) {
            const char* name;
            int shared;
            PyObject *py_bytes;
            int device_id;
            if(!PyArg_ParseTuple(args, "sSii", &name, &py_bytes, &shared, &device_id)) {
                std::cout << "loadBinary arg parse failed" << std::endl;
                return NULL;
            }

            // uint8_t* data = (uint8_t*) PyBytes_AsString(py_bytes);
            // int data_size = PyBytes_Size(py_bytes);

            if (device_id > devices.size()) {
                std::cout << "Device ID not found: " << device_id << std::endl;
                return NULL;
            }

            ze_device_handle_t device = devices[device_id];

            int32_t n_regs = 0;
            int32_t n_spills = 0;

            ze_module_desc_t module_desc = {};
            module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
            module_desc.inputSize = PyBytes_Size(py_bytes);
            module_desc.pInputModule = (uint8_t*) PyBytes_AsString(py_bytes);
            ze_module_handle_t module;
            // std::cout << "SPIRV binary size: " << module_desc.inputSize << std::endl;
            ZE_CHECK(zeModuleCreate(context, device, &module_desc, &module, nullptr));

            // std::cout << "loadBinary zeModuleCreated" << std::endl;
            ze_kernel_desc_t kernel_desc = {};
            kernel_desc.pKernelName = name;
            ze_kernel_handle_t fun;
            ZE_CHECK(zeKernelCreate(module, &kernel_desc, &fun));

            // std::cout << "loadBinary zeKernelCreated" << std::endl;

            if(PyErr_Occurred()) {
              std::cout << "loadBinary error occurred" << std::endl;
              return NULL;
            }

            std::cout << "loadBinary success!" << std::endl;
            return Py_BuildValue("(KKii)", (uint64_t)module, (uint64_t)fun, n_regs, n_spills);
        }

        static PyObject* initContext(PyObject* self, PyObject* args) {
            // Initialize driver
            ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
            uint32_t driverCount = 0;
            ZE_CHECK(zeDriverGet(&driverCount, nullptr));

            // Retrieve driver
            ZE_CHECK(zeDriverGet(&driverCount, &driverHandle));

            // Create context
            ze_context_desc_t contextDesc = {};
            ZE_CHECK(zeContextCreate(driverHandle, &contextDesc, &context));

            return Py_BuildValue("(K)", (uint64_t)context);
            // Py_RETURN_NONE;
        }

        static PyObject* initEventPool(PyObject* self, PyObject* args) {
            // Create event pool
            ze_event_pool_desc_t tsEventPoolDesc = {
                ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
                nullptr,
                ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
                1 // count
            };
            ZE_CHECK(zeEventPoolCreate(context, &tsEventPoolDesc, 0, nullptr, &eventPoolHandle));

            return Py_BuildValue("(K)", (uint64_t)eventPoolHandle);
            // Py_RETURN_NONE;
        }

        static PyObject* initDevices(PyObject* self, PyObject *args) {
            // Retrieve devices
            uint32_t deviceCount = 0;
            ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, nullptr));
            // std::cout << "Device count is: " << deviceCount << std::endl;
            for (uint32_t i = 0; i < deviceCount; ++i) {
                devices.push_back(nullptr);
                queues.push_back(nullptr);
            }
            ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, devices.data()));
            
            // npy_intp dims[1];
            // dims[0] = deviceCount;
            // std::cout << "Before PyArray_SimpleNewFromData: " << devices.size() << " " << devices.data()[0] << std::endl;
            // PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, reinterpret_cast<void*>(devices.data()));
            // std::cout << "After PyArray_SimpleNewFromData: " << devices.data()[0] << std::endl;
            // PyObject* ret = Py_BuildValue("(O)", arr);
            // std::cout << "After Py_BuildValue" << std::endl;
            // return ret;
            return Py_BuildValue("(i)", deviceCount);
            // Py_RETURN_NONE;
        }

        static PyObject* getQueue(PyObject* self, PyObject* args) {
            int device_id;
            if(!PyArg_ParseTuple(args, "i", &device_id))
                return NULL;

            if (device_id > devices.size())
                return NULL;

            // Get default queue
            ze_command_list_handle_t queue = queues[device_id];
            if (queue != NULL) {
                return Py_BuildValue("(K)", (uint64_t)queue);
            }

            // Create immediate command list
            ze_device_handle_t device = devices[device_id];
            uint32_t numQueueGroups = 0;
            ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr));
            if (numQueueGroups == 0) {
                return NULL;
            }
            std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
            ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups,
                                                            queueProperties.data()));
            uint32_t computeQueueGroupOrdinal = numQueueGroups;

            // Find the compute queue ordinal
            for (uint32_t i = 0; i < numQueueGroups; i++) {
                if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                    computeQueueGroupOrdinal = i;
                    break;
                }
            }
            if (computeQueueGroupOrdinal == numQueueGroups) {
                return NULL; // no compute queue found
            }

            ze_command_queue_desc_t cmdQueueDesc = {
                ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                nullptr,
                computeQueueGroupOrdinal,
                0, // index
                0, // flags
                ZE_COMMAND_QUEUE_MODE_DEFAULT,
                ZE_COMMAND_QUEUE_PRIORITY_NORMAL
            };

            ZE_CHECK(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &queue));
            return Py_BuildValue("(K)", (uint64_t)queue);
        }

        static PyMethodDef ModuleMethods[] = {
          {"load_binary", loadBinary, METH_VARARGS, "Load provided SPV into ZE driver"},
          {"get_device_properties", getDeviceProperties, METH_VARARGS, "Get the properties for a given device"},
          {"init_context", initContext, METH_NOARGS, "Initialize the ZE GPU context"},
          {"init_devices", initDevices, METH_NOARGS, "Initialize the ZE GPU devices and return device count"},
          {"get_queue", getQueue, METH_VARARGS, "Get immediate command list for device_id"},
          {"init_event_pool", initEventPool, METH_VARARGS, "Initialize ZE event pool"},
          {NULL, NULL, 0, NULL} // sentinel
        };

        static struct PyModuleDef ModuleDef = {
          PyModuleDef_HEAD_INIT,
          "spirv_utils",
          NULL, //documentation
          -1, //size
          ModuleMethods
        };

        PyMODINIT_FUNC PyInit_spirv_utils(void) {
          PyObject *m = PyModule_Create(&ModuleDef);
          if(m == NULL) {
            return NULL;
          }
          PyModule_AddFunctions(m, ModuleMethods);
          return m;
        }
        """

    def __init__(self):
        src = self._generate_src()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "spirv_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build("spirv_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("spirv_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.get_queue = mod.get_queue
        self.context = mod.init_context()
        self.device_count = mod.init_devices()
        self.event_pool = mod.init_event_pool()[0]
        self.current_device = 0 if self.device_count[0] > 0 else -1

    def get_current_device(instance):
        return instance.current_device
    
    def get_event_pool(instance):
        return instance.event_pool
    
    def set_current_device(instance, idx):
        assert instance.device_count[0] > idx, "Device id not found"
        instance.current_device = idx
    
    def get_device_capability(instance, idx):
        return (0, 0)
    
    def get_queue(instance, idx):
        return instance.get_queue(idx)[0]
        
