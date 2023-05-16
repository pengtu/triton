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
        #include <level_zero/ze_api.h>

        #define PY_SSIZE_T_CLEAN
        #include <Python.h>

        static inline void gpuAssert(ze_result_t code, const char *file, int line)
        {
           if (code != ZE_RESULT_SUCCESS)
           {
              const char* prefix = "Triton Error [ZE]: ";
              const char* str = to_string(code).str();
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

            if (device_id > devices.size())
                return NULL;

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
            //    std::cout << to_string( pMemoryProperties[ mem ] ) << "\n";
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
            const char* data;
            Py_ssize_t data_size;
            int shared;
            int device_id;
            if(!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared, &device_id)) {
                return NULL;
            }

            if (device_id > devices.size()) {
                return NULL;
            }
            ze_device_handle_t device = devices[device_id];

            int32_t n_regs = 0;
            int32_t n_spills = 0;

            ze_module_desc_t module_desc = {};
            module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
            module_desc.inputSize = data_size;
            module_desc.pInputModule = data;
            ze_module_handle_t mod;
            ZE_CHECK(zeModuleCreate(context, device, &module_desc, &module, nullptr));

            ze_kernel_desc_t kernel_desc = {};
            kernel_desc.pKernelName = name;
            ze_kernel_handle_t fun;
            ZE_CHECK(zeKernelCreate(module, &kernel_desc, &fun));

            if(PyErr_Occurred()) {
              return NULL;
            }
            return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs, n_spills);
        }

        static PyObject* initContext(PyObject* self) {
            // Initialize driver
            ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
            uint32_t driverCount = 0;
            ZE_CHECK(zeDriverGet(&driverCount, nullptr));

            // Retrieve driver
            ze_driver_handle_t driverHandle;
            ZE_CHECK(zeDriverGet(&driverCount, &driverHandle));

            // Create context
            ze_context_desc_t contextDesc = {};
            static ze_context_handle_t context;
            ZE_CHECK(zeContextCreate(driverHandle, &contextDesc, &context));

            return Py_BuildValue(\"(K)", (uint64_t)context);
            // Py_RETURN_NONE;
        }

        static PyObject* initDevices(PyObject* self) {
            // Retrieve devices
            uint32_t deviceCount = 0;
            ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, nullptr));
            static std::vector<ze_device_handle_t> devices(deviceCount);
            ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, devices.data()));

            // Default immediate command list of each device
            static std::vector<ze_command_list_handle_t> queues(deviceCount);

            return Py_BuildValue(\"(O)", PyArray_SimpleNewFromData(1, &devices.size(), NPY_UINT64, devices.data()));
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
                return Py_BuildValue(\"(K)", (uint64_t)queue);
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
                computeQUeueGroupOrdinal,
                0, // index
                0, // flags
                ZE_COMMAND_QUEUE_MODE_DEFAULT,
                ZE_COMMAND_QUEUE_PRIORITY_NORMAL
            };

            ZE_CHECK(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &queue));
            return Py_BuildValue(\"(K)", (uint64_t)queue);
        }

        static PyMethodDef ModuleMethods[] = {
          {"load_binary", loadBinary, METH_VARARGS, "Load provided SPV into ZE driver"},
          {"get_device_properties", getDeviceProperties, METH_VARARGS, "Get the properties for a given device"},
          {"init_context", initContext, METH_VARARGS, "Initialize the ZE GPU context"},
          {"init_devices", initDevices, METH_VARARGS, "Initialize the ZE GPU devices"},
          {"get_queue", getQueue, METH_VARARGS, "Get immediate command list for device_id"},
          {NULL, NULL, 0, NULL} // sentinel
        };

        static struct PyModuleDef ModuleDef = {
          PyModuleDef_HEAD_INIT,
          \"spirv_utils\",
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
                src_path = os.path.join(tmpdir, "main.c")
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
        self.context = mod.init_context()
        self.devices = mod.init_devices()
        self.get_queue = mod.get_queue()
        self.current_device = 0 if self.devices.size > 0 else -1

    def get_current_device(instance):
        return instance.current_device
    
    def set_current_device(instance, idx):
        assert instance.devices.size > idx, "Device id not found"
        instance.current_device = idx
    
    def get_device_capability(instance, idx):
        return (0, 0)
    
    def get_queue(instance, idx):
        return instance.get_queue(idx)
        
