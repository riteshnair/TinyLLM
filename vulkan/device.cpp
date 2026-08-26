#include "lm/lm.h"

#include <vulkan/vulkan.h>

#include <cstring>

static VkInstance make_instance() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "tiny-lm";
    app.applicationVersion = 1u;
    app.pEngineName = "tiny-lm";
    app.engineVersion = 1u;
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    return vkCreateInstance(&info, nullptr, &instance) == VK_SUCCESS ? instance : VK_NULL_HANDLE;
}

static uint8_t has_extension(VkPhysicalDevice device, const char *wanted) {
    uint32_t count = 0u;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS) return 0u;
    VkExtensionProperties *extensions = new VkExtensionProperties[count];
    const VkResult result = vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions);
    uint8_t found = 0u;
    if (result == VK_SUCCESS) {
        for (uint32_t i = 0u; i < count; ++i) {
            if (std::strcmp(extensions[i].extensionName, wanted) == 0) { found = 1u; break; }
        }
    }
    delete[] extensions;
    return found;
}

lm_status lm_vulkan_device_count(uint32_t *out_count) {
    if (!out_count) return LM_ERR_ARGUMENT;
    *out_count = 0u;
    VkInstance instance = make_instance();
    if (instance == VK_NULL_HANDLE) return LM_ERR_UNSUPPORTED;
    const VkResult result = vkEnumeratePhysicalDevices(instance, out_count, nullptr);
    if (result != VK_SUCCESS) *out_count = 0u;
    vkDestroyInstance(instance, nullptr);
    return result == VK_SUCCESS ? LM_OK : LM_ERR_UNSUPPORTED;
}

lm_status lm_vulkan_device_info_get(uint32_t index, lm_vulkan_device_info *out_info) {
    if (!out_info) return LM_ERR_ARGUMENT;
    std::memset(out_info, 0, sizeof(*out_info));
    VkInstance instance = make_instance();
    if (instance == VK_NULL_HANDLE) return LM_ERR_UNSUPPORTED;
    uint32_t count = 0u;
    VkResult result = vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (result != VK_SUCCESS || index >= count) {
        vkDestroyInstance(instance, nullptr);
        return result == VK_SUCCESS ? LM_ERR_RANGE : LM_ERR_UNSUPPORTED;
    }
    VkPhysicalDevice *devices = new VkPhysicalDevice[count];
    result = vkEnumeratePhysicalDevices(instance, &count, devices);
    if (result != VK_SUCCESS) {
        delete[] devices;
        vkDestroyInstance(instance, nullptr);
        return LM_ERR_UNSUPPORTED;
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(devices[index], &properties);
    std::strncpy(out_info->name, properties.deviceName, sizeof(out_info->name) - 1u);
    out_info->api_version = properties.apiVersion;
    out_info->driver_version = properties.driverVersion;
    out_info->vendor_id = properties.vendorID;
    out_info->device_id = properties.deviceID;
    out_info->is_cpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? 1u : 0u;
    out_info->shader_int_dot = has_extension(devices[index], VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    out_info->subgroup = properties.apiVersion >= VK_API_VERSION_1_1 ? 1u : 0u;
    delete[] devices;
    vkDestroyInstance(instance, nullptr);
    return LM_OK;
}
