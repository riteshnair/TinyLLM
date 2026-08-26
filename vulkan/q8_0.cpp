#include "lm/lm.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {

uint32_t find_memory_type(VkPhysicalDevice device, uint32_t bits, VkMemoryPropertyFlags wanted) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(device, &props);
    for (uint32_t i = 0u; i < props.memoryTypeCount; ++i)
        if ((bits & (1u << i)) != 0u && (props.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    return UINT32_MAX;
}

bool read_spv(const char *path, std::vector<uint32_t> *out) {
    if (!path || !out) return false;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamoff end = file.tellg();
    if (end <= 0 || (end % 4) != 0) return false;
    out->resize(static_cast<size_t>(end) / 4u);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(out->data()), end);
    return static_cast<std::streamoff>(file.gcount()) == end;
}

} // namespace

lm_status lm_vulkan_dot_q8_0(const char *spv_path, uint32_t device_index,
                             const void *packed_q8_0, uint32_t blocks,
                             const float *input, float *out_result) {
    if (!spv_path || !packed_q8_0 || !input || !out_result || blocks == 0u) return LM_ERR_ARGUMENT;
    if (blocks > std::numeric_limits<uint64_t>::max() / 34u ||
        blocks > std::numeric_limits<uint64_t>::max() / 32u)
        return LM_ERR_RANGE;
    const uint64_t packed_bytes_64 = static_cast<uint64_t>(blocks) * 34u;
    const uint64_t input_bytes_64 = static_cast<uint64_t>(blocks) * 32u * sizeof(float);
    if (packed_bytes_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        input_bytes_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        input_bytes_64 > static_cast<uint64_t>(std::numeric_limits<VkDeviceSize>::max()))
        return LM_ERR_CAPACITY;
    const size_t packed_bytes = static_cast<size_t>(packed_bytes_64);
    const size_t input_bytes = static_cast<size_t>(input_bytes_64);
    std::vector<uint32_t> code;
    if (!read_spv(spv_path, &code)) return LM_ERR_IO;

    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkBuffer buffers[3] = {};
    VkDeviceMemory memories[3] = {};

    const auto cleanup = [&]() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        for (uint32_t i = 0u; i < 3u; ++i) {
            if (buffers[i] != VK_NULL_HANDLE) vkDestroyBuffer(device, buffers[i], nullptr);
            if (memories[i] != VK_NULL_HANDLE) vkFreeMemory(device, memories[i], nullptr);
        }
        if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
        if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        if (shader != VK_NULL_HANDLE) vkDestroyShaderModule(device, shader, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    };
    const auto fail = [&](lm_status status) { cleanup(); return status; };

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "tiny-lm";
    app.applicationVersion = 1u;
    app.pEngineName = "tiny-lm";
    app.engineVersion = 1u;
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) return LM_ERR_UNSUPPORTED;

    uint32_t device_count = 0u;
    if (vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != VK_SUCCESS || device_index >= device_count)
        return fail(device_index >= device_count ? LM_ERR_RANGE : LM_ERR_UNSUPPORTED);
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(instance, &device_count, devices.data()) != VK_SUCCESS) return fail(LM_ERR_UNSUPPORTED);
    const VkPhysicalDevice physical = devices[device_index];
    uint32_t queue_count = 0u;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_props(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, queue_props.data());
    uint32_t queue_family = UINT32_MAX;
    for (uint32_t i = 0u; i < queue_count; ++i) {
        if (queue_props[i].queueCount != 0u && (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
            queue_family = i;
            break;
        }
    }
    if (queue_family == UINT32_MAX) return fail(LM_ERR_UNSUPPORTED);
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1u;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1u;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) return fail(LM_ERR_UNSUPPORTED);
    vkGetDeviceQueue(device, queue_family, 0u, &queue);

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = code.size() * sizeof(uint32_t);
    shader_info.pCode = code.data();
    if (vkCreateShaderModule(device, &shader_info, nullptr, &shader) != VK_SUCCESS) return fail(LM_ERR_UNSUPPORTED);
    VkDescriptorSetLayoutBinding bindings[3] = {};
    for (uint32_t i = 0u; i < 3u; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1u;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 3u;
    layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout) != VK_SUCCESS) return fail(LM_ERR_UNSUPPORTED);
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1u;
    pipeline_layout_info.pSetLayouts = &set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1u;
    pipeline_layout_info.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) return fail(LM_ERR_UNSUPPORTED);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo compute_info{};
    compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_info.stage = stage;
    compute_info.layout = pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1u, &compute_info, nullptr, &pipeline) != VK_SUCCESS) return fail(LM_ERR_UNSUPPORTED);

    const VkDeviceSize sizes[3] = {static_cast<VkDeviceSize>(packed_bytes), static_cast<VkDeviceSize>(input_bytes), sizeof(float)};
    for (uint32_t i = 0u; i < 3u; ++i) {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = sizes[i];
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(device, &buffer_info, nullptr, &buffers[i]) != VK_SUCCESS) return fail(LM_ERR_CAPACITY);
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffers[i], &requirements);
        const uint32_t type = find_memory_type(physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX) return fail(LM_ERR_UNSUPPORTED);
        VkMemoryAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.allocationSize = requirements.size;
        allocate_info.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &allocate_info, nullptr, &memories[i]) != VK_SUCCESS) return fail(LM_ERR_CAPACITY);
        if (vkBindBufferMemory(device, buffers[i], memories[i], 0u) != VK_SUCCESS) return fail(LM_ERR_STATE);
    }
    void *mapped = nullptr;
    if (vkMapMemory(device, memories[0], 0u, sizes[0], 0u, &mapped) != VK_SUCCESS) return fail(LM_ERR_STATE);
    std::memcpy(mapped, packed_q8_0, packed_bytes);
    vkUnmapMemory(device, memories[0]);
    if (vkMapMemory(device, memories[1], 0u, sizes[1], 0u, &mapped) != VK_SUCCESS) return fail(LM_ERR_STATE);
    std::memcpy(mapped, input, input_bytes);
    vkUnmapMemory(device, memories[1]);
    if (vkMapMemory(device, memories[2], 0u, sizeof(float), 0u, &mapped) != VK_SUCCESS) return fail(LM_ERR_STATE);
    std::memset(mapped, 0, sizeof(float));
    vkUnmapMemory(device, memories[2]);

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3u;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1u;
    pool_info.poolSizeCount = 1u;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) return fail(LM_ERR_CAPACITY);
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = descriptor_pool;
    set_info.descriptorSetCount = 1u;
    set_info.pSetLayouts = &set_layout;
    if (vkAllocateDescriptorSets(device, &set_info, &descriptor_set) != VK_SUCCESS) return fail(LM_ERR_CAPACITY);
    VkDescriptorBufferInfo buffer_infos[3] = {};
    VkWriteDescriptorSet writes[3] = {};
    for (uint32_t i = 0u; i < 3u; ++i) {
        buffer_infos[i].buffer = buffers[i];
        buffer_infos[i].range = sizes[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device, 3u, writes, 0u, nullptr);
    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = queue_family;
    if (vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS) return fail(LM_ERR_CAPACITY);
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1u;
    if (vkAllocateCommandBuffers(device, &command_info, &command_buffer) != VK_SUCCESS) return fail(LM_ERR_CAPACITY);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command_buffer, &begin) != VK_SUCCESS) return fail(LM_ERR_STATE);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0u, 1u, &descriptor_set, 0u, nullptr);
    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(blocks), &blocks);
    vkCmdDispatch(command_buffer, 1u, 1u, 1u);
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) return fail(LM_ERR_STATE);
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1u;
    submit.pCommandBuffers = &command_buffer;
    if (vkQueueSubmit(queue, 1u, &submit, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(queue) != VK_SUCCESS) return fail(LM_ERR_STATE);
    if (vkMapMemory(device, memories[2], 0u, sizeof(float), 0u, &mapped) != VK_SUCCESS) return fail(LM_ERR_STATE);
    std::memcpy(out_result, mapped, sizeof(float));
    vkUnmapMemory(device, memories[2]);
    cleanup();
    return std::isfinite(*out_result) ? LM_OK : LM_ERR_RANGE;
}
