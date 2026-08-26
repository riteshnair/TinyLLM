#include "lm/lm.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <vector>

struct lm_vulkan_f32_context {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkShaderModule shader;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkBuffer buffers[3];
    VkDeviceMemory memories[3];
    VkDeviceSize capacities[3];
    uint32_t queue_family;
};

namespace {

uint32_t memory_type(VkPhysicalDevice physical, uint32_t bits, VkMemoryPropertyFlags wanted) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0u; i < properties.memoryTypeCount; ++i)
        if ((bits & (1u << i)) != 0u && (properties.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    return UINT32_MAX;
}

bool read_spv(const char *path, std::vector<uint32_t> *code) {
    if (!path || !code) return false;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamoff end = file.tellg();
    if (end <= 0 || end % 4 != 0) return false;
    code->resize(static_cast<size_t>(end) / 4u);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(code->data()), end);
    return static_cast<std::streamoff>(file.gcount()) == end;
}

void destroy_buffer(lm_vulkan_f32_context *context, uint32_t index) {
    if (context->buffers[index] != VK_NULL_HANDLE) vkDestroyBuffer(context->device, context->buffers[index], nullptr);
    if (context->memories[index] != VK_NULL_HANDLE) vkFreeMemory(context->device, context->memories[index], nullptr);
    context->buffers[index] = VK_NULL_HANDLE;
    context->memories[index] = VK_NULL_HANDLE;
    context->capacities[index] = 0u;
}

void cleanup(lm_vulkan_f32_context *context) {
    if (!context) return;
    if (context->device != VK_NULL_HANDLE) vkDeviceWaitIdle(context->device);
    if (context->device != VK_NULL_HANDLE)
        for (uint32_t i = 0u; i < 3u; ++i) destroy_buffer(context, i);
    if (context->device != VK_NULL_HANDLE && context->command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(context->device, context->command_pool, nullptr);
    if (context->device != VK_NULL_HANDLE && context->descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(context->device, context->descriptor_pool, nullptr);
    if (context->device != VK_NULL_HANDLE && context->pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->pipeline, nullptr);
    if (context->device != VK_NULL_HANDLE && context->pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(context->device, context->pipeline_layout, nullptr);
    if (context->device != VK_NULL_HANDLE && context->set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(context->device, context->set_layout, nullptr);
    if (context->device != VK_NULL_HANDLE && context->shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->shader, nullptr);
    if (context->device != VK_NULL_HANDLE) vkDestroyDevice(context->device, nullptr);
    if (context->instance != VK_NULL_HANDLE) vkDestroyInstance(context->instance, nullptr);
    delete context;
}

lm_status ensure_buffer(lm_vulkan_f32_context *context, uint32_t index, VkDeviceSize bytes) {
    if (context->capacities[index] >= bytes) return LM_OK;
    destroy_buffer(context, index);
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(context->device, &buffer_info, nullptr, &context->buffers[index]) != VK_SUCCESS)
        return LM_ERR_CAPACITY;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context->device, context->buffers[index], &requirements);
    const uint32_t type = memory_type(context->physical, requirements.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) return LM_ERR_UNSUPPORTED;
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    if (vkAllocateMemory(context->device, &allocation, nullptr, &context->memories[index]) != VK_SUCCESS)
        return LM_ERR_CAPACITY;
    if (vkBindBufferMemory(context->device, context->buffers[index], context->memories[index], 0u) != VK_SUCCESS)
        return LM_ERR_STATE;
    context->capacities[index] = bytes;
    return LM_OK;
}

} // namespace

lm_status lm_vulkan_f32_context_create(const char *spv_path, uint32_t device_index,
                                       lm_vulkan_f32_context **out_context) {
    if (!spv_path || !out_context) return LM_ERR_ARGUMENT;
    *out_context = nullptr;
    std::vector<uint32_t> code;
    if (!read_spv(spv_path, &code)) return LM_ERR_IO;
    lm_vulkan_f32_context *context = new (std::nothrow) lm_vulkan_f32_context{};
    if (!context) return LM_ERR_CAPACITY;
    context->queue_family = UINT32_MAX;
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
    if (vkCreateInstance(&instance_info, nullptr, &context->instance) != VK_SUCCESS) { cleanup(context); return LM_ERR_UNSUPPORTED; }
    uint32_t count = 0u;
    if (vkEnumeratePhysicalDevices(context->instance, &count, nullptr) != VK_SUCCESS || device_index >= count) {
        cleanup(context);
        return device_index >= count ? LM_ERR_RANGE : LM_ERR_UNSUPPORTED;
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(context->instance, &count, devices.data()) != VK_SUCCESS) { cleanup(context); return LM_ERR_UNSUPPORTED; }
    context->physical = devices[device_index];
    uint32_t queue_count = 0u;
    vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_properties(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &queue_count, queue_properties.data());
    for (uint32_t i = 0u; i < queue_count; ++i)
        if (queue_properties[i].queueCount != 0u && (queue_properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
            context->queue_family = i;
            break;
        }
    if (context->queue_family == UINT32_MAX) { cleanup(context); return LM_ERR_UNSUPPORTED; }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = context->queue_family;
    queue_info.queueCount = 1u;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1u;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(context->physical, &device_info, nullptr, &context->device) != VK_SUCCESS) { cleanup(context); return LM_ERR_UNSUPPORTED; }
    vkGetDeviceQueue(context->device, context->queue_family, 0u, &context->queue);
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = code.size() * sizeof(uint32_t);
    shader_info.pCode = code.data();
    if (vkCreateShaderModule(context->device, &shader_info, nullptr, &context->shader) != VK_SUCCESS) { cleanup(context); return LM_ERR_UNSUPPORTED; }
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
    if (vkCreateDescriptorSetLayout(context->device, &layout_info, nullptr, &context->set_layout) != VK_SUCCESS) { cleanup(context); return LM_ERR_UNSUPPORTED; }
    struct Push { uint32_t rows; uint32_t columns; };
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(Push);
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1u;
    pipeline_layout_info.pSetLayouts = &context->set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1u;
    pipeline_layout_info.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(context->device, &pipeline_layout_info, nullptr, &context->pipeline_layout) != VK_SUCCESS) { cleanup(context); return LM_ERR_UNSUPPORTED; }
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = context->shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = context->pipeline_layout;
    if (vkCreateComputePipelines(context->device, VK_NULL_HANDLE, 1u, &pipeline_info, nullptr, &context->pipeline) != VK_SUCCESS) { cleanup(context); return LM_ERR_UNSUPPORTED; }
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3u;
    VkDescriptorPoolCreateInfo descriptor_pool_info{};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 1u;
    descriptor_pool_info.poolSizeCount = 1u;
    descriptor_pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(context->device, &descriptor_pool_info, nullptr, &context->descriptor_pool) != VK_SUCCESS) { cleanup(context); return LM_ERR_CAPACITY; }
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = context->descriptor_pool;
    set_info.descriptorSetCount = 1u;
    set_info.pSetLayouts = &context->set_layout;
    if (vkAllocateDescriptorSets(context->device, &set_info, &context->descriptor_set) != VK_SUCCESS) { cleanup(context); return LM_ERR_CAPACITY; }
    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = context->queue_family;
    if (vkCreateCommandPool(context->device, &command_pool_info, nullptr, &context->command_pool) != VK_SUCCESS) { cleanup(context); return LM_ERR_CAPACITY; }
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = context->command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1u;
    if (vkAllocateCommandBuffers(context->device, &command_info, &context->command_buffer) != VK_SUCCESS) { cleanup(context); return LM_ERR_CAPACITY; }
    *out_context = context;
    return LM_OK;
}

void lm_vulkan_f32_context_destroy(lm_vulkan_f32_context *context) { cleanup(context); }

lm_status lm_vulkan_f32_context_matvec(lm_vulkan_f32_context *context,
                                       const float *matrix, uint32_t rows,
                                       uint32_t columns, const float *input,
                                       float *out) {
    if (!context || context->device == VK_NULL_HANDLE || !matrix || !input || !out || rows == 0u || columns == 0u) return LM_ERR_ARGUMENT;
    const uint64_t elements = static_cast<uint64_t>(rows) * columns;
    if (elements > std::numeric_limits<uint64_t>::max() / sizeof(float) ||
        elements * sizeof(float) > static_cast<uint64_t>(std::numeric_limits<VkDeviceSize>::max())) return LM_ERR_CAPACITY;
    const VkDeviceSize matrix_bytes = static_cast<VkDeviceSize>(elements * sizeof(float));
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(static_cast<uint64_t>(columns) * sizeof(float));
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(static_cast<uint64_t>(rows) * sizeof(float));
    for (uint32_t i = 0u; i < 3u; ++i) {
        const lm_status status = ensure_buffer(context, i, i == 0u ? matrix_bytes : (i == 1u ? input_bytes : output_bytes));
        if (status != LM_OK) return status;
    }
    void *mapped = nullptr;
    if (vkMapMemory(context->device, context->memories[0], 0u, matrix_bytes, 0u, &mapped) != VK_SUCCESS) return LM_ERR_STATE;
    std::memcpy(mapped, matrix, static_cast<size_t>(matrix_bytes));
    vkUnmapMemory(context->device, context->memories[0]);
    if (vkMapMemory(context->device, context->memories[1], 0u, input_bytes, 0u, &mapped) != VK_SUCCESS) return LM_ERR_STATE;
    std::memcpy(mapped, input, static_cast<size_t>(input_bytes));
    vkUnmapMemory(context->device, context->memories[1]);
    if (vkMapMemory(context->device, context->memories[2], 0u, output_bytes, 0u, &mapped) != VK_SUCCESS) return LM_ERR_STATE;
    std::memset(mapped, 0, static_cast<size_t>(output_bytes));
    vkUnmapMemory(context->device, context->memories[2]);
    VkDescriptorBufferInfo infos[3] = {};
    VkWriteDescriptorSet writes[3] = {};
    const VkDeviceSize sizes[3] = {matrix_bytes, input_bytes, output_bytes};
    for (uint32_t i = 0u; i < 3u; ++i) {
        infos[i].buffer = context->buffers[i];
        infos[i].range = sizes[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = context->descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(context->device, 3u, writes, 0u, nullptr);
    if (vkResetCommandBuffer(context->command_buffer, 0u) != VK_SUCCESS) return LM_ERR_STATE;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(context->command_buffer, &begin) != VK_SUCCESS) return LM_ERR_STATE;
    struct Push { uint32_t rows; uint32_t columns; } push{rows, columns};
    vkCmdBindPipeline(context->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline);
    vkCmdBindDescriptorSets(context->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline_layout, 0u, 1u,
                            &context->descriptor_set, 0u, nullptr);
    vkCmdPushConstants(context->command_buffer, context->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                       sizeof(push), &push);
    vkCmdDispatch(context->command_buffer, rows, 1u, 1u);
    if (vkEndCommandBuffer(context->command_buffer) != VK_SUCCESS) return LM_ERR_STATE;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1u;
    submit.pCommandBuffers = &context->command_buffer;
    if (vkQueueSubmit(context->queue, 1u, &submit, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(context->queue) != VK_SUCCESS) return LM_ERR_STATE;
    if (vkMapMemory(context->device, context->memories[2], 0u, output_bytes, 0u, &mapped) != VK_SUCCESS) return LM_ERR_STATE;
    std::memcpy(out, mapped, static_cast<size_t>(output_bytes));
    vkUnmapMemory(context->device, context->memories[2]);
    for (uint32_t row = 0u; row < rows; ++row) if (!std::isfinite(out[row])) return LM_ERR_RANGE;
    return LM_OK;
}
