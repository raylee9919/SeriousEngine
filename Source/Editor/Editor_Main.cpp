// Copyright Seong Woo Lee. All Rights Reserved.

#include "ThirdParty/stb/Include/stb_image.h"

#include "Core/SE_Basics.h"
#include "Core/SE_String.h"
#include "Core/SE_Math.h"
#include "Window/Window.h"
#include "File/FileSystem.h"
#include "Asset/SE_Asset.h"
#include "RHI/RHI.h"
#include "Shader/DXIL/DXIL_Compiler.h"
#include "Renderer/Renderer.h"
#include "Scene/Entity.h"

#include "Renderer/Pass/Pass_GBuffer.h"
#include "Renderer/Pass/Pass_Defer.h"
#include "Renderer/Pass/Pass_Cloud.h"
#include "Renderer/Pass/Pass_Blit.h"
#include "Renderer/Pass/Pass_Composition.h"

#include "Editor.h"
#include "Editor_UI.h"

using namespace Engine;
using namespace DXIL;


struct GPU_Arena {
    DX12_Resource* resource;
};


// Sync with shader!
struct Shader_Camera {
    vec4 position;
    m4x4 view;
    m4x4 proj;
    m4x4 view_proj;
    m4x4 inv_view;
    m4x4 inv_proj;
};

struct Camera {
    vec3    position;
    f32     aspect_ratio;
    f32     fov;
    f32     near_z;
    f32     far_z;
    f32     yaw;
    f32     pitch;
    f32     speed;
    f32     last_mouse_x;
    f32     last_mouse_y;
    m4x4    view;
    m4x4    proj;
    m4x4    view_proj;

    void update(f32 dt, Input_System* input)
    {
        vec3 forward = vec3(0.f, 0.f, 1.f);
        vec4 f = y_rotation(yaw) * x_rotation(pitch) * vec4(forward, 0.f);
        forward = normalize(f.xyz);
        vec3 right = cross(forward, vec3(0.f, 1.f, 0.f));
        vec3 up = vec3(0.f, 1.f, 0.f);

        f32 move_speed = speed;
        if (input->key_is_down[KEY_LEFT_SHIFT]) {
            move_speed *= 0.333333f;
        }

        f32 factor = dt * move_speed;

        if (input->key_is_down[KEY_E]) { position += ( up      * factor ); }
        if (input->key_is_down[KEY_Q]) { position -= ( up      * factor ); }
        if (input->key_is_down[KEY_W]) { position += ( forward * factor ); }
        if (input->key_is_down[KEY_S]) { position -= ( forward * factor ); }
        if (input->key_is_down[KEY_D]) { position += ( right   * factor ); }
        if (input->key_is_down[KEY_A]) { position -= ( right   * factor ); }

        if (input->mouse_is_down[MOUSE_LEFT]) {
            if (!input->mouse_was_down[MOUSE_LEFT]) {
                last_mouse_x = input->current_mouse_x;
                last_mouse_y = input->current_mouse_y;
            } else {
                f32 dx = input->current_mouse_x - last_mouse_x;
                f32 dy = input->current_mouse_y - last_mouse_y;

                f32 rot_speed = 0.25f;
                yaw   -= (rot_speed * dx * dt);
                pitch += (rot_speed * dy * dt);
                yaw   = Engine::fmod(yaw, PI * 2.f);
                pitch = Clamp(pitch, -PI * 0.45f, PI * 0.45f);

                last_mouse_x = input->current_mouse_x;
                last_mouse_y = input->current_mouse_y;
            }
        }

        view      = look_to_rh(position, forward, vec3(0.f, 1.f, 0.f));
        proj      = perspective_rh(fov, aspect_ratio, near_z, far_z);
        view_proj = proj * view;
    }
};

Shader_Camera get_shader_camera(Camera* camera) {
    Shader_Camera result = {
        .position  = vec4(camera->position, 1.0f),
        .view      = camera->view,
        .proj      = camera->proj,
        .view_proj = camera->view_proj,
        .inv_view  = inverse(camera->view),
        .inv_proj  = inverse(camera->proj),
    };
    return result;
}

INTERNAL Mesh_Resource
alloc_mesh_resource(DX12_Device* device, DX12_Command_Queue* cmd_queue, DX12_Command_List* cmd_list, DX12_Fence* fence, DX12_Descriptor_Heap* srv_heap, Mesh* mesh)
{
    Mesh_Resource result = {};
    result.name = mesh->name;

    for (u32 slice_index = 0; slice_index < mesh->meshes.size(); ++slice_index) {
        auto* slice = mesh->meshes[slice_index];
        Mesh_Slice_Resource slice_res = {};

        slice_res.material_name = slice->material_name;

        {
            u32 count  = (u32)slice->vertices.size();
            u32 stride = sizeof(slice->vertices[0]);
            u64 size   = count * stride;

            auto* vertex_buffer = dx12_alloc_resource(device, { .type = DX12_RESOURCE_TYPE_BUFFER, .heap_type = D3D12_HEAP_TYPE_DEFAULT, .buffer = { .size = size } });
            dx12_upload_buffer(device, cmd_queue, cmd_list, fence, vertex_buffer, slice->vertices.data(), size);

            DX12_Descriptor vertex_buffer_descriptor = dx12_alloc_descriptor(srv_heap);
            dx12_create_srv(device, vertex_buffer, &vertex_buffer_descriptor, DXGI_FORMAT_UNKNOWN, count, stride);

            slice_res.vertex_buffer = vertex_buffer;
            slice_res.vertex_buffer_descriptor = vertex_buffer_descriptor;
            slice_res.num_vertices = count;
        }

        {
            u32 count  = (u32)slice->indices.size();
            u32 stride = sizeof(slice->indices[0]);
            u64 size   = count * stride;

            auto* index_buffer = dx12_alloc_resource(device, { .type = DX12_RESOURCE_TYPE_BUFFER, .heap_type = D3D12_HEAP_TYPE_DEFAULT, .buffer = { .size = size } });
            dx12_upload_buffer(device, cmd_queue, cmd_list, fence, index_buffer, slice->indices.data(), size);

            slice_res.index_buffer.resource = index_buffer;
            slice_res.index_buffer.num_indices  = count;
        }

        result.slices.push_back(slice_res);
    }

    return result;
}


struct Bitmap {
    u32 width;
    u32 height;
    u32 pitch;
    u8* data;
};

INTERNAL Array <Bitmap> load_image(const String& path, bool generate_mipmap)
{
    Array <Bitmap> result = {};

    int x, y, forced_channels = 4;
    u8* image = stbi_load(path.c_str(), &x, &y, nullptr, forced_channels);
    CORE_ASSERT(image);

    {
        Bitmap bitmap = {};
        {
            bitmap.width  = (u32)x;
            bitmap.height = (u32)y;
            bitmap.pitch  = bitmap.width * forced_channels;

            u64 sz = bitmap.pitch * bitmap.height;
            bitmap.data = new u8 [sz];
            memcpy(bitmap.data, image, sz);
        }

        result.push_back(bitmap);
    }

    // @Todo: Better mipmap generation. SIMD or use compute shader.
    // Take sRGB textures into account.
    // Something better than box filter. JBlow's articles my be handy.
    //
    if (generate_mipmap) {
        u16 levels = (u16)floor(log2(max(x, y))) + 1;
        for (u16 i = 0; i < levels - 1; ++i) {
            Bitmap* src = &result[i];
            Bitmap dst = {};

            dst.width  = max(1u, src->width  >> 1);
            dst.height = max(1u, src->height >> 1);
            dst.pitch  = dst.width * forced_channels;
            dst.data   = new u8 [dst.pitch * dst.height];

            for (u32 row = 0; row < dst.height; ++row) {
                for (u32 col = 0; col < dst.width; ++col) {
                    // Map dst texel to 4 src texels (2x2 box filter)
                    u32 sx = col * 2;
                    u32 sy = row * 2;

                    // Clamp in case src dimension was odd
                    u32 sx1 = min(sx + 1, src->width  - 1);
                    u32 sy1 = min(sy + 1, src->height - 1);

                    u8* s00 = src->data + sy  * src->pitch + sx  * forced_channels;
                    u8* s10 = src->data + sy  * src->pitch + sx1 * forced_channels;
                    u8* s01 = src->data + sy1 * src->pitch + sx  * forced_channels;
                    u8* s11 = src->data + sy1 * src->pitch + sx1 * forced_channels;

                    u8* d = dst.data + row * dst.pitch + col * forced_channels;

                    for (int c = 0; c < forced_channels; ++c) {
                        d[c] = (u8)((s00[c] + s10[c] + s01[c] + s11[c] + 2) >> 2);
                    }
                }
            }

            result.push_back(dst);
        }
    }

    stbi_image_free(image);

    return result;
}

INTERNAL std::pair <DX12_Resource*, u32> alloc_resource_and_srv_and_return_id(DX12_Device* device, DX12_Command_Queue* cmd_queue, DX12_Command_List* cmd_list, DX12_Fence* fence, DX12_Descriptor_Heap* srv_heap, const nlohmann::json& json, const String& key, DXGI_FORMAT format, bool mip, u32 default_id)
{
    std::pair<DX12_Resource*, u32> result = { nullptr, default_id };
    DX12_Resource* tex = nullptr;
    u32 id = default_id;

    String path = json[key].get<std::string>();

    if (path != "") {
        auto bitmaps = load_image(path, mip);
        u16 mip_levels = (u16)bitmaps.size();
        u32 tex_width  = bitmaps[0].width;
        u32 tex_height = bitmaps[0].height;

        DX12_Resource_Desc desc = {
            .type = DX12_RESOURCE_TYPE_TEXTURE_2D,
            .texture = {
                .format      = format,
                .width       = tex_width,
                .height      = tex_height,
                .mip_levels  = mip_levels,
                .depth       = 1,
                .num_samples = 1,
            }
        };
        tex = dx12_alloc_resource(device, desc);

        D3D12_RESOURCE_DESC resource_desc = tex->native_resource->GetDesc();
        u32 num_subresources = resource_desc.DepthOrArraySize * resource_desc.MipLevels;
        Array <D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(num_subresources);
        Array <u32> num_rows(num_subresources);
        Array <u64> row_size_in_bytes(num_subresources);
        u64 total_bytes;

        device->native_device->GetCopyableFootprints(&resource_desc, 0, num_subresources, 0, layouts.data(), num_rows.data(), row_size_in_bytes.data(), &total_bytes);

        DX12_Resource* upload_buffer = dx12_alloc_resource(device, { .type = DX12_RESOURCE_TYPE_BUFFER, .heap_type = D3D12_HEAP_TYPE_UPLOAD, .buffer = { .size = total_bytes } });
        const D3D12_RANGE read_range = { 0, 0 };
        void* ptr;
        upload_buffer->native_resource->Map(0, &read_range, &ptr);
        {
            for (u32 array_index = 0; array_index < resource_desc.DepthOrArraySize; ++array_index) {
                for (u32 mip_index = 0; mip_index < resource_desc.MipLevels; ++mip_index) {
                    u32 index = mip_index + array_index * resource_desc.MipLevels;
                    auto layout = layouts[index];
                    u32 subresource_height  = num_rows[index];
                    u32 subresource_pitch   = align_up(layout.Footprint.RowPitch, (u32)D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
                    u32 subresource_depth   = layout.Footprint.Depth;

                    u8* dst = (u8*)ptr + layout.Offset;

                    for (u32 slice_index = 0; slice_index < subresource_depth; ++slice_index) {
                        auto& bitmap = bitmaps[mip_index];
                        u8* src = bitmap.data;
                        for (u32 height = 0; height < subresource_height; ++height) {
                            memcpy(dst, src, min(subresource_pitch, bitmap.pitch));
                            dst += subresource_pitch;
                            src += bitmap.pitch;
                        }
                    }
                }
            }
        }
        upload_buffer->native_resource->Unmap(0, nullptr);

        // schedule upload
        cmd_list->begin();
        {
            cmd_list->transition_barrier(tex, 0, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd_list->transition_barrier(upload_buffer, 0, D3D12_RESOURCE_STATE_COPY_SOURCE);

            for (u32 subresource_index = 0; subresource_index < num_subresources; ++subresource_index) {
                D3D12_TEXTURE_COPY_LOCATION dst = {
                    .pResource        = tex->native_resource,
                    .Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    .SubresourceIndex = subresource_index,
                };
                D3D12_TEXTURE_COPY_LOCATION src = {
                    .pResource        = upload_buffer->native_resource,
                    .Type             = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    .PlacedFootprint  = layouts[subresource_index]
                };
                cmd_list->native_cmd_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }

            cmd_list->transition_barrier(tex, 0, D3D12_RESOURCE_STATE_COMMON);
            cmd_list->transition_barrier(upload_buffer, 0, D3D12_RESOURCE_STATE_COMMON);
        }
        cmd_list->end();
        dx12_execute_command_list(cmd_queue, cmd_list);
        dx12_signal_fence(cmd_queue, fence);
        dx12_wait_fence(fence);
        dx12_dealloc_resource(upload_buffer);

        // allocate srv and get bindless id.
        auto srv = dx12_alloc_descriptor(srv_heap);
        dx12_create_srv(device, tex, &srv, tex->desc.texture.format);
        id = srv.index;
    }

    result.first  = tex;
    result.second = id;

    return result;
}

INTERNAL Material material_from_json(DX12_Device* device, DX12_Command_Queue* cmd_queue, DX12_Command_List* cmd_list, DX12_Fence* fence, DX12_Descriptor_Heap* srv_heap, const nlohmann::json& json)
{
    Material mat = {};

    mat.name = json["name"].get<std::string>();

    // @Todo: default material id
    // @Robustness
    auto [albedo_tex, albedo_id]     = alloc_resource_and_srv_and_return_id(device, cmd_queue, cmd_list, fence, srv_heap, json,   "albedo", DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, 0xffffffff);
    auto [orm_tex, orm_id]           = alloc_resource_and_srv_and_return_id(device, cmd_queue, cmd_list, fence, srv_heap, json,      "orm",      DXGI_FORMAT_R8G8B8A8_UNORM, true, 0xffffffff);
    auto [normal_tex, normal_id]     = alloc_resource_and_srv_and_return_id(device, cmd_queue, cmd_list, fence, srv_heap, json,   "normal",      DXGI_FORMAT_R8G8B8A8_UNORM, true, 0xffffffff);
    auto [emissive_tex, emissive_id] = alloc_resource_and_srv_and_return_id(device, cmd_queue, cmd_list, fence, srv_heap, json, "emissive", DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true, 0xffffffff);

    mat.shader_material.albedo_id   = albedo_id;
    mat.shader_material.orm_id      = orm_id;
    mat.shader_material.normal_id   = normal_id;
    mat.shader_material.emissive_id = emissive_id;

    mat.resources.push_back(albedo_tex);
    mat.resources.push_back(orm_tex);
    mat.resources.push_back(normal_tex);
    mat.resources.push_back(emissive_tex);

    { // alloc material id.
        u32 size = (u32)sizeof(mat.shader_material);
        auto* mat_res = dx12_alloc_resource(device, { .type = DX12_RESOURCE_TYPE_BUFFER, .buffer = { .size = size } });
        dx12_upload_buffer(device, cmd_queue, cmd_list, fence, mat_res, &mat.shader_material, size);
        auto srv = dx12_alloc_descriptor(srv_heap);
        dx12_create_srv(device, mat_res, &srv, mat_res->desc.texture.format, 1, size);

        mat.id       = srv.index;
        mat.resource = mat_res;
        mat.srv      = srv;
    }

    return mat;
}

void init_graphics_pipeline(DX12_Device* device, Shader_Compiler* compiler,
                            DX12_Pipeline_State* out_pso,
                            ID3D12RootSignature* root_signature,
                            DXGI_FORMAT* rtv_formats, 
                            bool has_depth, DXGI_FORMAT dsv_format,
                            D3D12_CULL_MODE cull_mode,
                            const String& shader_path)
{
    // read file.
    u64 length = read_entire_file(shader_path, nullptr);
    u8* source = new u8[length];
    read_entire_file(shader_path, source);

    // entries/profiles
    const char* vs_entry   = "vs_main";
    const char* vs_profile = "vs_6_6";
    const char* ps_entry   = "ps_main";
    const char* ps_profile = "ps_6_6";

    // compile and reflect.
    auto compiled_vs   = compiler->compile(true, source, length, shader_path, vs_entry, vs_profile);
    auto vs_reflection = compiler->reflect(compiled_vs.result);

    auto compiled_ps   = compiler->compile(true, source, length, shader_path, ps_entry, ps_profile);
    auto ps_reflection = compiler->reflect(compiled_ps.result);

#if 0
    // Generate reflected push constant header.
    Array <std::pair<String, String>> type_and_name;
    u32 num_constant_buffers = vs_reflection.shader_desc.ConstantBuffers;
    CORE_ASSERT(num_constant_buffers <= 1);

    auto* cbuf = ps_reflection.reflection->GetConstantBufferByIndex(0);

    D3D12_SHADER_BUFFER_DESC cbuf_desc;
    cbuf->GetDesc(&cbuf_desc);
    CORE_ASSERT(!strcmp(cbuf_desc.Name, "push"));

    for (u32 i = 0; i < cbuf_desc.Variables; ++i) {
        auto* var = cbuf->GetVariableByIndex(i);

        D3D12_SHADER_VARIABLE_DESC var_desc;
        var->GetDesc(&var_desc);

        auto* var_type = var->GetType();
        D3D12_SHADER_TYPE_DESC var_type_desc;
        var_type->GetDesc(&var_type_desc);

        for (u32 m = 0; m < var_type_desc.Members; ++m) {
            ID3D12ShaderReflectionType* member_type = var_type->GetMemberTypeByIndex(m);

            D3D12_SHADER_TYPE_DESC member_type_desc;
            member_type->GetDesc(&member_type_desc);

            auto type = member_type_desc.Type;
            const char* name = var_type->GetMemberTypeName(m);

            std::pair<String, String> tn;
            if (type == D3D_SVT_UINT) {
                tn.first = "uint32_t";
            } else {
                CORE_ASSERT(!"Not implemented.");
            }
            tn.second = name;

            type_and_name.push_back(tn);
        }
    }
#endif


    u32 num_input_elements = vs_reflection.shader_desc.InputParameters;
    u32 num_render_targets = ps_reflection.shader_desc.OutputParameters;

    Array <D3D12_INPUT_ELEMENT_DESC> vs_input_elements(num_input_elements);
    vs_reflection.get_input_parameters(vs_input_elements.data());

    // pso creation
    DX12_Graphics_Pipeline_Desc pso_desc = {
        .root_signature     = root_signature,

        .num_input_elements = num_input_elements,
        .input_elements     = vs_input_elements.data(),

        .topology           = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,

        .cull_mode          = cull_mode,

        .depth_enabled = has_depth,
        .num_render_targets = num_render_targets,
        .dsv_format  = dsv_format,

        .vs_bytecode = compiled_vs.bytecode,
        .vs_length   = compiled_vs.length, 

        .ps_bytecode = compiled_ps.bytecode,
        .ps_length   = compiled_ps.length, 
    };
    memcpy(&pso_desc.rtv_formats, rtv_formats, sizeof(pso_desc.rtv_formats[0]) * num_render_targets);

    *out_pso = dx12_create_graphics_pipeline_state(device, pso_desc);

    // Cleanup
    vs_reflection.release();
    compiled_vs.release();

    compiled_ps.release();
    ps_reflection.release();

    delete [] source;
}

void init_compute_pipeline(DX12_Device* device, Shader_Compiler* compiler,
                           DX12_Pipeline_State* out_pso,
                           ID3D12RootSignature* root_signature,
                           const String& shader_path)
{
    // read file.
    u64 length = read_entire_file(shader_path, nullptr);
    u8* source = new u8[length];
    read_entire_file(shader_path, source);

    // entries/profiles
    const char* cs_entry   = "cs_main";
    const char* cs_profile = "cs_6_6";

    // compile and reflect.
    auto compiled_cs   = compiler->compile(true, source, length, shader_path, cs_entry, cs_profile);
    auto cs_reflection = compiler->reflect(compiled_cs.result);

    // pso creation
    DX12_Compute_Pipeline_Desc pso_desc = {
        .root_signature    = root_signature,
        .bytecode          = compiled_cs.bytecode,
        .length            = compiled_cs.length,
        .thread_group_size = {}
    };

    *out_pso = dx12_create_compute_pipeline_state(device, pso_desc);

    // Get thread group size.
    cs_reflection.reflection->GetThreadGroupSize(&pso_desc.thread_group_size[0], &pso_desc.thread_group_size[1], &pso_desc.thread_group_size[2]);

    // Set
    out_pso->compute = pso_desc;

    // Cleanup
    cs_reflection.release();
    compiled_cs.release();

    delete [] source;
}

enum Pipeline_Type {
    PIPELINE_TYPE_INVALID = 0,
    PIPELINE_TYPE_GRAPHICS,
    PIPELINE_TYPE_COMPUTE,
};

struct Shader_Asset {
    Pipeline_Type       type;

    // Graphics
    String              name;
    String              shader_path;
    D3D12_CULL_MODE     cull_mode;
    Array <DXGI_FORMAT> render_target_formats;
    bool                has_depth;
    DXGI_FORMAT         depth_format;

    // Compute
};

Shader_Asset shader_asset_from_file(const String& file_path) {
    Shader_Asset result = {};

    std::ifstream f(file_path);
    nlohmann::json json = nlohmann::json::parse(f);

    // If key doesn't exist, it'll fail, which is neat.
    String pipeline    = json["pipeline"];
    String name        = json["name"];
    String shader_path = json["shader"];

    result.name        = name;
    result.shader_path = shader_path;

    if (pipeline == "Graphics") {
        auto render_targets = json["render_targets"];
        String cull         = json["cull"];
        String depth_format = json["depth_format"];

        // Render target formats
        Array <DXGI_FORMAT> formats;
        for (auto& it : render_targets) {
            DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
            String str = it.get<String>();
            if (str == "R32G32B32A32_FLOAT") {
                format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            } else if (str == "R8G8B8A8_UNORM") {
                format = DXGI_FORMAT_R8G8B8A8_UNORM;
            } else if (str == "R8G8B8A8_UNORM_SRGB") {
                format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            } else if (str == "R8G8B8A8_SNORM") {
                format = DXGI_FORMAT_R8G8B8A8_SNORM;
            } else if (str == "R32G32_FLOAT") {
                format = DXGI_FORMAT_R32G32_FLOAT;
            } else if (str == "R32_UINT") {
                format = DXGI_FORMAT_R32_UINT;
            } else if (str == "R16_UINT") {
                format = DXGI_FORMAT_R16_UINT;
            } else if (str == "R11G11B10_FLOAT") {
                format = DXGI_FORMAT_R11G11B10_FLOAT;
            } else {
                CORE_ASSERT(0);
            }
            formats.push_back(format);
        }

        // Cull mode
        D3D12_CULL_MODE cull_mode = D3D12_CULL_MODE_NONE;
        if (cull == "none") {
            cull_mode = D3D12_CULL_MODE_NONE;
        } else if (cull == "back") {
            cull_mode = D3D12_CULL_MODE_BACK;
        } else if (cull == "front") {
            cull_mode = D3D12_CULL_MODE_FRONT;
        } else {
            CORE_ASSERT(0);
        }

        // Depth
        bool has_depth = true;
        DXGI_FORMAT depth = DXGI_FORMAT_UNKNOWN;
        if (depth_format == "none") {
            has_depth = false;
        } else if (depth_format == "D24S8") {
            depth = DXGI_FORMAT_D24_UNORM_S8_UINT;
        } else if (depth_format == "D32") {
            depth = DXGI_FORMAT_D32_FLOAT;
        } else {
            CORE_ASSERT(0);
        }

        result.type                  = PIPELINE_TYPE_GRAPHICS,
        result.cull_mode             = cull_mode;
        result.render_target_formats = std::move(formats);
        result.has_depth             = has_depth;
        result.depth_format          = depth;

    } else if (pipeline == "Compute") {
        result.type = PIPELINE_TYPE_COMPUTE;
    } else {
        CORE_ASSERT(!"Unknown pipeline type");
    }

    return result;
}

void init_pipeline_state_from_shader_asset(DX12_Device* device, Shader_Compiler* compiler, DX12_Pipeline_State* out_pso, ID3D12RootSignature* bindless_root_signature, const Shader_Asset& asset) {
    if (asset.type == PIPELINE_TYPE_GRAPHICS) {
        init_graphics_pipeline(device, compiler, out_pso, bindless_root_signature,
                               (DXGI_FORMAT*)asset.render_target_formats.data(), 
                               asset.has_depth, asset.depth_format, 
                               asset.cull_mode, asset.shader_path);
    } else if (asset.type == PIPELINE_TYPE_COMPUTE) {
        init_compute_pipeline(device, compiler, out_pso, bindless_root_signature, asset.shader_path);
    } else {
        CORE_ASSERT(!"Unknown pipeline type");
    }
}

void deinit_pipeline_state(DX12_Pipeline_State* pso) {
    if (pso) {
        if (pso->pso) { pso->pso->Release(); }
    }
}

#if 1
#  define WIDTH  2560
#  define HEIGHT 1440
#else
#  define WIDTH  1920
#  define HEIGHT 1080
#endif

void update_per_instance_transforms(Scene* world, Entity* entity, u32& current_index, void* mapped_ptr, u32 stride, m4x4 parent_transform)
{
    m4x4 local_transform =to_m4x4(entity->position, entity->orientation, entity->scaling);
    m4x4 transform = local_transform * parent_transform;
    u8* ptr = (u8*)mapped_ptr + stride * current_index;
    memcpy(ptr, &transform, stride);

    entity->current_transform_index = current_index;

    for (auto id : entity->children) {
        auto* child = world->get_entity(id);
        update_per_instance_transforms(world, child, ++current_index, mapped_ptr, stride, transform);
    }
}

int main()
{
    // Scene
    Scene* scene = new Scene;

    // Editor
    Editor_State* editor = new Editor_State;

    // Asset
    Asset_State* asset_state = new Asset_State;
    file_sys::path project_dir  = "C:/dev/swl/RHI/Project";
    file_sys::path asset_dir    = project_dir / "Asset";
    file_sys::path material_dir = asset_dir / "Material";
    file_sys::path shader_dir   = asset_dir / "Shader";

    file_sys::path generated_shader_header_dir = project_dir / "Generated" / "Shader";

    // Create a window.
    String title = "This is a window";
    int window_width  = WIDTH;
    int window_height = HEIGHT;
    Window* window = create_window(title, window_width, window_height);

    // Resource state
    Resource_State* resource_state = new Resource_State;

    Shader_Compiler* compiler = new Shader_Compiler;
    init_shader_compiler(compiler);

    auto* device = new DX12_Device;
    assert(dx12_init_device(device, true));

    auto* cmd_queue = new DX12_Command_Queue;
    assert(dx12_init_command_queue(device, cmd_queue, D3D12_COMMAND_LIST_TYPE_DIRECT));

    auto* cmd_list = new DX12_Command_List;
    assert(dx12_init_command_list(device, cmd_list, D3D12_COMMAND_LIST_TYPE_DIRECT));

    auto* compute_queue = new DX12_Command_Queue;
    assert(dx12_init_command_queue(device, compute_queue, D3D12_COMMAND_LIST_TYPE_COMPUTE));

    auto* compute_list = new DX12_Command_List;
    assert(dx12_init_command_list(device, compute_list, D3D12_COMMAND_LIST_TYPE_COMPUTE));

    auto* fence = new DX12_Fence;
    assert(dx12_init_fence(device, fence));

    auto* rtv_heap     = dx12_create_descriptor_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV,          32);
    auto* dsv_heap     = dx12_create_descriptor_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,          32);
    auto* srv_heap     = dx12_create_descriptor_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 512);
    auto* sampler_heap = dx12_create_descriptor_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,      32);

    u32 tex_width  = WIDTH;
    u32 tex_height = HEIGHT;
    u32 num_frames = 3;
    HWND hwnd = (HWND)window->get_platform_window();
    auto* swap_chain = dx12_create_swap_chain(device, cmd_queue, rtv_heap, hwnd, tex_width, tex_height, num_frames);

    auto* bindless_root_signature = dx12_create_bindless_root_signature(device);







    // Import GLTF manually.
    if (0) {
        String path = (asset_dir / "Model/Sponza/Sponza.gltf").string();

        auto loaded = load_gltf(scene, asset_state, path, false);
        for (auto* entity : loaded.entities) {
            scene->root->children.push_back(entity->id);
        }

        // Material
        auto& json_array = loaded.materials;
        for (const auto& json : json_array) {
            Material mat = material_from_json(device, cmd_queue, cmd_list, fence, srv_heap, json);
            resource_state->alloc_material(mat);
        }

        // Mesh
        for (auto* mesh : loaded.meshes) {
            auto res = alloc_mesh_resource(device, cmd_queue, cmd_list, fence, srv_heap, mesh);
            resource_state->meshes[res.name] = res;
        }
    }

    if (0) {
        String path = (asset_dir / "Model/DamagedHelmet/DamagedHelmet.gltf").string();

        auto loaded = load_gltf(scene, asset_state, path, true);
        for (auto* entity : loaded.entities) {
            scene->root->children.push_back(entity->id);
        }

        // Material
        auto& json_array = loaded.materials;
        for (const auto& json : json_array) {
            Material mat = material_from_json(device, cmd_queue, cmd_list, fence, srv_heap, json);
            resource_state->alloc_material(mat);
        }

        // Mesh
        for (auto* mesh : loaded.meshes) {
            auto res = alloc_mesh_resource(device, cmd_queue, cmd_list, fence, srv_heap, mesh);
            resource_state->meshes[res.name] = res;
        }
    }


    // @Temporary
    //
    Camera camera; 
    {
        camera.position     = vec3(4.f, 2.f, 0.f);
        camera.aspect_ratio = 9.f / 16.f;
        camera.fov          = to_radian(90.0f);
        camera.near_z       = 0.1f;
        camera.far_z        = 10000.0f;
        camera.yaw          = to_radian(-90.0f);
        camera.pitch        = to_radian(0.0f);
        camera.speed        = 1.0f;
    };

    Shader_Camera shader_camera = get_shader_camera(&camera);
    auto* camera_resource = dx12_alloc_resource(device, { .type = DX12_RESOURCE_TYPE_BUFFER, .buffer = { .size = (u32)sizeof(shader_camera) } });
    auto camera_srv = dx12_alloc_descriptor(srv_heap);
    dx12_create_srv(device, camera_resource, &camera_srv, DXGI_FORMAT_UNKNOWN, 1, (u32)sizeof(shader_camera));


    // @Temporary: Create linear sampler.
    //
    auto linear_wrap = dx12_alloc_descriptor(sampler_heap);
    {
        D3D12_SAMPLER_DESC desc = {
            .Filter         = D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            .AddressU       = D3D12_TEXTURE_ADDRESS_MODE_WRAP, 
            .AddressV       = D3D12_TEXTURE_ADDRESS_MODE_WRAP, 
            .AddressW       = D3D12_TEXTURE_ADDRESS_MODE_WRAP, 
            .MipLODBias     = 0.f,
            .ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER,
            .MinLOD         = 0.f,
            .MaxLOD         = 1e10f
        };
        device->native_device->CreateSampler(&desc, linear_wrap.cpu_handle);
    }

    auto bilinear_clamp = dx12_alloc_descriptor(sampler_heap);
    {
        D3D12_SAMPLER_DESC desc = {
            .Filter         = D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            .AddressU       = D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 
            .AddressV       = D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 
            .AddressW       = D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 
            .MipLODBias     = 0.f,
            .ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER,
            .MinLOD         = 0.f,
            .MaxLOD         = 1e10f
        };
        device->native_device->CreateSampler(&desc, bilinear_clamp.cpu_handle);
    }

    // @Temporary: Create linear sampler.
    auto anisotropic_sampler_descriptor = dx12_alloc_descriptor(sampler_heap);
    {
        u32 x = 16;
        D3D12_SAMPLER_DESC desc = {
            .Filter         = D3D12_FILTER_ANISOTROPIC,
            .AddressU       = D3D12_TEXTURE_ADDRESS_MODE_MIRROR, 
            .AddressV       = D3D12_TEXTURE_ADDRESS_MODE_MIRROR, 
            .AddressW       = D3D12_TEXTURE_ADDRESS_MODE_MIRROR, 
            .MipLODBias     = 0.f,
            .MaxAnisotropy  = x,
            .ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER,
            .MinLOD         = 0.f,
            .MaxLOD         = 1e10f
        };
        device->native_device->CreateSampler(&desc, anisotropic_sampler_descriptor.cpu_handle);
    }


    // Load shader meta data assets from disk.
    //
    // @Todo: GETTER??
    Hash_Table <String, DX12_Pipeline_State*> shader_table;

    for (const auto& entry : file_sys::directory_iterator(shader_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            String file = entry.path().string();
            auto shader_asset = shader_asset_from_file(file);

            auto* shader = new DX12_Pipeline_State;
            init_pipeline_state_from_shader_asset(device, compiler, shader, bindless_root_signature, shader_asset);

            if (!shader_table.contains(shader_asset.name)) {
                shader_table[shader_asset.name] = shader;
            } else  {
                CORE_ASSERT(0);
            }
        }
    }


    // Persistently mapped per instance draw data
    u32 max_transforms = 1024;
    u32 stride = sizeof(m4x4);
    auto* transforms_resource = dx12_alloc_resource(device, { .type = DX12_RESOURCE_TYPE_BUFFER, .heap_type = D3D12_HEAP_TYPE_UPLOAD, .buffer = { .size = stride * max_transforms } }, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* transforms_ptr = nullptr;
    D3D12_RANGE read_range = {0,0};
    transforms_resource->native_resource->Map(0, &read_range, &transforms_ptr);
    auto transforms_srv = dx12_alloc_descriptor(srv_heap);
    dx12_create_srv(device, transforms_resource, &transforms_srv, DXGI_FORMAT_UNKNOWN, max_transforms, stride);

    // @Temporary
    // @Temporary
    DX12_Resource* noise = nullptr;
    auto noise_uav = dx12_alloc_descriptor(srv_heap);
    auto noise_srv = dx12_alloc_descriptor(srv_heap);
    {
        DX12_Resource_Desc desc = {
            .type = DX12_RESOURCE_TYPE_TEXTURE_3D,
            .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            .texture = {
                .format      = DXGI_FORMAT_R8G8B8A8_UNORM,
                .width       = 128,
                .height      = 128,
                .mip_levels  = 1,
                .depth       = 128,
                .num_samples = 1,
            }
        };
        noise = dx12_alloc_resource(device, desc);
        noise->native_resource->SetName(L"CloudNoiseTexture");
        dx12_create_uav(device, noise, &noise_uav, noise->desc.texture.format);
        dx12_create_srv(device, noise, &noise_srv, noise->desc.texture.format);

        auto* shader = shader_table.at("CloudNoise");
        cmd_list->begin();
        {
            cmd_list->set_resource_and_sampler_heap(srv_heap, sampler_heap);
            cmd_list->set_pipeline_state(shader);
            cmd_list->set_compute_root_signature(shader->compute.root_signature);

            u32 texture_id = noise_uav.index;
            cmd_list->set_compute_root_constants(0, 1, &texture_id);

            // thread group dimension
            u32 tx = shader->compute.thread_group_size[0];
            u32 ty = shader->compute.thread_group_size[1];
            u32 tz = shader->compute.thread_group_size[2];

            u32 x = noise->desc.texture.width;
            u32 y = noise->desc.texture.height;
            u32 z = noise->desc.texture.depth;

            cmd_list->dispatch((x+tx-1)/tx, (y+ty-1)/ty, (z+tz-1)/tz);
        }
        cmd_list->end();
        dx12_execute_command_list(cmd_queue, cmd_list);
        dx12_signal_fence(cmd_queue, fence);
        dx12_wait_fence(fence);
    }

    // @Temporary
    // @Temporary
    DX12_Resource* weather_map = nullptr;
    auto weather_map_srv = dx12_alloc_descriptor(srv_heap);
    auto weather_map_uav = dx12_alloc_descriptor(srv_heap);
    {
        DX12_Resource_Desc desc = {
            .type = DX12_RESOURCE_TYPE_TEXTURE_2D,
            .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            .texture = {
                .format      = DXGI_FORMAT_R11G11B10_FLOAT,
                .width       = 1024,
                .height      = 1024,
                .mip_levels  = 1,
                .depth       = 1,
                .num_samples = 1,
            }
        };
        weather_map = dx12_alloc_resource(device, desc);
        weather_map->native_resource->SetName(L"WeatherMap");

        dx12_create_srv(device, weather_map, &weather_map_srv, weather_map->desc.texture.format);
        dx12_create_uav(device, weather_map, &weather_map_uav, weather_map->desc.texture.format);

        auto* shader = shader_table.at("WeatherMapGeneration");
        cmd_list->begin();
        {
            cmd_list->set_resource_and_sampler_heap(srv_heap, sampler_heap);
            cmd_list->set_pipeline_state(shader);
            cmd_list->set_compute_root_signature(shader->compute.root_signature);

            u32 texture_id = weather_map_uav.index;
            cmd_list->set_compute_root_constants(0, 1, &texture_id);

            // thread group dimension
            u32 tx = shader->compute.thread_group_size[0];
            u32 ty = shader->compute.thread_group_size[1];

            u32 x = weather_map->desc.texture.width;
            u32 y = weather_map->desc.texture.height;

            cmd_list->dispatch((x+tx-1)/tx, (y+ty-1)/ty, 1);
        }
        cmd_list->end();
        dx12_execute_command_list(cmd_queue, cmd_list);
        dx12_signal_fence(cmd_queue, fence);
        dx12_wait_fence(fence);
    }


    // Alloc resources
    //
    resource_state->alloc_resource("GBufferPosition", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, .texture = { .format = DXGI_FORMAT_R32G32B32A32_FLOAT, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
    resource_state->alloc_resource(  "GBufferNormal", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, .texture = { .format =     DXGI_FORMAT_R8G8B8A8_SNORM, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
    resource_state->alloc_resource(      "GBufferUV", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, .texture = { .format =       DXGI_FORMAT_R32G32_FLOAT, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
    resource_state->alloc_resource("GBufferMaterial", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, .texture = { .format =           DXGI_FORMAT_R16_UINT, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
    resource_state->alloc_resource(          "Depth", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, .texture = { .format =          DXGI_FORMAT_D32_FLOAT, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 }, .do_clear = true, .clear_value = { .Format = DXGI_FORMAT_D32_FLOAT, .DepthStencil = { .Depth = 1.0f, .Stencil = 0 } } });
    resource_state->alloc_resource(          "Color", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET|D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, .texture = { .format =     DXGI_FORMAT_R11G11B10_FLOAT, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
    resource_state->alloc_resource(     "Composited", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET|D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, .texture = { .format =     DXGI_FORMAT_R11G11B10_FLOAT, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
    resource_state->alloc_resource(        "BlitOut", device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, .texture = { .format =     DXGI_FORMAT_R8G8B8A8_UNORM, .width = tex_width, .height = tex_height, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
    
    resource_state->alloc_pass_resource("GBufferPosition", device, srv_heap,  nullptr, rtv_heap,  nullptr);
    resource_state->alloc_pass_resource(  "GBufferNormal", device, srv_heap,  nullptr, rtv_heap,  nullptr);
    resource_state->alloc_pass_resource(      "GBufferUV", device, srv_heap,  nullptr, rtv_heap,  nullptr);
    resource_state->alloc_pass_resource("GBufferMaterial", device, srv_heap,  nullptr, rtv_heap,  nullptr);
    resource_state->alloc_pass_resource(          "Depth", device,  nullptr,  nullptr,  nullptr, dsv_heap);
    resource_state->alloc_pass_resource(          "Color", device, srv_heap, srv_heap, rtv_heap,  nullptr);
    resource_state->alloc_pass_resource(     "Composited", device, srv_heap, srv_heap, rtv_heap,  nullptr);
    resource_state->alloc_pass_resource(        "BlitOut", device, srv_heap,  nullptr, rtv_heap,  nullptr);



    auto* gbuffer_pass = new GBuffer_Pass;
    {
        auto* pass = gbuffer_pass;
        pass->pipeline_state  = shader_table.at("GBuffer");
        pass->viewport_width  = tex_width;
        pass->viewport_height = tex_height;
        pass->scissor_width   = tex_width;
        pass->scissor_height  = tex_height;
        pass->topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

        pass->outputs.push_back("GBufferPosition");
        pass->outputs.push_back("GBufferNormal");
        pass->outputs.push_back("GBufferUV");
        pass->outputs.push_back("GBufferMaterial");
        pass->depth_target = "Depth";
    }

    auto* defer_pass = new Defer_Pass;
    {
        auto* pass = defer_pass;
        pass->pipeline_state  = shader_table.at("Defer");
        pass->viewport_width  = tex_width;
        pass->viewport_height = tex_height;
        pass->scissor_width   = tex_width;
        pass->scissor_height  = tex_height;
        pass->topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

        pass->inputs.push_back("GBufferPosition");
        pass->inputs.push_back("GBufferNormal");
        pass->inputs.push_back("GBufferUV");
        pass->inputs.push_back("GBufferMaterial");
        pass->outputs.push_back("Color");
    }

    auto* cloud_pass = new Cloud_Pass;
    {
        auto* pass = cloud_pass;
        pass->pipeline_state  = shader_table.at("Cloud");
        pass->viewport_width  = tex_width;
        pass->viewport_height = tex_height;
        pass->scissor_width   = tex_width;
        pass->scissor_height  = tex_height;
        pass->topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

        pass->outputs.push_back("Color");
        pass->depth_target = "Depth"; // @Fix: Why not working
    }

    auto* composition_pass = new Composition_Pass;
    {
        auto* pass = composition_pass;
        pass->pipeline_state  = shader_table.at("Composition");
        pass->viewport_width  = tex_width;
        pass->viewport_height = tex_height;
        pass->scissor_width   = tex_width;
        pass->scissor_height  = tex_height;
        pass->topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

        pass->inputs.push_back("Color");
        pass->inputs.push_back("Bloom0");
        pass->outputs.push_back("Composited");
    }

    auto* blit_pass = new Blit_Pass;
    {
        auto* pass = blit_pass;
        pass->pipeline_state = shader_table.at("Blit");
        pass->viewport_width  = tex_width;
        pass->viewport_height = tex_height;
        pass->scissor_width   = tex_width;
        pass->scissor_height  = tex_height;
        pass->topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

        pass->inputs.push_back("Composited");
        pass->outputs.push_back("BlitOut");
    }
    // @Todo:
    // I'm implicitly connecting blit_output_resource to swap chain's current 
    // frame resource. I think the system should manage the final connection thing.


    


    // UI
    //
    g_imgui_srv_heap = srv_heap;
    float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    editor->font_body = io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf"); // @Temporary
    ImGui::PushFont(editor->font_body);
  
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    ui_apply_style();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale;        // Set initial font scale. (in docking branch: using io.ConfigDpiScaleFonts=true automatically overrides this for every window depending on the current monitor)

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForD3D(window->sdl);
    ImGui_ImplDX12_InitInfo init_info = {};
    {
        init_info.Device               = device->native_device;
        init_info.CommandQueue         = cmd_queue->native_cmd_queue;
        init_info.NumFramesInFlight    = swap_chain->num_frames;
        init_info.RTVFormat            = DXGI_FORMAT_R8G8B8A8_UNORM;
        init_info.DSVFormat            = DXGI_FORMAT_UNKNOWN;
        // Allocating SRV descriptors (for textures) is up to the application, so we provide callbacks.
        // (current version of the backend will only allocate one descriptor, future versions will need to allocate more)
        init_info.SrvDescriptorHeap    = g_imgui_srv_heap->native_heap;
        init_info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle) {
            auto desc = dx12_alloc_descriptor(g_imgui_srv_heap);
            *out_cpu_handle = desc.cpu_handle;
            *out_gpu_handle = desc.gpu_handle;
        };
        init_info.SrvDescriptorFreeFn  = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) {
            // @Todo: Callbacks...
        };
    }
    ImGui_ImplDX12_Init(&init_info);


    // @Temporary: Bloom
    //
    f32 bloom_threshold    = 1.0f;
    f32 bloom_radius_scale = 2.0f;
    f32 bloom_strength     = 0.0f;
    u32 bloom_num_mips     = 5;
    for (u32 i = 0; i < bloom_num_mips; ++i) {
        const String name = "Bloom" + std::to_string(i);
        u32 w = max(tex_width >> i, 1u);
        u32 h = max(tex_height >> i, 1u);
        resource_state->alloc_resource(name, device, { .type = DX12_RESOURCE_TYPE_TEXTURE_2D, .resource_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, .texture = { .format = DXGI_FORMAT_R11G11B10_FLOAT, .width = w, .height = h, .mip_levels = 1, .depth = 1, .num_samples = 1 } });
        resource_state->alloc_pass_resource(name, device, srv_heap, srv_heap, nullptr,  nullptr);
    }

    // @Temporary: Atmosphere
    //
    f32 sun_illuminance = 1.2e5;
    vec3 sun_color_linear = vec3(1.0f, 1.0f, 1.0f);
    //f32 sun_illuminance = 0.01f;
    f32 sun_azimuth     = 0.0f;
    f32 sun_theta       = 45.0f;
    // Unit: Degree.
    // 0.527 ~ 0.545 angular diameter, depends on time of year.
    // Interestingly, the sun is actually small.
    f32 sun_angular_radius = 0.545f * 0.5f;
    s32 num_view_samples  = 16;
    s32 num_sun_samples   = 8;
    s32 num_cloud_samples = 8;

    // @Temporary
    f32 shader_time = 0.0f;




    // Main loop
    //
    while (window->is_running) {
        window->poll_events();

        // Update
        f32 time_elapsed = 0.017f;
        constexpr f32 dt = 1.f / 60.f;
        for (;time_elapsed >= dt; time_elapsed -= dt) {
            camera.update(dt, window->my_input_system);
        }
        shader_time += dt;

        // UI
        if (window->my_input_system->key_is_down[KEY_F1] && !window->my_input_system->key_was_down[KEY_F1]) {
            editor->show_ui = !editor->show_ui;
        }
        ui_begin(); 
        if (editor->show_ui) 
        {
            ImGui::Begin("Panel");
            {
                ImGui::Text("%.3f mspf (%.1f fps)", 1000.0f / io.Framerate, io.Framerate);

                ImGui::SeparatorText("Atmosphere");
                {
                    ImGui::SliderFloat("Sun Illuminance", &sun_illuminance, 5.0e4, 3.0e5, "%.f lux");
                    ImGui::SliderFloat3("Sun Color (RGB Linear)", &sun_color_linear.x, 0.0f, 1.0f);
                    ImGui::SliderFloat("Sun Theta", &sun_theta, -180.0f, 180.0f);
                    ImGui::SliderFloat("Sun Azimuth", &sun_azimuth, 0.0f, 180.0f);
                    ImGui::SliderInt("Number of view samples", &num_view_samples, 1, 128);
                    ImGui::SliderInt("Number of sun samples", &num_sun_samples, 1, 128);
                    ImGui::SliderInt("Number of cloud samples", &num_cloud_samples, 1, 128);
                }

                ImGui::SeparatorText("Bloom");
                {
                    ImGui::SliderFloat("Bloom Threshold", &bloom_threshold, 0.0f, 32.0f);
                    ImGui::SliderFloat("Bloom Radius Scale", &bloom_radius_scale, 1.0f, 32.0f);
                    ImGui::SliderFloat("Bloom Strength", &bloom_strength, 0.0f, 1.0f);
                }
            }
            ImGui::End();

            ui_main_menu_bar();
            ui_scene_hierarchy(editor, scene);
            ui_entity_property(editor, scene);
            ui_gizmo(editor, scene, camera.view, camera.proj, 0, 0, tex_width, tex_height);

        }
        ui_end();



        // Upload camera resource
        shader_camera = get_shader_camera(&camera);
        dx12_upload_buffer(device, cmd_queue, cmd_list, fence, camera_resource, &shader_camera, (u32)sizeof(shader_camera));

        // Upload per instance data
        u32 current_index = 0;
        for (auto id : scene->root->children) {
            auto* entity = scene->get_entity(id);
            update_per_instance_transforms(scene, entity, current_index, transforms_ptr, stride, m4x4::identity());
            current_index++;
        }




        cmd_list->begin();
        {
            cmd_list->set_resource_and_sampler_heap(srv_heap, sampler_heap);
            cmd_list->set_graphics_root_signature(bindless_root_signature);

            {
                gbuffer_pass->begin(resource_state, cmd_list);
                gbuffer_pass->execute(resource_state, cmd_list, scene, transforms_srv.index, camera_srv.index, anisotropic_sampler_descriptor.index);
            }

            //auto rtv = resource_state->get_pass_resource("Color").rtv;
            //cmd_list->clear_rtv(&rtv, 0.3, 0.3, 0.9, 1.0);

            if (0) {
                Defer_Pass::Draw_Data param = {
                    .camera_id      = camera_srv.index,
                    .linear_id      = linear_wrap.index,
                    .anisotropic_id = anisotropic_sampler_descriptor.index,
                };
                defer_pass->begin(resource_state, cmd_list);
                defer_pass->execute(resource_state, cmd_list, param);
            }

            //sun_theta = (cos(shader_time*0.25f) * 0.5f + 0.5f) * 90.0f - 10.0f;

            if (1) {
                float theta   = to_radian(sun_theta);
                float azimuth = to_radian(sun_azimuth);
                vec3 sun_direction = vec3(cos(theta) * sin(azimuth),
                                          sin(theta),
                                          cos(theta) * cos(azimuth));

                Cloud_Pass::Push_Constants push = {
                    .camera_id          = camera_srv.index,

                    .noise_id           = noise_srv.index,
                    .weather_map_id     = weather_map_srv.index,

                    .linear_wrap_id     = linear_wrap.index,

                    .sun_illuminance    = sun_illuminance,
                    .sun_color_linear   = sun_color_linear,
                    .sun_direction      = sun_direction,
                    .sun_angular_radius = to_radian(sun_angular_radius),

                    .num_view_samples   = num_view_samples,
                    .num_sun_samples    = num_sun_samples,
                    .num_cloud_samples  = num_cloud_samples
                };
                cloud_pass->begin(resource_state, cmd_list);
                cloud_pass->execute(resource_state, cmd_list, push);
            }

            // @Temporary:
            //
            {
                auto* shader = shader_table.at("BloomThreshold");
                cmd_list->set_pipeline_state(shader);
                cmd_list->set_compute_root_signature(shader->compute.root_signature);

                struct Push {
                    u32 input_srv;
                    u32 output_uav;
                    u32 bilinear_clamp;
                    f32 threshold;
                };

                const String in_name  = "Color";
                const String out_name = "Bloom0";
                const auto& in  = resource_state->get_pass_resource(in_name);
                const auto& out = resource_state->get_pass_resource(out_name);
                Push push = {
                    .input_srv      = in.srv.index,
                    .output_uav     = out.uav.index,
                    .bilinear_clamp = bilinear_clamp.index,
                    .threshold      = bloom_threshold
                };
                cmd_list->set_compute_root_constants(0, sizeof(push) >> 2, &push);

                auto* in_res  = resource_state->get_resource(in_name);
                auto* out_res = resource_state->get_resource(out_name);

                cmd_list->transition_barrier(in_res,  0, D3D12_RESOURCE_STATE_COMMON);
                cmd_list->transition_barrier(out_res, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                u32 w = out_res->desc.texture.width;
                u32 h = out_res->desc.texture.height;
                cmd_list->dispatch((w + 7) / 8, (h + 7) / 8, 1);

                // Barrier
                D3D12_RESOURCE_BARRIER uav_barrier = {
                    .Type  = D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    .UAV   = { .pResource = out_res->native_resource }
                };
                cmd_list->native_cmd_list->ResourceBarrier(1, &uav_barrier);
            }
            {
                auto* shader = shader_table["BloomDownsample"];
                cmd_list->set_pipeline_state(shader);
                cmd_list->set_compute_root_signature(shader->compute.root_signature);

                struct Push {
                    u32 input_srv;
                    u32 output_uav;
                    u32 bilinear_clamp;
                    f32 radius_scale;
                };

                auto bloom_downsample = [&](const String& in_name, const String& out_name) {
                    const auto& in  = resource_state->get_pass_resource(in_name);
                    const auto& out = resource_state->get_pass_resource(out_name);
                    Push push = {
                        .input_srv      = in.srv.index,
                        .output_uav     = out.uav.index,
                        .bilinear_clamp = bilinear_clamp.index,
                        .radius_scale   = bloom_radius_scale
                    };
                    cmd_list->set_compute_root_constants(0, sizeof(push) >> 2, &push);

                    auto* in_res  = resource_state->get_resource(in_name);
                    auto* out_res = resource_state->get_resource(out_name);

                    cmd_list->transition_barrier(in_res,  0, D3D12_RESOURCE_STATE_COMMON);
                    cmd_list->transition_barrier(out_res, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                    u32 w = out_res->desc.texture.width;
                    u32 h = out_res->desc.texture.height;
                    cmd_list->dispatch((w + 7) / 8, (h + 7) / 8, 1);

                    // Barrier
                    D3D12_RESOURCE_BARRIER uav_barrier = {
                        .Type  = D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        .UAV   = { .pResource = out_res->native_resource }
                    };
                    cmd_list->native_cmd_list->ResourceBarrier(1, &uav_barrier);
                };

                for (u32 i = 0; i < bloom_num_mips - 1; ++i) {
                    bloom_downsample("Bloom" + std::to_string(i), "Bloom" + std::to_string(i + 1));
                }
            }

            if (1) {
                auto* shader = shader_table["BloomUpsample"];
                cmd_list->set_pipeline_state(shader);
                cmd_list->set_compute_root_signature(shader->compute.root_signature);

                struct Push {
                    u32 input_srv;
                    u32 output_uav;
                    u32 bilinear_clamp;
                };

                auto bloom_upsample = [&](const String& in_name, const String& out_name) {
                    const auto& in  = resource_state->get_pass_resource(in_name);
                    const auto& out = resource_state->get_pass_resource(out_name);
                    Push push = {
                        .input_srv      = in.srv.index,
                        .output_uav     = out.uav.index,
                        .bilinear_clamp = bilinear_clamp.index
                    };
                    cmd_list->set_compute_root_constants(0, sizeof(push) >> 2, &push);

                    auto* in_res  = resource_state->get_resource(in_name);
                    auto* out_res = resource_state->get_resource(out_name);

                    cmd_list->transition_barrier(in_res,  0, D3D12_RESOURCE_STATE_COMMON);
                    cmd_list->transition_barrier(out_res, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                    u32 w = out_res->desc.texture.width;
                    u32 h = out_res->desc.texture.height;
                    cmd_list->dispatch((w + 7) / 8, (h + 7) / 8, 1);

                    // Barrier
                    D3D12_RESOURCE_BARRIER uav_barrier = {
                        .Type  = D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        .UAV   = { .pResource = out_res->native_resource }
                    };
                    cmd_list->native_cmd_list->ResourceBarrier(1, &uav_barrier);
                };

                for (u32 i = bloom_num_mips - 1; i >= 1; --i) {
                    bloom_upsample("Bloom" + std::to_string(i), "Bloom" + std::to_string(i - 1));
                }
            }

            {
                Composition_Pass::Push_Constants push = {
                    .color_id          = resource_state->get_pass_resource("Color").get_srv_index(),
                    .bilinear_clamp_id = bilinear_clamp.index,
                    .bloom_id          = resource_state->get_pass_resource("Bloom0").get_srv_index(),
                    .bloom_strength    = bloom_strength,
                };
                composition_pass->begin(resource_state, cmd_list);
                composition_pass->execute(resource_state, cmd_list, push);
            }

            cmd_list->set_graphics_root_signature(bindless_root_signature);
            {
                Blit_Pass::Draw_Data param = {
                    .linear_id = linear_wrap.index
                };
                blit_pass->begin(resource_state, cmd_list);
                blit_pass->execute(resource_state, cmd_list, param);
            }


            ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list->native_cmd_list);



            auto* final_resource = resource_state->get_resource("BlitOut");
            cmd_list->transition_barrier(final_resource, 0, D3D12_RESOURCE_STATE_COPY_SOURCE);
            cmd_list->transition_barrier(swap_chain->get_current_resource(), 0, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd_list->native_cmd_list->CopyResource(swap_chain->get_current_resource()->native_resource, final_resource->native_resource);
        }
        cmd_list->transition_barrier(swap_chain->get_current_resource(), 0, D3D12_RESOURCE_STATE_PRESENT);
        cmd_list->end();
        dx12_execute_command_list(cmd_queue, cmd_list);
        dx12_signal_fence(cmd_queue, fence);
        dx12_wait_fence(fence);
        swap_chain->present();
    }

    return 0;
}
