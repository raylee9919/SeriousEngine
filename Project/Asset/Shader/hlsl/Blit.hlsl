// Copyright Seong Woo Lee. All Rights Reserved.

#define PUSH_CONSTANTS(Type, Name) ConstantBuffer<Type> Name : register(b0)

struct PS_Input {
    float4 sv_position : SV_POSITION;
    float2 screen_uv   : TEXCOORD0;
};

struct Push_Constants {
    uint color_id;
    uint linear_sampler_id;
};
PUSH_CONSTANTS(Push_Constants, push);

float3 tonemap_reinhard(float3 x) {
    return x / (x + 1.0f);
}

float3 tonemap_aces(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

float3 tonemap_jodie_reinhard(float3 c) {
    // https://www.shadertoy.com/view/tdSXzD
    float luma = dot(c, float3(0.2126, 0.7152, 0.0722));
    float3 tc = c / (c + 1.0);
    return lerp(c / (luma + 1.0), tc, tc);
}

float3 srgb_from_linear(float3 c) {
    float3 lo = c * 12.92;
    float3 hi = (pow(abs(c), 1.0/2.4) * 1.055) - 0.055;
    float3 srgb = select(c <= 0.0031308, lo, hi);
    return srgb;
}

PS_Input vs_main(uint vertex_id : SV_VertexID)
{
    PS_Input result;
    float2 pos  = float2((vertex_id & 1) ? 3.0 : -1.0,
                         (vertex_id & 2) ? -3.0 : 1.0);
    result.sv_position = float4(pos, 0.0, 1.0);
    result.screen_uv   = float2((vertex_id & 1) ? 2.0 : 0.0,
                                (vertex_id & 2) ? 2.0 : 0.0);
    return result;
}

float4 ps_main(PS_Input input) : SV_TARGET
{
    // sampler.
    SamplerState linear_sampler = SamplerDescriptorHeap[push.linear_sampler_id];

    // g-buffer.
    Texture2D <float3> color_texture = ResourceDescriptorHeap[push.color_id];
    float3 color = color_texture.Sample(linear_sampler, input.screen_uv);

    // Tone mapping.
    color = tonemap_jodie_reinhard(color);

    // Accurate gamma correction
    color = srgb_from_linear(color);

    float4 result = float4(color, 1.0);
    return result;
}
