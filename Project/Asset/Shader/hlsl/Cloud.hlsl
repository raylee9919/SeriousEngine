// Copyright Seong Woo Lee. All Rights Reserved.

#define PI              3.14159265
#define TWO_PI          6.28318530
#define QUARTER_PI      0.78539816339
#define NDC_DEPTH_MAX   1.0
static const float  PLANET_RADIUS       = 6360e3;
static const float  ATMOSPHERE_RADIUS   = 6420e3;
static const float3 PLANET_CENTER       = float3(0.0, -PLANET_RADIUS, 0.0);

struct Camera {
    float4   position;
    float4x4 view;
    float4x4 proj;
    float4x4 view_proj;
    float4x4 inv_view;
    float4x4 inv_proj;
};

#define PUSH_CONSTANTS(Type, Name) ConstantBuffer<Type> Name : register(b0)
struct Push_Constants {
    uint camera_id;
    
    uint noise_id;
    uint weather_map_id;

    uint linear_wrap_id;

    float  sun_illuminance;
    float3 sun_color_linear;
    float3 sun_direction;
    float  sun_angular_radius; // radian

    int num_view_samples;
    int num_sun_samples;
    int num_cloud_samples;
};
PUSH_CONSTANTS(Push_Constants, push);

struct PS_Input {
    float4 sv_position : SV_POSITION;
    float2 screen_uv   : TEXCOORD0;
};

float compute_EV100(float aperture, float shutter_time, float ISO) {
    return log2(aperture * aperture / shutter_time * 100 / ISO);
}

float exposure_from_EV100(float EV100) {
    float max_luminance = 1.2 * pow(2.0, EV100);
    return 1.0 / max_luminance;
}

bool ray_sphere(float3 ray_origin, float3 ray_dir, 
                float3 center, float radius, 
                out float tmin, out float tmax)
{
    float3 oc = ray_origin - center;

    float a = dot(ray_dir, ray_dir);
    float b = dot(oc, ray_dir);
    float c = dot(oc, oc) - radius * radius;

    float discriminant = b * b - a * c;

    if (discriminant < 0.0f) {
        tmin = tmax = 0.0f;
        return false;
    }

    float sqrtD = sqrt(discriminant);
    tmin = (-b - sqrtD) / a;
    tmax = (-b + sqrtD) / a;

    return tmax >= 0.0f;
}

float altitude_of(float3 position) {
    return distance(position, PLANET_CENTER) - PLANET_RADIUS;
}

float phase_rayleigh(float u) {
    return 0.05968310365 * (1 + u*u);
}

float phase_mie_cornett(float u, float g) {
    float g2 = g*g;
    return 0.11936620731 * (((1 - g2)*(1 + u*u)) / ((2 + g2) * pow(1 + g2 - 2*g*u, 1.5))); // @Todo: abs...?
}

float phase_mie_klein_nishina(float u, float g) {
    float e = 1.0;
    for (int i = 0; i < 8; ++i) {
        float g_from_e = 1.0 / e - 2.0 / log(2.0 * e + 1.0) + 1.0;
        float deriv = 4.0 / ((2.0 * e + 1.0) * pow(log(2.0 * e + 1.0), 2)) - 1.0 / pow(e, 2);
        if (abs(deriv) < 1e-7) break;
        e = e - (g_from_e - g) / deriv;
    }
    return e / (2.0 * PI * (e * (1.0 - u) + 1.0) * log(2.0 * e + 1.0));
}

PS_Input vs_main(uint vertex_id : SV_VertexID)
{
    PS_Input result;
    float2 pos  = float2((vertex_id & 1) ? 3.0 : -1.0, (vertex_id & 2) ? -3.0 : 1.0);
    result.sv_position = float4(pos, 0.0, 1.0);
    result.screen_uv   = float2((vertex_id & 1) ? 2.0 : 0.0, (vertex_id & 2) ? 2.0 : 0.0);
    return result;
}

float3 ps_main(PS_Input input) : SV_TARGET
{
    // Pull camera.
    StructuredBuffer <Camera> camera_buffer = ResourceDescriptorHeap[push.camera_id];
    Camera camera = camera_buffer[0];
    
    // Pull sampler and textures.
    SamplerState linear_wrap       = SamplerDescriptorHeap[push.linear_wrap_id];
    Texture3D <float4> noise_tex   = ResourceDescriptorHeap[push.noise_id];
    Texture2D <float3> weather_map = ResourceDescriptorHeap[push.weather_map_id];

    // Sun
    float  sun_solid_angle           = TWO_PI * (1.0 - cos(push.sun_angular_radius));
    float3 sun_illuminance           = push.sun_illuminance;
    float3 sun_luminance             = sun_illuminance / sun_solid_angle;
    float3 sun_transmittance         = float3(0.925, 0.861, 0.755);
    float3 sun_outer_luminance       = (sun_luminance / sun_transmittance);

    // 
    float3 to_light = push.sun_direction;
    float3 camera_position = camera.position.xyz;
    float ndc_x =  2.0 * input.screen_uv.x - 1.0;
    float ndc_y = -2.0 * input.screen_uv.y + 1.0;
    float4 ndc = float4(ndc_x, ndc_y, NDC_DEPTH_MAX, 1.0);
    float4 view_space = mul(camera.inv_proj, ndc);
    view_space /= view_space.w;

    // Reconstruct ray
    float3 view_dir = normalize(mul(camera.inv_view, float4(view_space.xyz, 0.0))).xyz;
    float tmin, tmax;
    if (!ray_sphere(camera_position, view_dir, PLANET_CENTER, ATMOSPHERE_RADIUS, tmin, tmax)) {
        return 0.0;
    }
    float dist = tmin < 0.0 ? tmax : tmin;

    float step_size = dist / float(push.num_view_samples + 1);
    float3 dX       = step_size * view_dir;                 // Step
    float3 X        = camera_position + dX;                 // Sample position along view direction
    float3 L0       = sun_outer_luminance;                  // Initial luminance
    float mu        = dot(view_dir, to_light);              // Cosine of view dir and light dir

    // Coefficients (often denoted as sigma in papers)
    float3 Sr       = float3(5.802e-6, 13.558e-6, 33.1e-6); // Rayleigh scattering
    float3 Sm       = 3.996e-6;                             // Mie scattering
    float3 Am       = 4.40e-6;                              // Mie absorption
    float3 Ao       = float3(0.650e-6, 1.881e-6, 0.085e-6); // Ozone absorption

    // Phase functions
    float Pr        = phase_rayleigh(mu);
    float Pm        = phase_mie_klein_nishina(mu, 0.8); // https://twitter.com/Cody_J_Bennett/status/2055998057406181504

    // Density distribution constants
    float Hr        = 8e3;
    float Hm        = 1.2e3;

    // Optical depth
    float ODr       = 0.0;
    float ODm       = 0.0;
    float ODo       = 0.0;

    float3 sum_r    = 0.0;
    float3 sum_m    = 0.0;
    float3 sum_o    = 0.0;

    [loop]
    for (int i = 0; i < push.num_view_samples; ++i) {
        float altitude = altitude_of(X);

        // Sampled density
        float Dr = step_size * exp(-altitude / Hr);
        float Dm = step_size * exp(-altitude / Hm);
        float Do = step_size * max(0, 1 - (abs(altitude - 25e3) / 15e3));

        ODr += Dr;
        ODm += Dm;
        ODo += Do;

        float t0, t1;
        ray_sphere(X, to_light, PLANET_CENTER, ATMOSPHERE_RADIUS, t0, t1);
        float dist_L      = t1; // @Todo: Outer earth
        float step_size_L = dist_L / float(push.num_sun_samples + 1);
        float3 step_L     = step_size_L * to_light;
        float3 sample_L   = X + step_L;
        float ODr_L = 0;
        float ODm_L = 0;
        float ODo_L = 0;

        for (int j = 0; j < push.num_sun_samples; ++j) {
            float h_L = altitude_of(sample_L);

            ODr_L += exp(-h_L / Hr) * step_size_L;
            ODm_L += exp(-h_L / Hm) * step_size_L;
            ODo_L += max(0, 1 - (abs(h_L - 25e3) / 15e3));

            sample_L += step_L;
        }

        float3 tau = (Sr*(ODr + ODr_L)) + ((Sm + Am)*(ODm + ODm_L)) + (Ao*(ODo + ODo_L));
        float3 transmittance = exp(-tau);
        sum_r += transmittance * Dr;
        sum_m += transmittance * Dm;
        sum_o += transmittance * Do;

        X += dX;
    }

    float3 L = L0;
    float3 T = (sum_r*Sr*Pr) + (sum_m*Sm*Pm);
    L *= T;

    // Our light buffer being FLOAT16 is the reason why we are multiplying 
    // exposure here to prevent overflow.
    float N = 16;
    float EV100    = compute_EV100(N, 1.0/125.0, 100.0);
    float exposure = exposure_from_EV100(EV100);
    L *= exposure;

    // Lens
    L *= (QUARTER_PI / (N*N));

    return L;
}
