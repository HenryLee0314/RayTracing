
struct Ray{
    float3 origin;
    float3 dir;
    float3 weight;
};

float3 RayAt(const struct Ray r, float t)
{
    return r.origin + t * r.dir;
}

struct HitRecord {
    float3 pos;
    float3 normal;
    float t;
};

struct Sphere{
    float radius;
    float3 pos;
    float3 color;
    bool is_light;
};

struct Camera {
    float3 pos;
    float3 look_at;
    float left_botton_corner;
};

struct Ray getRay(struct Camera camera, int width, int height, __global float* random_buffer)
{
    const int index = get_global_id(0);
    const float3 _UP_RIGHT_  = (float3)(0.0, 1.0, 0.0);
    float random_x = random_buffer[3 * get_global_id(0) + 0];
    float random_y = random_buffer[3 * get_global_id(0) + 1];

    int x = index % width;
    int y = index / width;
    float fx = (float)x + (random_x - 0.5);
    float fy = (float)y + (random_y - 0.5);
    fx = fx / (float)width;
    fy = fy / (float)height;

    

    float aspect_ratio = (float)(width) / (float)(height);
    fx = (fx - 0.5f) * aspect_ratio;
    fy = fy - 0.5f;

    float3 front = normalize(camera.look_at - camera.pos);
    float3 left = cross(_UP_RIGHT_, front);
    float3 top = cross(front, left);

    // float3 pixel_pos = (float3)(2 * fx + camera.look_at.x, 2 * fy + camera.look_at.y, camera.look_at.z);
    float3 pixel_pos = camera.pos + front + 2 * fx * left + 2 * fy * top;
    struct Ray ray;
    ray.origin = camera.pos;
    ray.dir = normalize(pixel_pos - camera.pos);
    ray.weight = (float3)(1.0,1.0,1.0);
    // ray.weight = (float3)(0.0,0.0,0.0);
    return ray;
}



bool hit_sphere(const struct Ray r, const struct Sphere sphere, const float t_min, const float t_max, struct HitRecord* record)
{
    float3 oc = r.origin - sphere.pos;
    float a = dot(r.dir, r.dir);
    float b = 2.0 * dot(oc, r.dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b*b - 4*a*c;
    if (discriminant > 0) {
        float root = sqrt(discriminant);
		float temp = (- b - root) / (2 * a);
		if (temp < t_max && temp > t_min) {
            record->pos = RayAt(r, temp);
            record->t = temp;
            record->normal = normalize(record->pos - sphere.pos);
            return true;
        }
        temp = (- b + root) / (2 * a);
        if (temp < t_max && temp > t_min) {
            record->pos = RayAt(r, temp);
			record->t = temp;
            record->normal = normalize(record->pos - sphere.pos);
            return true;
        }
    }
    return false;
}

float3 reflect(const float3 v, const float3 n) {
	return normalize(v - 2 * dot(v, n) * n);
}

float3 diffuse(const float3 n,  __global float* random_buffer) {
    const int index = get_global_id(0);
    float x = random_buffer[3 * index + 0] - 0.5;
    float y = random_buffer[3 * index + 1] - 0.5;
    float z = random_buffer[3 * index + 2] - 0.5;

    float3 ret = normalize((float3)(x,y,z));
    if (dot(n, ret) > 0) {
        return ret;
    }
    else {
        return -ret;
    }
}

bool ray_hit_scene(const struct Sphere* sphere, const struct Ray ray, struct HitRecord* record, struct Ray* new_ray,  __global float* random_buffer, float3* out_color)
{
    struct HitRecord temp_record;
    bool hit_anything = false;
    float closest = 9999;
    int sphere_index;
    for (int i = 0; i < 9; i++) {
        if (hit_sphere(ray, sphere[i], 0.001, closest, &temp_record)) {
            hit_anything = true;
            closest = temp_record.t;
            (*record) = temp_record;
            sphere_index = i;
        }
    }

    if (hit_anything) {
        const int index = get_global_id(0);
        float random_number = random_buffer[3 *index];
        const float P_RR = 0.9;
        if (random_number > P_RR) {
            new_ray->weight = (float3)(0.0, 0.0, 0.0);
            // no use
            new_ray->origin = ray.origin;
            new_ray->dir = ray.dir;
            (*out_color) = (float3)(0.0, 0.0, 0.0);
        }
        else {
            if (sphere[sphere_index].is_light) {
                // BRDF of light
                
                new_ray->origin = ray.origin;
                new_ray->dir = ray.dir;
                new_ray->weight = (float3)(0.0, 0.0, 0.0);//ray.weight * sphere[sphere_index].color * dot(record->normal, new_ray->dir) * 2.0f * 3.14159f / P_RR;

                (*out_color) += ray.weight * sphere[sphere_index].color / P_RR;
                
            }
            else {
            // BRDF of other
            if (sphere_index == 1) {
                new_ray->origin = record->pos;
                new_ray->dir = reflect(ray.dir, record->normal);
                new_ray->weight = ray.weight * dot(record->normal, new_ray->dir) / P_RR; // BRDF * cos(theta) / PDF(1) / P_RR
            }
            else {
                    new_ray->origin = record->pos;
                    new_ray->dir = diffuse(record->normal, random_buffer);
                    new_ray->weight = ray.weight * sphere[sphere_index].color * dot(record->normal, new_ray->dir) / P_RR * (2.0f * 3.14159f); // BRDF (color) * cos(theta) / PDF (1/(2PI)) / P_RR
                }
            }
        }

        
    }
    return hit_anything;
}

__kernel void render(__global float *image, int width, int height, __global float *random_buffer)
{
    // camera setting
    struct Camera camera;
    camera.pos = (float3)(-1.5, 0.0, -1.0);
    camera.look_at = (float3)(-1.5, 0.0, 0.0);

    // spheres
    struct Sphere sphere[9];
    sphere[0].pos = (float3)(-1.4, 1.5, 0.2);
    sphere[0].radius = 0.5;
    sphere[0].color = (float3)(1.0, 1.0, 1.0);
    sphere[0].is_light = true;

    sphere[1].pos = (float3)(-1.3, 0.0, 0.2);
    sphere[1].radius = 0.5;
    sphere[1].color = (float3)(1.0, 1.0, 1.0);
    sphere[1].is_light = false;

    sphere[2].pos = (float3)(0.0, -10002.0, 0.0);
    sphere[2].radius = 10000;
    sphere[2].color = (float3)(0.9, 0.9, 0.9);
    sphere[2].is_light = false;

    sphere[3].pos = (float3)(-10002.0, 0.0, 0.0);
    sphere[3].radius = 10000;
    sphere[3].color = (float3)(1.0, 0.0, 0.0);
    sphere[3].is_light = false;

    sphere[4].pos = (float3)(10002.0, 0.0, 0.0);
    sphere[4].radius = 10000;
    sphere[4].color = (float3)(0.0, 0.0, 1.0);
    sphere[4].is_light = false;

    sphere[5].pos = (float3)(0.0, 10002.0, 0.0);
    sphere[5].radius = 10000;
    sphere[5].color = (float3)(0.9, 0.9, 0.9);
    sphere[5].is_light = false;

    sphere[6].pos = (float3)(0.0, 0.0, -10002.0);
    sphere[6].radius = 10000;
    sphere[6].color = (float3)(1.0, 1.0, 0.0);
    sphere[6].is_light = false;

    sphere[7].pos = (float3)(0.0, 0.0, 10002.0);
    sphere[7].radius = 10000;
    sphere[7].color = (float3)(0.0, 1.0, 0.0);
    sphere[7].is_light = false;

    sphere[8].pos = (float3)(0, 0.0, 1.5);
    sphere[8].radius = 0.5;
    sphere[8].color = (float3)(0.0, 1.0, 1.0);
    sphere[8].is_light = false;


    // cl thread
    const int index = get_global_id(0);
    struct Ray ray = getRay(camera, width, height, random_buffer);


    bool hit_anything = false;
    float3 color = (float3)(0,0,0);
    for (int i = 0; i < 40; i++) {
        struct HitRecord record;
        struct Ray new_ray;
        bool hit_scene = ray_hit_scene(sphere, ray, &record, &new_ray, random_buffer, &color);
        if (hit_scene) {
            hit_anything = true;
            ray = new_ray;
        }
    }

    // to image
    if (hit_anything) {
        image[3 * index + 0] = color.x;
        image[3 * index + 1] = color.y;
        image[3 * index + 2] = color.z;
    }
    else {
        image[3 * index + 0] = 0.0;
        image[3 * index + 1] = 0.0;
        image[3 * index + 2] = 0.0;
    }

}

void get_cubemap_light(float3* result, const struct Ray ray, __global uchar* top, __global uchar* bottom, __global uchar* left, __global uchar* right, __global uchar* front, __global uchar* back)
{
    const int index = get_global_id(0);

    const float _PI_ = 3.1415926;
    const float _PI_2_ = _PI_ / 2;
    const float _PI_4_ = _PI_ / 4;
    float theta = atan2(ray.dir.x, ray.dir.z);
    float phi = atan2(ray.dir.y, sqrt(ray.dir.x * ray.dir.x + ray.dir.z * ray.dir.z));
    float normal_theta = 0;

    if (theta > -_PI_4_ && theta < _PI_4_) {
        float ratio = 1024.0 / ray.dir.z;
        float x = ratio * ray.dir.x;
        float y = -ratio * ray.dir.y;
        int px = (int)(x) + 1024;
        int py = (int)(y) + 1024;

        normal_theta = theta;

        result->x = float(front[3 * (py * 2048 + px) + 0]) / 255.0;
        result->y = float(front[3 * (py * 2048 + px) + 1]) / 255.0;
        result->z = float(front[3 * (py * 2048 + px) + 2]) / 255.0;
    }
    else if (theta > _PI_4_ && theta < _PI_2_ + _PI_4_) {
        float ratio = 1024.0 / ray.dir.x;
        float x = ratio * -ray.dir.z;
        float y = -ratio * ray.dir.y;
        int px = (int)(x) + 1024;
        int py = (int)(y) + 1024;

        normal_theta = theta - _PI_2_;

        result->x = float(right[3 * (py * 2048 + px) + 0]) / 255.0;
        result->y = float(right[3 * (py * 2048 + px) + 1]) / 255.0;
        result->z = float(right[3 * (py * 2048 + px) + 2]) / 255.0;
    }
    else if (theta > - (_PI_2_ + _PI_4_) && theta < -_PI_4_) {
        float ratio = 1024.0 / -ray.dir.x;
        float x = ratio * ray.dir.z;
        float y = -ratio * ray.dir.y;
        int px = (int)(x) + 1024;
        int py = (int)(y) + 1024;

        normal_theta = theta + _PI_2_;

        result->x = float(left[3 * (py * 2048 + px) + 0]) / 255.0;
        result->y = float(left[3 * (py * 2048 + px) + 1]) / 255.0;
        result->z = float(left[3 * (py * 2048 + px) + 2]) / 255.0;
    }
    else if ((theta < - (_PI_2_ + _PI_4_) || theta > (_PI_2_ + _PI_4_))) {
        float ratio = 1024.0 / -ray.dir.z;
        float x = ratio * -ray.dir.x;
        float y = -ratio * ray.dir.y;
        int px = (int)(x) + 1024;
        int py = (int)(y) + 1024;

        if (theta > 0) {
            normal_theta = theta - _PI_;
        }
        else {
            normal_theta = theta + _PI_;
        }

        result->x = float(back[3 * (py * 2048 + px) + 0]) / 255.0;
        result->y = float(back[3 * (py * 2048 + px) + 1]) / 255.0;
        result->z = float(back[3 * (py * 2048 + px) + 2]) / 255.0;
    }
    else {

    }

    float phiThreshold = atan2(1.0f, 1.0f / cos(normal_theta));

    if (phi > phiThreshold) {
        float ratio = 1024.0 / ray.dir.y;
        float x = ratio * ray.dir.x;
        float y = -ratio * -ray.dir.z;
        int px = (int)(x) + 1024;
        int py = (int)(y) + 1024;

        result->x = float(top[3 * (py * 2048 + px) + 0]) / 255.0;
        result->y = float(top[3 * (py * 2048 + px) + 1]) / 255.0;
        result->z = float(top[3 * (py * 2048 + px) + 2]) / 255.0;
    }
    else if (phi < -phiThreshold) {
        float ratio = 1024.0 / -ray.dir.y;
        float x = ratio * ray.dir.x;
        float y = -ratio * ray.dir.z;
        int px = (int)(x) + 1024;
        int py = (int)(y) + 1024;

        result->x = float(bottom[3 * (py * 2048 + px) + 0]) / 255.0;
        result->y = float(bottom[3 * (py * 2048 + px) + 1]) / 255.0;
        result->z = float(bottom[3 * (py * 2048 + px) + 2]) / 255.0;
    }
    else {

    }
}

void ray_hit_scene_2(const struct Sphere* sphere,
                     const struct Ray ray, 
                     struct HitRecord* record, 
                     struct Ray* new_ray, 
                     __global float* random_buffer, 
                     float3* out_color,
                     __global uchar* top, 
                     __global uchar* bottom, 
                     __global uchar* left,
                     __global uchar* right, 
                     __global uchar* front, 
                     __global uchar* back)
{
    struct HitRecord temp_record;
    bool hit_anything = false;
    float closest = 9999;
    int sphere_index;
    for (int i = 0; i < 2; i++) {
        if (hit_sphere(ray, sphere[i], 0.001, closest, &temp_record)) {
            hit_anything = true;
            closest = temp_record.t;
            (*record) = temp_record;
            sphere_index = i;
        }
    }

    if (hit_anything) {
        const int index = get_global_id(0);
        float random_number = random_buffer[3 *index];
        const float P_RR = 0.9;
        if (random_number > P_RR) {
            new_ray->weight = (float3)(0.0, 0.0, 0.0);
            // no use
            new_ray->origin = ray.origin;
            new_ray->dir = ray.dir;
            (*out_color) = (float3)(0.0, 0.0, 0.0);
        }
        else {
            new_ray->origin = record->pos;
            new_ray->dir = reflect(ray.dir, record->normal);
            new_ray->weight = ray.weight * dot(record->normal, new_ray->dir) / P_RR; // BRDF * cos(theta) / PDF(1) / P_RR
        }
    }
    else {
        // no hit, return background
        new_ray->origin = ray.origin;
        new_ray->dir = ray.dir;
        new_ray->weight = (float3)(0,0,0);
        float3 out_background;
        get_cubemap_light(&out_background, ray, top, bottom, left, right, front, back);
        (*out_color) += ray.weight * out_background;
    }
}

__kernel void demo_cubemap(__global float *image, int width, int height, __global float *random_buffer, __global uchar* top, __global uchar* bottom, __global uchar* left, __global uchar* right, __global uchar* front, __global uchar* back)
{
    struct Camera camera;
    camera.pos = (float3)(0.0, 0.0, 0.0);
    camera.look_at = (float3)(0.0, 0.0, 1);

    const int index = get_global_id(0);
    struct Ray ray = getRay(camera, width, height, random_buffer);

    struct Sphere sphere[2];
    sphere[0].pos = (float3)(0.0, 0.0, 2);
    sphere[0].radius = 0.5;
    sphere[0].color = (float3)(1.0, 1.0, 1.0);
    sphere[0].is_light = false;

    sphere[1].pos = (float3)(-1.5, 0.0, 2);
    sphere[1].radius = 0.5;
    sphere[1].color = (float3)(1.0, 1.0, 1.0);
    sphere[1].is_light = false;


    float3 color = (float3)(0,0,0);
    for (int i = 0; i < 40; i++) {
        struct HitRecord record;
        struct Ray new_ray;
        ray_hit_scene_2(sphere, ray, &record, &new_ray, random_buffer, &color, top, bottom, left, right, front, back);
        ray = new_ray;
    }


    image[3 * index + 0] = color.x;
    image[3 * index + 1] = color.y;
    image[3 * index + 2] = color.z;

}
