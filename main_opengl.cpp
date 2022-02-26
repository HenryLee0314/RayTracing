#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <random>
#include <algorithm>

// dear imgui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// If you are new to dear imgui, see examples/README.txt and documentation at the top of imgui.cpp.
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "log.h"
#include "opencl_manager.h"
#include "opencl_task.h"

// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! Here we are supporting a few common ones (gl3w, glew, glad).
//  You may use another loader/header of your choice (glext, glLoadGen, etc.), or chose to manually implement your own.
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>    // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>    // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING)
#define GLFW_INCLUDE_NONE         // GLFW including OpenGL headers causes ambiguity or multiple definition errors.
#include <glbinding/glbinding.h>  // Initialize with glbinding::initialize()
#include <glbinding/gl/gl.h>
using namespace gl;
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif


using namespace CGRA;

const std::string cubemap_top_path = "/Users/liyiheng/VUW-CGRA/PA/RayTracing/skybox/top.jpg";
const std::string cubemap_bottom_path = "/Users/liyiheng/VUW-CGRA/PA/RayTracing/skybox/bottom.jpg";
const std::string cubemap_left_path = "/Users/liyiheng/VUW-CGRA/PA/RayTracing/skybox/left.jpg";
const std::string cubemap_right_path = "/Users/liyiheng/VUW-CGRA/PA/RayTracing/skybox/right.jpg";
const std::string cubemap_front_path = "/Users/liyiheng/VUW-CGRA/PA/RayTracing/skybox/front.jpg";
const std::string cubemap_back_path = "/Users/liyiheng/VUW-CGRA/PA/RayTracing/skybox/back.jpg";
const std::string cl_file_path = "/Users/liyiheng/VUW-CGRA/PA/RayTracing/src/OpenCL/test.cl";
const int image_width = 640;
const int image_height = 480;


class Renderer : public OpenclTask
{
public:
    Renderer(const char* const fileAddress, int width, int height) : OpenclTask(fileAddress), _width(width), _height(height), times(0)
    {
        render_demo_A = false;

        k_render = clCreateKernel(program, "render", NULL);
        k_cubemap_demo = clCreateKernel(program, "demo_cubemap", NULL);
        /**Step 8: Initial input,output for the host and create memory objects for the kernel*/

        temp_color = (float*)malloc(3 * _width * _height * sizeof(float));
        final_color = (float*)malloc(3 * _width * _height * sizeof(float));
        random_number = (float*)malloc(3 * _width * _height * sizeof(float));

        memset(temp_color, 0, 3 * _width * _height * sizeof(float));
        memset(final_color, 0, 3 * _width * _height * sizeof(float));


        std::mt19937 generator(us_ticker_read());
        std::uniform_real_distribution<float> distribution(0.0, 1.0);

        for (int i = 0; i < 3 * _width * _height; i++) {
            random_number[i] = distribution(generator);
        }

        int cubemap_width, cubemap_height, nrChannels;
        _cubemap_top = stbi_load(cubemap_top_path.c_str(), &cubemap_width, &cubemap_height, &nrChannels, 0);
        _cubemap_bottom = stbi_load(cubemap_bottom_path.c_str(), &cubemap_width, &cubemap_height, &nrChannels, 0);
        _cubemap_left = stbi_load(cubemap_left_path.c_str(), &cubemap_width, &cubemap_height, &nrChannels, 0);
        _cubemap_right = stbi_load(cubemap_right_path.c_str(), &cubemap_width, &cubemap_height, &nrChannels, 0);
        _cubemap_front = stbi_load(cubemap_front_path.c_str(), &cubemap_width, &cubemap_height, &nrChannels, 0);
        _cubemap_back = stbi_load(cubemap_back_path.c_str(), &cubemap_width, &cubemap_height, &nrChannels, 0);

        // CGRA_LOGD("%d %d %d", cubemap_width, cubemap_height, nrChannels);

        _cl_mem_cubemap_top = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * cubemap_width * cubemap_height * sizeof(uint8_t), (void*)_cubemap_top, NULL);
        _cl_mem_cubemap_bottom = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * cubemap_width * cubemap_height * sizeof(uint8_t), (void*)_cubemap_bottom, NULL);
        _cl_mem_cubemap_left = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * cubemap_width * cubemap_height * sizeof(uint8_t), (void*)_cubemap_left, NULL);
        _cl_mem_cubemap_right = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * cubemap_width * cubemap_height * sizeof(uint8_t), (void*)_cubemap_right, NULL);
        _cl_mem_cubemap_front = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * cubemap_width * cubemap_height * sizeof(uint8_t), (void*)_cubemap_front, NULL);
        _cl_mem_cubemap_back = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * cubemap_width * cubemap_height * sizeof(uint8_t), (void*)_cubemap_back, NULL);
    }

    void step()
    {
        std::mt19937 generator(us_ticker_read());
        std::uniform_real_distribution<float> distribution(0.0, 1.0);

        for (int i = 0; i < 3 * _width * _height; i++) {
            random_number[i] = distribution(generator);
        }

        _cl_mem_random_number = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * _width * _height * sizeof(float), (void*)random_number, NULL);
        _cl_mem_image = clCreateBuffer(OpenclManager::getInstance()->getContent(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * _width * _height * sizeof(float), (void*)temp_color, NULL);

        if (render_demo_A) {
            /**Step 9: Sets Kernel arguments.*/
            clSetKernelArg(k_render, 0, sizeof(cl_mem), (void*)&_cl_mem_image);
            clSetKernelArg(k_render, 1, sizeof(int), (void*)&_width);
            clSetKernelArg(k_render, 2, sizeof(int), (void*)&_height);
            clSetKernelArg(k_render, 3, sizeof(cl_mem), (void*)&_cl_mem_random_number);

            /**Step 10: Running the kernel.*/
            size_t global_work_size[1] = {static_cast<size_t>(_width * _height)};
            _enentPoint = (cl_event*)malloc(1 * sizeof(cl_event));

            clEnqueueNDRangeKernel(OpenclManager::getInstance()->getCommandQueue(), k_render, 1, NULL, global_work_size, NULL, 0, NULL, _enentPoint);
        }
        else {
            /**Step 9: Sets Kernel arguments.*/
            clSetKernelArg(k_cubemap_demo, 0, sizeof(cl_mem), (void*)&_cl_mem_image);
            clSetKernelArg(k_cubemap_demo, 1, sizeof(int), (void*)&_width);
            clSetKernelArg(k_cubemap_demo, 2, sizeof(int), (void*)&_height);
            clSetKernelArg(k_cubemap_demo, 3, sizeof(cl_mem), (void*)&_cl_mem_random_number);
            clSetKernelArg(k_cubemap_demo, 4, sizeof(cl_mem), (void*)&_cl_mem_cubemap_top);
            clSetKernelArg(k_cubemap_demo, 5, sizeof(cl_mem), (void*)&_cl_mem_cubemap_bottom);
            clSetKernelArg(k_cubemap_demo, 6, sizeof(cl_mem), (void*)&_cl_mem_cubemap_left);
            clSetKernelArg(k_cubemap_demo, 7, sizeof(cl_mem), (void*)&_cl_mem_cubemap_right);
            clSetKernelArg(k_cubemap_demo, 8, sizeof(cl_mem), (void*)&_cl_mem_cubemap_front);
            clSetKernelArg(k_cubemap_demo, 9, sizeof(cl_mem), (void*)&_cl_mem_cubemap_back);

            /**Step 10: Running the kernel.*/
            size_t global_work_size[1] = {static_cast<size_t>(_width * _height)};
            _enentPoint = (cl_event*)malloc(1 * sizeof(cl_event));

            clEnqueueNDRangeKernel(OpenclManager::getInstance()->getCommandQueue(), k_cubemap_demo, 1, NULL, global_work_size, NULL, 0, NULL, _enentPoint);
        }


        clEnqueueWaitForEvents(OpenclManager::getInstance()->getCommandQueue(), 1, _enentPoint);
    }

    void wait()
    {

        memset(temp_color, 0, 3 * _width * _height * sizeof(float));

        clWaitForEvents(1, _enentPoint);
        clReleaseEvent(_enentPoint[0]);
        free(_enentPoint);

        // *Step 11: Read the cout put back to host memory.
        clEnqueueReadBuffer(OpenclManager::getInstance()->getCommandQueue(), _cl_mem_image, CL_TRUE, 0, 3 * _width * _height * sizeof(float), temp_color, 0, NULL, NULL);

        times++;

        clReleaseMemObject(_cl_mem_random_number);
        clReleaseMemObject(_cl_mem_image);
    }

    void change_render_scene()
    {
        times = 0;
        memset(temp_color, 0, 3 * _width * _height * sizeof(float));
        memset(final_color, 0, 3 * _width * _height * sizeof(float));
        render_demo_A = !render_demo_A;
    }

    ~Renderer()
    {
        free(temp_color);
        free(final_color);
        free(random_number);

        clReleaseKernel(k_render);
        clReleaseKernel(k_cubemap_demo);



        stbi_image_free(_cubemap_top);
        stbi_image_free(_cubemap_bottom);
        stbi_image_free(_cubemap_left);
        stbi_image_free(_cubemap_right);
        stbi_image_free(_cubemap_front);
        stbi_image_free(_cubemap_back);

        clReleaseMemObject(_cl_mem_cubemap_top);
        clReleaseMemObject(_cl_mem_cubemap_bottom);
        clReleaseMemObject(_cl_mem_cubemap_left);
        clReleaseMemObject(_cl_mem_cubemap_right);
        clReleaseMemObject(_cl_mem_cubemap_front);
        clReleaseMemObject(_cl_mem_cubemap_back);
    }

    bool render_demo_A;

    int _width;
    int _height;

    cl_kernel k_render;
    float* temp_color;
    cl_mem _cl_mem_image;

    uint32_t _size;
    cl_event* _enentPoint;

    float* final_color;
    int times;

    float* random_number;
    cl_mem _cl_mem_random_number;

    // cubemap
    uint8_t* _cubemap_top;
    uint8_t* _cubemap_bottom;
    uint8_t* _cubemap_left;
    uint8_t* _cubemap_right;
    uint8_t* _cubemap_front;
    uint8_t* _cubemap_back;

    cl_kernel k_cubemap_demo;

    cl_mem _cl_mem_cubemap_top;
    cl_mem _cl_mem_cubemap_bottom;
    cl_mem _cl_mem_cubemap_left;
    cl_mem _cl_mem_cubemap_right;
    cl_mem _cl_mem_cubemap_front;
    cl_mem _cl_mem_cubemap_back;

};



const char *vertexShaderSource = "#version 330 core\n"
                                 "layout (location = 0) in vec3 aPos;\n"
                                 "layout (location = 1) in vec2 aTexCoord;\n"
                                 "out vec2 TexCoord;\n"
                                 "void main()\n"
                                 "{\n"
                                 "    gl_Position = vec4(aPos, 1.0);\n"
                                 "    TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
                                 "}\n\0";

const char *fragmentShaderSource = "#version 330 core\n"
                                   "out vec4 FragColor;\n"
                                   "in vec2 TexCoord;\n"
                                   "uniform sampler2D texture1;\n"
                                   "void main()\n"
                                   "{\n"
                                   "	FragColor = texture(texture1, TexCoord);\n"
                                   "}\n\0";

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char**)
{


    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if __APPLE__
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(image_width, image_height, "RTRT", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING)
    bool err = false;
    glbinding::initialize([](const char* name) {
        return (glbinding::ProcAddress)glfwGetProcAddress(name);
    });
#else
    bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();



    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);



    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    float vertices[] = {
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f,  // top right
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f  // top left
    };
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    uint8_t* _data = (uint8_t*)malloc(image_height * image_width * 3 * sizeof(uint8_t));


    OpenclManager::getInstance();
    Renderer renderer_task(cl_file_path.c_str(), image_width, image_height);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        memset(_data, 0, image_height * image_width * 3 * sizeof(uint8_t));

        ImGui::Begin("Hello, world!");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        if (ImGui::Button("change scene")) {
            renderer_task.change_render_scene();
        }
        ImGui::End();

        // std::mt19937 generator(us_ticker_read());
        renderer_task.step();
        renderer_task.wait();

        for (int row = 0; row < image_height; row++) {
            for (int col = 0; col < image_width; col++) {


                renderer_task.final_color[(row * image_width + col)*3 + 0] += renderer_task.temp_color[(row * image_width + col)*3 + 0];
                renderer_task.final_color[(row * image_width + col)*3 + 1] += renderer_task.temp_color[(row * image_width + col)*3 + 1];
                renderer_task.final_color[(row * image_width + col)*3 + 2] += renderer_task.temp_color[(row * image_width + col)*3 + 2];

                float r = sqrt(renderer_task.final_color[(row * image_width + col)*3 + 0] / renderer_task.times);
                float g = sqrt(renderer_task.final_color[(row * image_width + col)*3 + 1] / renderer_task.times);
                float b = sqrt(renderer_task.final_color[(row * image_width + col)*3 + 2] / renderer_task.times);

                r = std::clamp(r, 0.0f, 1.0f);
                g = std::clamp(g, 0.0f, 1.0f);
                b = std::clamp(b, 0.0f, 1.0f);

                _data[(row * image_width + col)*3 + 0] = r * 255;
                _data[(row * image_width + col)*3 + 1] = g * 255;
                _data[(row * image_width + col)*3 + 2] = b * 255;
            }
        }

        unsigned int texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // load image, create texture and generate mipmaps
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, _data);
        glGenerateMipmap(GL_TEXTURE_2D);

        // if (show_demo_window)
        //     ImGui::ShowDemoWindow(&show_demo_window);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);


        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        // glBindVertexArray(0); // no need to unbind it every time
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        glDeleteTextures(1, &texture);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}



