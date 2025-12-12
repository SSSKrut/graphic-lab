#include <GLFW/glfw3.h>
#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "stb_image_write.h"
#include "tiny_obj_loader.h"

struct Material {
  glm::vec3 diffuse{0.8f};
  glm::vec3 specular{0.0f};
  glm::vec3 emission{0.0f};
  bool isLight = false;
};

struct Triangle {
  glm::vec3 v0, v1, v2;
  glm::vec3 n0, n1, n2;
  int materialId = -1;
};

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct HitInfo {
  bool hit = false;
  float t = 1e30f;
  glm::vec3 position;
  glm::vec3 normal;
  int materialId = -1;
};

class Scene {
public:
  std::vector<Triangle> triangles;
  std::vector<Material> materials;
  std::vector<int> lightIndices;

  RTCDevice device = nullptr;
  RTCScene rtcScene = nullptr;

  ~Scene() {
    if (rtcScene)
      rtcReleaseScene(rtcScene);
    if (device)
      rtcReleaseDevice(device);
  }

  bool load(const std::string &filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err,
                          filename.c_str())) {
      return false;
    }

    for (auto &m : objMaterials) {
      Material mat;
      mat.diffuse = glm::vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
      mat.specular = glm::vec3(m.specular[0], m.specular[1], m.specular[2]);
      mat.emission = glm::vec3(m.emission[0], m.emission[1], m.emission[2]);

      if (m.name == "Light") {
        mat.emission = glm::vec3(15.0f);
        mat.isLight = true;
      }

      float specSum = mat.specular.x + mat.specular.y + mat.specular.z;
      float diffSum = mat.diffuse.x + mat.diffuse.y + mat.diffuse.z;
      if (specSum + diffSum > 3.0f) {
        float scale = 3.0f / (specSum + diffSum);
        mat.specular *= scale;
        mat.diffuse *= scale;
      }

      materials.push_back(mat);
    }

    if (materials.empty()) {
      Material defaultMat;
      materials.push_back(defaultMat);
    }

    for (auto &shape : shapes) {
      for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
        Triangle tri;

        auto &idx0 = shape.mesh.indices[i + 0];
        auto &idx1 = shape.mesh.indices[i + 1];
        auto &idx2 = shape.mesh.indices[i + 2];

        tri.v0 = glm::vec3(attrib.vertices[3 * idx0.vertex_index + 0],
                           attrib.vertices[3 * idx0.vertex_index + 1],
                           attrib.vertices[3 * idx0.vertex_index + 2]);
        tri.v1 = glm::vec3(attrib.vertices[3 * idx1.vertex_index + 0],
                           attrib.vertices[3 * idx1.vertex_index + 1],
                           attrib.vertices[3 * idx1.vertex_index + 2]);
        tri.v2 = glm::vec3(attrib.vertices[3 * idx2.vertex_index + 0],
                           attrib.vertices[3 * idx2.vertex_index + 1],
                           attrib.vertices[3 * idx2.vertex_index + 2]);

        glm::vec3 faceNormal =
            glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));

        if (idx0.normal_index >= 0) {
          tri.n0 = glm::vec3(attrib.normals[3 * idx0.normal_index + 0],
                             attrib.normals[3 * idx0.normal_index + 1],
                             attrib.normals[3 * idx0.normal_index + 2]);
        } else
          tri.n0 = faceNormal;

        if (idx1.normal_index >= 0) {
          tri.n1 = glm::vec3(attrib.normals[3 * idx1.normal_index + 0],
                             attrib.normals[3 * idx1.normal_index + 1],
                             attrib.normals[3 * idx1.normal_index + 2]);
        } else
          tri.n1 = faceNormal;

        if (idx2.normal_index >= 0) {
          tri.n2 = glm::vec3(attrib.normals[3 * idx2.normal_index + 0],
                             attrib.normals[3 * idx2.normal_index + 1],
                             attrib.normals[3 * idx2.normal_index + 2]);
        } else
          tri.n2 = faceNormal;

        int matId = 0;
        if (!shape.mesh.material_ids.empty()) {
          int idx = (int)(i / 3);
          if (idx < (int)shape.mesh.material_ids.size()) {
            matId = shape.mesh.material_ids[idx];
            if (matId < 0)
              matId = 0;
          }
        }
        tri.materialId = matId;

        if (matId < (int)materials.size() && materials[matId].isLight) {
          lightIndices.push_back((int)triangles.size());
        }

        triangles.push_back(tri);
      }
    }

    buildEmbreeScene();
    return true;
  }

  void buildEmbreeScene() {
    device = rtcNewDevice(nullptr);
    rtcScene = rtcNewScene(device);

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float *vb = (float *)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float),
        triangles.size() * 3);
    unsigned *ib = (unsigned *)rtcSetNewGeometryBuffer(
        geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned),
        triangles.size());

    for (size_t i = 0; i < triangles.size(); i++) {
      vb[9 * i + 0] = triangles[i].v0.x;
      vb[9 * i + 1] = triangles[i].v0.y;
      vb[9 * i + 2] = triangles[i].v0.z;
      vb[9 * i + 3] = triangles[i].v1.x;
      vb[9 * i + 4] = triangles[i].v1.y;
      vb[9 * i + 5] = triangles[i].v1.z;
      vb[9 * i + 6] = triangles[i].v2.x;
      vb[9 * i + 7] = triangles[i].v2.y;
      vb[9 * i + 8] = triangles[i].v2.z;
      ib[3 * i + 0] = (unsigned)(3 * i + 0);
      ib[3 * i + 1] = (unsigned)(3 * i + 1);
      ib[3 * i + 2] = (unsigned)(3 * i + 2);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(rtcScene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(rtcScene);
  }

  HitInfo intersect(const Ray &ray) const {
    HitInfo info;
    RTCRayHit rayhit;
    rayhit.ray.org_x = ray.origin.x;
    rayhit.ray.org_y = ray.origin.y;
    rayhit.ray.org_z = ray.origin.z;
    rayhit.ray.dir_x = ray.direction.x;
    rayhit.ray.dir_y = ray.direction.y;
    rayhit.ray.dir_z = ray.direction.z;
    rayhit.ray.tnear = 0.001f;
    rayhit.ray.tfar = 1e30f;
    rayhit.ray.mask = (unsigned)-1;
    rayhit.ray.flags = 0;
    rayhit.ray.time = 0.0f;
    rayhit.ray.id = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(rtcScene, &rayhit);

    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
      info.hit = true;
      info.t = rayhit.ray.tfar;
      info.position = ray.origin + ray.direction * info.t;

      unsigned primID = rayhit.hit.primID;
      float u = rayhit.hit.u;
      float v = rayhit.hit.v;
      float w = 1.0f - u - v;

      const Triangle &tri = triangles[primID];
      info.normal = glm::normalize(tri.n0 * w + tri.n1 * u + tri.n2 * v);
      info.materialId = tri.materialId;
    }
    return info;
  }

  float triangleArea(int idx) const {
    const Triangle &t = triangles[idx];
    return 0.5f * glm::length(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
  }

  glm::vec3 sampleTriangle(int idx, float u, float v) const {
    const Triangle &t = triangles[idx];
    if (u + v > 1.0f) {
      u = 1.0f - u;
      v = 1.0f - v;
    }
    return t.v0 * (1.0f - u - v) + t.v1 * u + t.v2 * v;
  }

  glm::vec3 getTriangleNormal(int idx) const {
    const Triangle &t = triangles[idx];
    return glm::normalize(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
  }
};

class PathTracer {
public:
  Scene scene;

  glm::vec3 cameraPos{0.0f, 2.5f, 10.0f};
  glm::vec3 cameraLookAt{0.0f, 2.5f, 0.0f};
  float fov = 45.0f;

  int width = 512;
  int height = 512;
  int spp = 64;
  int maxDepth = 5;
  float gamma = 2.2f;

  std::vector<glm::vec3> accumBuffer;
  std::vector<unsigned char> displayBuffer;
  std::atomic<int> currentSample{0};
  std::atomic<bool> rendering{false};
  std::atomic<bool> stopRequested{false};
  std::mutex bufferMutex;

  GLuint texture = 0;
  std::thread renderThread;

  ~PathTracer() { stopRendering(); }

  bool loadScene(const std::string &path) { return scene.load(path); }

  void startRendering() {
    stopRendering();

    accumBuffer.assign(width * height, glm::vec3(0.0f));
    displayBuffer.assign(width * height * 3, 0);
    currentSample = 0;
    stopRequested = false;
    rendering = true;

    renderThread = std::thread([this]() {
      std::vector<std::mt19937> rngs(std::thread::hardware_concurrency());
      for (size_t i = 0; i < rngs.size(); i++) {
        rngs[i].seed((unsigned)(i + 1) * 12345);
      }

      for (int s = 0; s < spp && !stopRequested; s++) {

#pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; y++) {
          if (stopRequested)
            continue;

          unsigned tid = 0;
#ifdef _OPENMP
          tid = omp_get_thread_num();
#endif
          std::mt19937 &rng = rngs[tid % rngs.size()];
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);

          std::vector<glm::vec3> rowBuffer(width);
          for (int x = 0; x < width; x++) {
            Ray ray = generateRay(x, y, dist(rng), dist(rng));
            rowBuffer[x] = trace(ray, rng, 0);
          }

#pragma omp critical
          {
            for (int x = 0; x < width; x++) {
              int idx = y * width + x;
              accumBuffer[idx] += rowBuffer[x];
            }
          }
        }
        currentSample++;
        updateDisplayBuffer();
      }
      rendering = false;
    });
  }

  void stopRendering() {
    stopRequested = true;
    if (renderThread.joinable()) {
      renderThread.join();
    }
  }

  Ray generateRay(int x, int y, float jx, float jy) {
    float aspect = (float)width / height;
    float tanHalfFov = std::tan(glm::radians(fov) * 0.5f);

    float px = (2.0f * (x + jx) / width - 1.0f) * aspect * tanHalfFov;
    float py = (1.0f - 2.0f * (y + jy) / height) * tanHalfFov;

    glm::vec3 forward = glm::normalize(cameraLookAt - cameraPos);
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    glm::vec3 up = glm::cross(right, forward);

    glm::vec3 dir = glm::normalize(forward + right * px + up * py);
    return {cameraPos, dir};
  }

  glm::vec3 trace(const Ray &ray, std::mt19937 &rng, int depth) {
    if (depth >= maxDepth)
      return glm::vec3(0.0f);

    HitInfo hit = scene.intersect(ray);
    if (!hit.hit)
      return glm::vec3(0.0f);

    const Material &mat = scene.materials[hit.materialId];

    if (mat.isLight) {
      return mat.emission;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    glm::vec3 result(0.0f);

    result += sampleDirectLight(hit, rng);

    float diffMax = std::max({mat.diffuse.x, mat.diffuse.y, mat.diffuse.z});
    float specMax = std::max({mat.specular.x, mat.specular.y, mat.specular.z});
    float total = diffMax + specMax;

    if (total < 0.001f)
      return result;

    float rr = dist(rng);
    if (rr > total)
      return result;

    float pDiff = diffMax / total;

    if (dist(rng) < pDiff) {
      glm::vec3 newDir = sampleHemisphereCosine(hit.normal, rng);
      Ray newRay{hit.position + hit.normal * 0.001f, newDir};
      glm::vec3 indirect = trace(newRay, rng, depth + 1);
      result += mat.diffuse * indirect / total;
    } else {
      glm::vec3 reflected = glm::reflect(ray.direction, hit.normal);
      Ray newRay{hit.position + hit.normal * 0.001f, reflected};
      glm::vec3 indirect = trace(newRay, rng, depth + 1);
      result += mat.specular * indirect / total;
    }

    return result;
  }

  glm::vec3 sampleDirectLight(const HitInfo &hit, std::mt19937 &rng) {
    if (scene.lightIndices.empty())
      return glm::vec3(0.0f);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int lightIdx = scene.lightIndices[rng() % scene.lightIndices.size()];
    glm::vec3 lightPoint = scene.sampleTriangle(lightIdx, dist(rng), dist(rng));
    glm::vec3 lightNormal = scene.getTriangleNormal(lightIdx);

    glm::vec3 toLight = lightPoint - hit.position;
    float dist2 = glm::dot(toLight, toLight);
    float distance = std::sqrt(dist2);
    glm::vec3 L = toLight / distance;

    float cosTheta = glm::dot(hit.normal, L);
    if (cosTheta <= 0.0f)
      return glm::vec3(0.0f);

    float cosLight = -glm::dot(lightNormal, L);
    if (cosLight <= 0.0f)
      return glm::vec3(0.0f);

    Ray shadowRay{hit.position + hit.normal * 0.001f, L};
    HitInfo shadowHit = scene.intersect(shadowRay);

    if (!shadowHit.hit || shadowHit.t < distance - 0.01f)
      return glm::vec3(0.0f);

    const Material &mat = scene.materials[hit.materialId];
    const Material &lightMat =
        scene.materials[scene.triangles[lightIdx].materialId];

    float area = scene.triangleArea(lightIdx);
    float pdf = dist2 / (cosLight * area * scene.lightIndices.size());

    glm::vec3 brdf = mat.diffuse / glm::pi<float>();
    return lightMat.emission * brdf * cosTheta / pdf;
  }

  glm::vec3 sampleHemisphereCosine(const glm::vec3 &n, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u1 = dist(rng), u2 = dist(rng);

    float r = std::sqrt(u1);
    float theta = 2.0f * glm::pi<float>() * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(1.0f - u1);

    glm::vec3 t =
        (std::abs(n.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 u = glm::normalize(glm::cross(t, n));
    glm::vec3 v = glm::cross(n, u);

    return glm::normalize(u * x + v * y + n * z);
  }

  void updateDisplayBuffer() {
    std::lock_guard<std::mutex> lock(bufferMutex);
    int samples = std::max(1, currentSample.load());

    float maxLum = 0.0f;
#pragma omp parallel for reduction(max : maxLum)
    for (int i = 0; i < width * height; i++) {
      glm::vec3 c = accumBuffer[i] / (float)samples;
      float lum = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
      maxLum = std::max(maxLum, lum);
    }

    float scale = maxLum > 0.0f ? 1.0f / maxLum : 1.0f;
    scale = std::min(scale, 2.0f);

#pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
      glm::vec3 c = accumBuffer[i] / (float)samples * scale;
      c = glm::clamp(c, 0.0f, 1.0f);
      c = glm::pow(c, glm::vec3(1.0f / gamma));

      displayBuffer[3 * i + 0] = (unsigned char)(c.r * 255);
      displayBuffer[3 * i + 1] = (unsigned char)(c.g * 255);
      displayBuffer[3 * i + 2] = (unsigned char)(c.b * 255);
    }
  }

  void saveImage(const std::string &filename) {
    updateDisplayBuffer();
    stbi_write_ppm(filename.c_str(), width, height, 3, displayBuffer.data());
  }

  void updateTexture() {
    if (texture == 0) {
      glGenTextures(1, &texture);
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    std::lock_guard<std::mutex> lock(bufferMutex);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, displayBuffer.data());
  }
};

int main() {
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(1280, 720, "Path Tracer", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  PathTracer pt;
  if (!pt.loadScene("../assets/cornell-box.obj")) {
    fprintf(stderr, "Failed to load scene!\\n");
    return -1;
  }

  float camPos[3] = {0.0f, 2.5f, 10.0f};
  float lookAt[3] = {0.0f, 2.5f, -3.0f};
  int resolution[2] = {512, 512};
  char savePath[256] = "output/render.ppm";

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Settings");

    ImGui::Text("Camera");
    ImGui::DragFloat3("Position", camPos, 0.1f);
    ImGui::DragFloat3("Look At", lookAt, 0.1f);
    ImGui::SliderFloat("FOV", &pt.fov, 10.0f, 120.0f);

    ImGui::Separator();
    ImGui::Text("Render");
    ImGui::SliderInt("SPP", &pt.spp, 1, 1024);
    ImGui::SliderInt("Max Depth", &pt.maxDepth, 1, 16);
    ImGui::SliderFloat("Gamma", &pt.gamma, 1.0f, 3.0f);
    ImGui::SliderInt2("Resolution", resolution, 500, 1000);

    ImGui::Separator();

    bool isRendering = pt.rendering.load();

    if (isRendering) {
      ImGui::Text("Rendering: %d / %d SPP", pt.currentSample.load(), pt.spp);
      if (ImGui::Button("Stop")) {
        pt.stopRendering();
      }
    } else {
      if (ImGui::Button("Start Render")) {
        pt.cameraPos = glm::vec3(camPos[0], camPos[1], camPos[2]);
        pt.cameraLookAt = glm::vec3(lookAt[0], lookAt[1], lookAt[2]);
        pt.width = resolution[0];
        pt.height = resolution[1];
        pt.startRendering();
      }
    }

    ImGui::Separator();
    ImGui::InputText("Save Path", savePath, sizeof(savePath));
    if (ImGui::Button("Save to PPM")) {
      pt.saveImage(savePath);
    }

    ImGui::End();

    ImGui::Begin("Render Output");
    if (pt.currentSample > 0) {
      pt.updateTexture();
      ImVec2 size((float)pt.width, (float)pt.height);
      ImGui::Image((ImTextureID)(intptr_t)pt.texture, size);
    } else {
      ImGui::Text("Press 'Start Render' to begin");
    }
    ImGui::End();

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
