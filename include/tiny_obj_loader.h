#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#define TINY_OBJ_LOADER_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tinyobj {

struct material_t {
  std::string name;
  float ambient[3];
  float diffuse[3];
  float specular[3];
  float transmittance[3];
  float emission[3];
  float shininess;
  float ior;
  float dissolve;
  int illum;
  std::string ambient_texname;
  std::string diffuse_texname;
  std::string specular_texname;
  std::string specular_highlight_texname;
  std::string bump_texname;
  std::string displacement_texname;
  std::string alpha_texname;
  std::map<std::string, std::string> unknown_parameter;

  material_t() {
    for (int i = 0; i < 3; i++) {
      ambient[i] = 0.f;
      diffuse[i] = 0.f;
      specular[i] = 0.f;
      transmittance[i] = 0.f;
      emission[i] = 0.f;
    }
    shininess = 1.f;
    ior = 1.f;
    dissolve = 1.f;
    illum = 0;
  }
};

struct index_t {
  int vertex_index;
  int normal_index;
  int texcoord_index;
};

struct mesh_t {
  std::vector<index_t> indices;
  std::vector<unsigned char> num_face_vertices;
  std::vector<int> material_ids;
  std::vector<unsigned int> smoothing_group_ids;
};

struct shape_t {
  std::string name;
  mesh_t mesh;
};

struct attrib_t {
  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<float> texcoords;
  std::vector<float> colors;
};

class MaterialReader {
public:
  virtual ~MaterialReader() {}
  virtual bool operator()(const std::string &matId,
                          std::vector<material_t> *materials,
                          std::map<std::string, int> *matMap, std::string *warn,
                          std::string *err) = 0;
};

class MaterialFileReader : public MaterialReader {
public:
  explicit MaterialFileReader(const std::string &mtl_basedir)
      : m_mtlBaseDir(mtl_basedir) {}
  virtual ~MaterialFileReader() {}
  virtual bool operator()(const std::string &matId,
                          std::vector<material_t> *materials,
                          std::map<std::string, int> *matMap, std::string *warn,
                          std::string *err) override {
    std::string filepath;
    if (!m_mtlBaseDir.empty()) {
      filepath = m_mtlBaseDir + matId;
    } else {
      filepath = matId;
    }
    std::ifstream matIStream(filepath.c_str());
    if (!matIStream) {
      if (err)
        *err = "Cannot open file: " + filepath;
      return false;
    }
    LoadMtl(matMap, materials, &matIStream, warn, err);
    return true;
  }

private:
  std::string m_mtlBaseDir;

  void LoadMtl(std::map<std::string, int> *material_map,
               std::vector<material_t> *materials, std::istream *inStream,
               std::string *warning, std::string *err) {
    (void)warning;
    (void)err;
    material_t material;
    bool has_d = false;
    std::string linebuf;
    while (inStream->peek() != -1) {
      std::getline(*inStream, linebuf);
      if (linebuf.empty() || linebuf[0] == '#')
        continue;
      std::istringstream iss(linebuf);
      std::string token;
      iss >> token;
      if (token == "newmtl") {
        if (!material.name.empty()) {
          (*material_map)[material.name] = (int)materials->size();
          materials->push_back(material);
        }
        material = material_t();
        has_d = false;
        iss >> material.name;
      } else if (token == "Ka") {
        iss >> material.ambient[0] >> material.ambient[1] >>
            material.ambient[2];
      } else if (token == "Kd") {
        iss >> material.diffuse[0] >> material.diffuse[1] >>
            material.diffuse[2];
      } else if (token == "Ks") {
        iss >> material.specular[0] >> material.specular[1] >>
            material.specular[2];
      } else if (token == "Ke") {
        iss >> material.emission[0] >> material.emission[1] >>
            material.emission[2];
      } else if (token == "Ns") {
        iss >> material.shininess;
      } else if (token == "Ni") {
        iss >> material.ior;
      } else if (token == "d") {
        iss >> material.dissolve;
        has_d = true;
      } else if (token == "Tr") {
        if (!has_d) {
          float tr;
          iss >> tr;
          material.dissolve = 1.0f - tr;
        }
      } else if (token == "illum") {
        iss >> material.illum;
      }
    }
    if (!material.name.empty()) {
      (*material_map)[material.name] = (int)materials->size();
      materials->push_back(material);
    }
  }
};

static inline bool fixIndex(int idx, int n, int *ret) {
  if (idx > 0) {
    *ret = idx - 1;
    return true;
  }
  if (idx == 0) {
    return false;
  }
  *ret = n + idx;
  return true;
}

static inline std::string parseString(std::istringstream &iss) {
  std::string s;
  iss >> s;
  return s;
}

static inline index_t parseRawTriple(const std::string &token, int vn, int nn,
                                     int tn) {
  index_t vi;
  vi.vertex_index = -1;
  vi.normal_index = -1;
  vi.texcoord_index = -1;

  size_t pos1 = token.find('/');
  if (pos1 == std::string::npos) {
    vi.vertex_index = std::atoi(token.c_str()) - 1;
    return vi;
  }
  vi.vertex_index = std::atoi(token.substr(0, pos1).c_str()) - 1;

  size_t pos2 = token.find('/', pos1 + 1);
  if (pos2 == std::string::npos) {
    if (pos1 + 1 < token.size()) {
      vi.texcoord_index = std::atoi(token.substr(pos1 + 1).c_str()) - 1;
    }
    return vi;
  }
  if (pos2 > pos1 + 1) {
    vi.texcoord_index =
        std::atoi(token.substr(pos1 + 1, pos2 - pos1 - 1).c_str()) - 1;
  }
  if (pos2 + 1 < token.size()) {
    vi.normal_index = std::atoi(token.substr(pos2 + 1).c_str()) - 1;
  }
  return vi;
}

bool LoadObj(attrib_t *attrib, std::vector<shape_t> *shapes,
             std::vector<material_t> *materials, std::string *warn,
             std::string *err, const char *filename,
             const char *mtl_basedir = nullptr, bool triangulate = true) {
  (void)warn;
  (void)triangulate;

  attrib->vertices.clear();
  attrib->normals.clear();
  attrib->texcoords.clear();
  attrib->colors.clear();
  shapes->clear();
  materials->clear();

  std::ifstream ifs(filename);
  if (!ifs) {
    if (err)
      *err = "Cannot open file: " + std::string(filename);
    return false;
  }

  std::string basedir;
  if (mtl_basedir) {
    basedir = mtl_basedir;
  } else {
    std::string fn(filename);
    size_t pos = fn.find_last_of("/\\");
    if (pos != std::string::npos) {
      basedir = fn.substr(0, pos + 1);
    }
  }

  std::map<std::string, int> material_map;
  int material_id = -1;
  shape_t shape;
  std::string linebuf;

  while (ifs.peek() != -1) {
    std::getline(ifs, linebuf);
    if (linebuf.empty() || linebuf[0] == '#')
      continue;

    std::istringstream iss(linebuf);
    std::string token;
    iss >> token;

    if (token == "v") {
      float x, y, z;
      iss >> x >> y >> z;
      attrib->vertices.push_back(x);
      attrib->vertices.push_back(y);
      attrib->vertices.push_back(z);
    } else if (token == "vn") {
      float x, y, z;
      iss >> x >> y >> z;
      attrib->normals.push_back(x);
      attrib->normals.push_back(y);
      attrib->normals.push_back(z);
    } else if (token == "vt") {
      float u, v;
      iss >> u >> v;
      attrib->texcoords.push_back(u);
      attrib->texcoords.push_back(v);
    } else if (token == "f") {
      std::vector<index_t> face;
      std::string faceToken;
      int vn = (int)attrib->vertices.size() / 3;
      int nn = (int)attrib->normals.size() / 3;
      int tn = (int)attrib->texcoords.size() / 2;
      while (iss >> faceToken) {
        face.push_back(parseRawTriple(faceToken, vn, nn, tn));
      }
      for (size_t i = 1; i + 1 < face.size(); i++) {
        shape.mesh.indices.push_back(face[0]);
        shape.mesh.indices.push_back(face[i]);
        shape.mesh.indices.push_back(face[i + 1]);
        shape.mesh.material_ids.push_back(material_id);
      }
    } else if (token == "o" || token == "g") {
      if (!shape.mesh.indices.empty()) {
        shapes->push_back(shape);
        shape = shape_t();
      }
      iss >> shape.name;
    } else if (token == "usemtl") {
      std::string matname;
      iss >> matname;
      auto it = material_map.find(matname);
      if (it != material_map.end()) {
        material_id = it->second;
      } else {
        material_id = -1;
      }
    } else if (token == "mtllib") {
      std::string matfile;
      iss >> matfile;
      MaterialFileReader reader(basedir);
      std::string w, e;
      reader(matfile, materials, &material_map, &w, &e);
    }
  }

  if (!shape.mesh.indices.empty()) {
    shapes->push_back(shape);
  }

  return true;
}

} // namespace tinyobj
