#define RAYCOUNT 5000 // rays per source

#define HIT_THRESHOLD 1000
#define RAYS_PER_THREAD 80
#define L1_THRESHOLD 1
#define L2_THRESHOLD 170000

const float tuningCnst = 0.0087209302; // constants derived by fitting to logarithmic function
const float tuningScale = 0.0820614573;
const float incrRay = 0.15f;
const float degradeFac = 0.98f;
const float addFac = 100.f;

struct ray 
{
  Vec2<float> position;
  Vec2<float> velocity;
  float intensity;
  png::rgb_pixel color;
};

typedef struct ray lightray;

inline bool loadFromFile(std::string fileName,
                         std::vector<lightray> &lightSources) {
  std::ifstream f(fileName);
  assert((bool)f && "Cannot open input file");

  std::string line;
  while (std::getline(f, line)) {
    lightray pRay;
    std::stringstream sstream(line);
    std::string str;
    std::getline(sstream, str, ' ');
    pRay.position.x = (float)atof(str.c_str());
    std::getline(sstream, str, ' ');
    pRay.position.y = (float)atof(str.c_str());
    std::getline(sstream, str, ' ');
    pRay.color.red = (int)atoi(str.c_str());
    std::getline(sstream, str, ' ');
    pRay.color.green = (int)atoi(str.c_str());
    std::getline(sstream, str, ' ');
    pRay.color.blue = (int)atoi(str.c_str());
    std::getline(sstream, str, '\n');
    pRay.intensity = (float)atof(str.c_str());
    lightSources.push_back(pRay);
  }
  return true;
}