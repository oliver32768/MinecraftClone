#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <queue>
#include <random>
#include <functional>
#include <bitset>
#include <syncstream>
#include <unordered_set>
#include <array>
#include <numbers>
#include <set>
#include <optional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <stb_image.h>
#include <FastNoiseLite.h>
#include <Shader.h>
#include <Camera.h>

constexpr size_t CHUNK_W = 16;
constexpr size_t CHUNK_H = 16;
constexpr size_t CHUNK_L = 16;
constexpr size_t ATLAS_X = 16;
constexpr size_t ATLAS_Y = 16;
constexpr size_t RENDER_DIST = 16;
constexpr size_t NUM_MESHING_THREADS = 4;
constexpr glm::vec3 LEFT_FACE = { -1.0f,  0.0f,  0.0f };
constexpr glm::vec3 RIGHT_FACE = { 1.0f,  0.0f,  0.0f };
constexpr glm::vec3 BOTTOM_FACE = { 0.0f, -1.0f,  0.0f };
constexpr glm::vec3 TOP_FACE = { 0.0f,  1.0f,  0.0f };
constexpr glm::vec3 BACK_FACE = { 0.0f,  0.0f, -1.0f };
constexpr glm::vec3 FRONT_FACE = { 0.0f,  0.0f,  1.0f };
constexpr glm::vec3 INSIDE_FACE = { 0.0f,  0.0f,  0.0f };
constexpr glm::vec4 skyClearColor{ 174.0f / 255.0f, 207.0f / 255.0f, 255.0f / 255.0f, 1.0f };
constexpr glm::vec4 voidClearColor{ 0.0f, 0.0f, 0.0f, 0.0f };

enum class Block : uint8_t
{
    AIR = 0,
    GRASS = 1,
    STONE = 2,
    DIRT = 3,
    OAK = 4,
    COBBLE = 5,
    BEDROCK = 6,
    GRAVEL = 7,
    SAND = 8,
    OAK_LEAVES = 9,
    WATER = 10,
    ROSE = 11,
    DAFFODIL = 12,
    RED_MUSHROOM = 13,
    MUSHROOM = 14,
    SUGAR_CANE = 15
};

enum class Pass : uint8_t
{
    IDLING = 0,
    GENERATING = 1,
    DECORATING = 2,
    MESHING = 3
};

enum class CameraMode : uint8_t
{
    FREECAM = 0,
    FIRST_PERSON = 1,
    THIRD_PERSON_REAR = 2,
    THIRD_PERSON_FRONT = 3
};

enum class FaceDirection : uint8_t
{
    LEFT = 0,
    RIGHT = 1,
    BOTTOM = 2,
    TOP = 3,
    BACK = 4,
    FRONT = 5
};

struct SelectedBlock
{
    Block ID;
    glm::vec3 pos;
    glm::vec3 face;
};

struct Vertex
{
    // Bits 0-4   : Local space X coordinate
    // Bits 5-9   : Local space Y coordinate
    // Bits 10-14 : Local space Z coordinate
    // Bits 15-16 : Index into array of pre-computed brightness values
    // Bit 17     : Texture X coordinate
    // Bit 18     : Texture Y coordinate
    // Bit 19-26  : Texture2DArray layer
    // Bits 26-31 : Unused
    uint32_t packed;
};

struct Mesh
{
    std::vector<Vertex> vertices;
};

struct ChunkMeshes
{
    Mesh nonWaterMesh;
    Mesh waterMesh;
    glm::vec3 origin;
};
bool operator==(const ChunkMeshes& lhs, const ChunkMeshes& rhs)
{
    return lhs.origin == rhs.origin;
}
struct ChunkMeshesHasher
{
    std::size_t operator()(const ChunkMeshes& chunkMeshes) const
    {
        static std::hash<glm::vec3> hasher;
        return hasher(chunkMeshes.origin);
    }
};

struct MeshHandle
{
    GLuint VAO;
    GLuint VBO;
    size_t numIndices;
};
struct ChunkMeshHandles
{
    MeshHandle nonWaterMesh;
    MeshHandle waterMesh;
};
struct Chunk
{
    ChunkMeshHandles chunkMeshHandles;
    glm::vec3 origin;
    std::array<Block, CHUNK_W * CHUNK_H * CHUNK_L> chunkData;
};

struct Actor
{
    glm::vec3 pos;
    float yaw;

    bool onGround = false;
    float sideMag = 0.0f;
    float upMag = 0.0f;
    float forwardMag = 0.0f;
    glm::vec3 forward = { 0.0f, 0.0f, 1.0f };
    glm::vec3 up = { 0.0f, 1.0f, 0.0f };
    glm::vec3 side = { 1.0f, 0.0f, 0.0f };    

    Block hotbar[9] = { Block::AIR, Block::AIR, Block::AIR, Block::AIR, Block::AIR, Block::AIR, Block::AIR, Block::AIR, Block::AIR };

    const glm::vec3 aabbDims = { 0.6f, 1.8f, 0.6f };
    const float lateralAccel = 500.0f;
    const float jumpAccel = 20.0f;
    const float gravityAccel = 20.0f;
    const float lateralMagLimits[2] = { -5.0f, 5.0f };
    const float verticalMagLimits[2] = { -78.4f, 78.4f };
};

struct Frustum
{
    glm::vec4 planes[6];

    void updatePlanes(glm::mat4 vp)
    {
        planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
        planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
        planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
        planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
        planes[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]);
        planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);
    }

    bool inFrustum(glm::vec3 aabbDims, glm::vec3 pos)
    {
        glm::vec3 llb = pos;
        glm::vec3 rtf = pos + aabbDims;
        return
            planes[0].x * (planes[0].x < 0 ? llb.x : rtf.x) + planes[0].y * (planes[0].y < 0 ? llb.y : rtf.y) + planes[0].z * (planes[0].z < 0 ? llb.z : rtf.z) >= -planes[0].w &&
            planes[1].x * (planes[1].x < 0 ? llb.x : rtf.x) + planes[1].y * (planes[1].y < 0 ? llb.y : rtf.y) + planes[1].z * (planes[1].z < 0 ? llb.z : rtf.z) >= -planes[1].w &&
            planes[2].x * (planes[2].x < 0 ? llb.x : rtf.x) + planes[2].y * (planes[2].y < 0 ? llb.y : rtf.y) + planes[2].z * (planes[2].z < 0 ? llb.z : rtf.z) >= -planes[2].w &&
            planes[3].x * (planes[3].x < 0 ? llb.x : rtf.x) + planes[3].y * (planes[3].y < 0 ? llb.y : rtf.y) + planes[3].z * (planes[3].z < 0 ? llb.z : rtf.z) >= -planes[3].w &&
            planes[4].x * (planes[4].x < 0 ? llb.x : rtf.x) + planes[4].y * (planes[4].y < 0 ? llb.y : rtf.y) + planes[4].z * (planes[4].z < 0 ? llb.z : rtf.z) >= -planes[4].w &&
            planes[5].x * (planes[5].x < 0 ? llb.x : rtf.x) + planes[5].y * (planes[5].y < 0 ? llb.y : rtf.y) + planes[5].z * (planes[5].z < 0 ? llb.z : rtf.z) >= -planes[5].w;
    }
};

GLFWwindow* initOpenGL();
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void mouseCallback(GLFWwindow* window, double xposIn, double yposIn);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window, Actor& playerActor, float deltaTime);
GLuint loadTexture(char const* path, GLenum wrapParam);
bool compareVec3(const glm::vec3& a, const glm::vec3& b);
void meshingThread
(
    std::atomic<bool>& suspendMeshingThreads,
    std::atomic<uint8_t>& numMeshingThreadsWaiting, 
    std::condition_variable& mainThreadCV, 
    std::mutex& ChunksMutex,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::unordered_set<ChunkMeshes, ChunkMeshesHasher>& needsBuffering, 
    std::multiset<glm::vec3, decltype(&compareVec3)>& needsGenerating,
    std::mutex& needsGeneratingMutex,
    std::mutex& needsBufferingMutex,
    std::condition_variable& meshingThreadCV, 
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshing,
    std::atomic<Pass>& currentPass,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshingHighPriority,
    std::mutex& needsMeshingHPMutex,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsDecorating,
    std::mutex& needsDecoratingMutex,
    GLFWwindow* window
);
MeshHandle bufferMesh(const Mesh& mesh);
void draw(const MeshHandle& meshHandle);
void needsGeneratingAllSphere
(
    std::multiset<glm::vec3, decltype(&compareVec3)>& needsGenerating,
    glm::vec3 playerPos
);
ChunkMeshes buildMeshFromChunkData
(
    const std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, 
    const std::array<Block, CHUNK_W * CHUNK_H * CHUNK_L>& chunkData, 
    const glm::vec3 chunkOrigin
);
std::bitset<6> getOccludedFaces
(
    const std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, 
    const glm::vec3& localCoords,
    const glm::vec3& origin,
    const std::array<Block, CHUNK_W * CHUNK_H * CHUNK_L>& chunkData
);
int localChunkPosToIndex16cubed(const glm::vec3& v);
void addFace(Mesh& mesh, size_t& currentMeshIndex, const glm::vec3& pos, const uint8_t texture2DArrayDepth, FaceDirection dir, const glm::vec3& chunkOrigin);
void clearMeshes(std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks);
glm::vec3 getChunkOfPositionVector(glm::vec3 pos);
std::unordered_set<glm::vec3, std::hash<glm::vec3>> tryReplaceBlock
(
    const glm::vec3& targetBlockWorldPos, 
    const Block targetBlockNewValue, 
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex
);
Block tryReadBlock(const glm::vec3& targetBlockWorldPos, std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, std::mutex& ChunksMutex);
int randomIntInRange(int min, int max);
bool randomChance(float decimalProbability);
std::unordered_set<glm::vec3, std::hash<glm::vec3>> createTree
(
    const glm::vec3& targetBlockWorldPos, 
    std::unordered_map<glm::vec3, Chunk,
    std::hash<glm::vec3>>& Chunks, std::mutex& ChunksMutex
);
uint8_t getSideFaceTexture(Block blockEnum);
uint8_t getTopFaceTexture(Block blockEnum);
uint8_t getBottomFaceTexture(Block blockEnum);
bool isTransparent(Block blockEnum);
void bufferChunksFunc
(
    std::mutex& needsBufferingMutex,
    std::mutex& ChunksMutex,
    std::unordered_set<ChunkMeshes, ChunkMeshesHasher>& needsBuffering,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks
);
void drawChunks
(
    const std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    Shader& shader,
    Frustum& frustum,
    GLuint atlas,
    GLuint nonWaterFBO,
    GLuint waterFBO,
    Shader& fboShader,
    GLuint screenQuadVAO,
    GLuint nonWaterColor,
    GLuint nonWaterDepth,
    GLuint waterColor,
    GLuint waterDepth,
    const glm::mat4& view,
    const glm::mat4& projection
);
void simulateActor(Actor& actor, std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, std::mutex& ChunksMutex, float deltaTime);
bool aabbIsColliding(glm::vec3 pos, glm::vec3 aabbDims, std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, std::mutex& ChunksMutex);
SelectedBlock castRayToBlock
(
    glm::vec3 rayOrigin,
    glm::vec3 rayDirection,
    float stepSize,
    float maxRayLength,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex
);
void reloadChunksMinimal
(
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex,
    std::multiset<glm::vec3, decltype(&compareVec3)>& needsGenerating,
    std::mutex& needsGeneratingMutex,
    std::atomic<bool>& suspendMeshingThreads,
    std::atomic<uint8_t>& numMeshingThreadsWaiting,
    std::mutex& mainThreadCVMutex,
    std::condition_variable& mainThreadCV,
    std::condition_variable& meshingThreadCV,
    glm::vec3 playerPos,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsDecorating,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshing,
    std::unordered_set<ChunkMeshes, ChunkMeshesHasher>& needsBuffering
);
std::unordered_set<glm::vec3, std::hash<glm::vec3>> generateSphericalRenderingVolume(glm::vec3 playerPos);
void processInputFreecam(GLFWwindow* window, Actor& playerActor, float deltaTime);
bool blockInAABB(glm::vec3 blockPos, const Actor& actor);
std::unordered_set<glm::vec3, std::hash<glm::vec3>> decorateChunkNormalTerrain
(
    std::array<Block, CHUNK_W* CHUNK_H* CHUNK_L>& chunkData,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex,
    FastNoiseLite& HeightGen,
    FastNoiseLite& ErosionGen,
    const glm::vec3& sourceChunkOrigin
);
std::array<Block, CHUNK_W* CHUNK_H* CHUNK_L> normalTerrain(const glm::vec3& chunkOrigin, FastNoiseLite& HeightGen, FastNoiseLite& ErosionGen);
int getTerrainHeight(int x, int z, FastNoiseLite& HeightGen, FastNoiseLite& ErosionGen);
bool isSelfOccludingBlock(Block blockEnum);
void addVisibleFaces(const std::bitset<6>& occludedFaces, Mesh& mesh, size_t& meshIndex, const glm::vec3& localChunkPos, const Block blockEnum, const glm::vec3& chunkOrigin);
GLuint loadTexturesIntoArray(const std::vector<const char*>& paths);
GLuint genFramebufferTextureAttachment(GLenum internalFormat, GLenum format, GLenum type);
GLuint genFramebufferWithDepthAndColor(GLuint colorTexture, GLuint depthTexture);
GLuint genTexturedQuadVAO();
void resizeFBOTexture(GLuint texture, GLenum internalFormat, GLenum format, GLenum type);
void debugKeys(GLFWwindow* window, Actor& playerActor, float deltaTime);
void mergeFramebuffers(GLuint nonWaterColor, GLuint nonWaterDepth, GLuint waterColor, GLuint waterDepth, Shader& fboShader, GLuint texturedQuadVAO);
void drawClouds(GLuint nonWaterFBO, GLuint cloudTexture, GLuint texturedQuadVAO, Shader& cloudShader, const glm::mat4& view, const glm::mat4& projection);
void drawHud(GLuint texturedQuadVAO, GLuint hudAtlas, Shader& hudShader, GLuint blockAtlas, const Actor& playerActor);
void drawQuadHudShader(Shader& hudShader, const glm::vec3& renderPos, const glm::vec3& renderScale, const glm::vec2& atlasCutout, const glm::vec2& atlasOrigin);
glm::vec2 getSideFaceTextureAtlas(Block blockEnum);
bool isBillboard(Block blockEnum);
void addBillboard(Mesh& mesh, size_t& currentMeshIndex, const glm::vec3& localChunkPos, const glm::vec3& chunkOrigin, const Block blockEnum, const uint8_t texture2DArrayDepth);
bool isSolid(Block blockEnum);
constexpr uint32_t packAttributes32(int posX, int posY, int posZ, int brightnessLookup, int uvX, int uvY, int uvZ);
void destroyBlockFunc
(
    const SelectedBlock& targetedBlock,
    std::atomic<Pass>& currentPass,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshingHighPriority,
    std::mutex& needsMeshingHPMutex,
    std::condition_variable& meshingThreadCV
);
void placeBlockFunc
(
    const SelectedBlock& targetedBlock,
    std::atomic<Pass>& currentPass,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshingHighPriority,
    std::mutex& needsMeshingHPMutex,
    std::condition_variable& meshingThreadCV,
    Actor& playerActor
);
void selectBlockFunc(const SelectedBlock& targetedBlock, Actor& playerActor);

uint32_t WIDTH = 800;
uint32_t HEIGHT = 600;

Camera camera(glm::vec3(0.0f, 72.0f, 0.0f));
CameraMode cameraMode = CameraMode::FIRST_PERSON;

float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool firstMouse = true;
bool destroyBlock = false;
bool placeBlock = false;
bool selectBlock = false;
bool viewportSizeChanged = false;
bool flyMode = false;
bool reloadChunksBool = false;
std::atomic<bool> programEnded = false;

std::random_device rd;
std::mt19937 gen(rd());
glm::vec3 playerChunkSnapshot{};

size_t drawCallsThisFrame{};

int selectedItem = 0;

int main()
{
    GLFWwindow* window = initOpenGL();

    Shader chunkShader("res/shaders/chunk.vert", "res/shaders/chunk.frag");
    Shader fboShader("res/shaders/fbo.vert", "res/shaders/fbo.frag");
    Shader cloudShader("res/shaders/clouds.vert", "res/shaders/clouds.frag");
    Shader hudShader("res/shaders/hud.vert", "res/shaders/hud.frag");

    GLuint cloudTexture = loadTexture("res/textures/environment/clouds.png", GL_REPEAT);
    GLuint sunTexture = loadTexture("res/textures/environment/sun.png", GL_CLAMP_TO_EDGE);
    GLuint hudAtlas = loadTexture("res/textures/icons/widgets.png", GL_CLAMP_TO_EDGE);
    GLuint blockAtlas = loadTexture("res/textures/blocks/atlas/blocks.png", GL_CLAMP_TO_EDGE); // Atlas for 2D rendering, Array for 3D rendering

    // Invariant uniforms
    fboShader.use();
    fboShader.setInt("fbo1ColorTexture", 1);
    fboShader.setInt("fbo1DepthTexture", 2);
    fboShader.setInt("fbo2ColorTexture", 3);
    fboShader.setInt("fbo2DepthTexture", 4);
    cloudShader.use();
    cloudShader.setInt("texture1", 0);
    chunkShader.use();
    chunkShader.setInt("texture1", 0);

    GLuint blockArray = loadTexturesIntoArray
    ({
        "res/textures/blocks/array/bedrock.png",
        "res/textures/blocks/array/brick.png",
        "res/textures/blocks/array/clay.png",
        "res/textures/blocks/array/coal_ore.png",
        "res/textures/blocks/array/cobble.png",
        "res/textures/blocks/array/dandelion.png",
        "res/textures/blocks/array/diamond_ore.png",
        "res/textures/blocks/array/dirt.png",
        "res/textures/blocks/array/double_stone_slab.png",
        "res/textures/blocks/array/glass.png",
        "res/textures/blocks/array/gold_ore.png",
        "res/textures/blocks/array/grass_side.png",
        "res/textures/blocks/array/grass_top.png",
        "res/textures/blocks/array/gravel.png",
        "res/textures/blocks/array/iron_ore.png",
        "res/textures/blocks/array/lava.png",
        "res/textures/blocks/array/oak_leaves_fancy.png",
        "res/textures/blocks/array/oak_leaves_fast.png",
        "res/textures/blocks/array/oak_plank.png",
        "res/textures/blocks/array/oak_side.png",
        "res/textures/blocks/array/oak_top.png",
        "res/textures/blocks/array/redstone_ore.png",
        "res/textures/blocks/array/rose.png",
        "res/textures/blocks/array/sand.png",
        "res/textures/blocks/array/smooth_stone.png",
        "res/textures/blocks/array/stone.png",
        "res/textures/blocks/array/water.png",
        "res/textures/blocks/array/red_mushroom.png",
        "res/textures/blocks/array/mushroom.png",
        "res/textures/blocks/array/sugar_cane.png",
    });

    GLuint nonWaterColor = genFramebufferTextureAttachment(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint nonWaterDepth = genFramebufferTextureAttachment(GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint nonWaterFBO = genFramebufferWithDepthAndColor(nonWaterColor, nonWaterDepth);

    GLuint waterColor = genFramebufferTextureAttachment(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
    GLuint waterDepth = genFramebufferTextureAttachment(GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    GLuint waterFBO = genFramebufferWithDepthAndColor(waterColor, waterDepth);

    GLuint texturedQuadVAO = genTexturedQuadVAO();

    auto compareVec3Function = compareVec3;
    std::multiset<glm::vec3, decltype(&compareVec3)> needsGenerating(compareVec3);
    std::unordered_set<glm::vec3, std::hash<glm::vec3>> needsDecorating{};
    std::unordered_set<ChunkMeshes, ChunkMeshesHasher> needsBuffering{};
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>> Chunks{};
    std::unordered_set<glm::vec3, std::hash<glm::vec3>> needsMeshing{};
    std::mutex needsGeneratingMutex{};
    std::mutex needsDecoratingMutex{};
    std::mutex needsBufferingMutex{};
    std::mutex ChunksMutex{};
    std::mutex mainThreadCVMutex{};
    std::mutex meshingThreadCVMutex{};
    std::condition_variable meshingThreadCV{};
    std::condition_variable mainThreadCV{};
    std::atomic<uint8_t> numMeshingThreadsWaiting = 0;
    std::atomic<bool> suspendMeshingThreads = true;
    std::atomic<Pass> currentPass = Pass::IDLING;
    std::vector<std::jthread> meshingThreads{};
    std::unordered_set<glm::vec3, std::hash<glm::vec3>> needsMeshingHighPriority{};
    std::mutex needsMeshingHPMutex;

    Chunks.reserve(std::pow((2 * RENDER_DIST) + 1, 3));

    for (size_t i = 0; i < NUM_MESHING_THREADS; ++i)
    {
        meshingThreads.emplace_back
        (
            meshingThread,
            std::ref(suspendMeshingThreads),
            std::ref(numMeshingThreadsWaiting),
            std::ref(mainThreadCV),
            std::ref(ChunksMutex),
            std::ref(Chunks),
            std::ref(needsBuffering),
            std::ref(needsGenerating),
            std::ref(needsGeneratingMutex),
            std::ref(needsBufferingMutex),
            std::ref(meshingThreadCV),
            std::ref(needsMeshing),
            std::ref(currentPass),
            std::ref(needsMeshingHighPriority),
            std::ref(needsMeshingHPMutex),
            std::ref(needsDecorating),
            std::ref(needsDecoratingMutex),
            window
        );
    }

    Actor playerActor
    { 
        .pos = camera.Position,
        .yaw = camera.Yaw 
    };
    
    glm::vec3 playerChunk = getChunkOfPositionVector(playerActor.pos);
    glm::vec3 previousPlayerChunk = playerChunk;

    Frustum frustum;

    float lastPrint = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        if (viewportSizeChanged)
        {
            viewportSizeChanged = false;

            resizeFBOTexture(nonWaterColor, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
            resizeFBOTexture(nonWaterDepth, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
            resizeFBOTexture(waterColor, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
            resizeFBOTexture(waterDepth, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
        }

        SelectedBlock targetedBlock{};
        if (cameraMode == CameraMode::FREECAM)
        {
            processInputFreecam(window, playerActor, deltaTime);
            simulateActor(playerActor, Chunks, ChunksMutex, deltaTime);
        }
        else if (cameraMode == CameraMode::FIRST_PERSON)
        {
            processInput(window, playerActor, deltaTime);
            simulateActor(playerActor, Chunks, ChunksMutex, deltaTime);
            camera.Position = { playerActor.pos.x, playerActor.pos.y + 0.7f, playerActor.pos.z };
            playerActor.yaw = camera.Yaw;
            targetedBlock = castRayToBlock(camera.Position, camera.Front, 0.1f, 20.0f, Chunks, ChunksMutex);
        }
        else if (cameraMode == CameraMode::THIRD_PERSON_REAR) { /* stub */ }
        else if (cameraMode == CameraMode::THIRD_PERSON_FRONT) { /* stub */ }
        
        if (destroyBlock) destroyBlockFunc(targetedBlock, currentPass, Chunks, ChunksMutex, needsMeshingHighPriority, needsMeshingHPMutex, meshingThreadCV);
        if (placeBlock) placeBlockFunc(targetedBlock, currentPass, Chunks, ChunksMutex, needsMeshingHighPriority, needsMeshingHPMutex, meshingThreadCV, playerActor);
        if (selectBlock) selectBlockFunc(targetedBlock, playerActor);

        playerChunk = getChunkOfPositionVector(playerActor.pos);
        if (previousPlayerChunk != playerChunk)
        {
            reloadChunksMinimal
            (
                Chunks,
                ChunksMutex,
                needsGenerating,
                needsGeneratingMutex,
                suspendMeshingThreads,
                numMeshingThreadsWaiting,
                mainThreadCVMutex,
                mainThreadCV,
                meshingThreadCV,
                playerActor.pos,
                needsDecorating,
                needsMeshing,
                needsBuffering
            );
        }
        previousPlayerChunk = playerChunk;

        if (!needsBuffering.empty()) bufferChunksFunc(needsBufferingMutex, ChunksMutex, needsBuffering, Chunks);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, nonWaterFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, waterFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)WIDTH / (float)HEIGHT, 0.1f, 1000.0f);

        drawClouds(nonWaterFBO, cloudTexture, texturedQuadVAO, cloudShader, view, projection);
        drawChunks
        (
            Chunks,
            chunkShader,
            frustum, 
            blockArray,
            nonWaterFBO,
            waterFBO, 
            fboShader, 
            texturedQuadVAO,
            nonWaterColor,
            nonWaterDepth,
            waterColor,
            waterDepth,
            view,
            projection
        );
        mergeFramebuffers(nonWaterColor, nonWaterDepth, waterColor, waterDepth, fboShader, texturedQuadVAO);
        drawHud(texturedQuadVAO, hudAtlas, hudShader, blockAtlas, playerActor);

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (currentFrame - lastPrint > 3.0f)
        {
            lastPrint = currentFrame;
            std::cout << "[LOG] # Draw calls: " << drawCallsThisFrame << '\n';
            std::cout << "[LOG] FPS: " << 1.0f / deltaTime << '\n';
        }

        drawCallsThisFrame = 0;
    }

    programEnded = true;
    meshingThreadCV.notify_all();

    return 0;
}

void selectBlockFunc(const SelectedBlock& targetedBlock, Actor& playerActor)
{
    selectBlock = false;

    playerActor.hotbar[selectedItem] = targetedBlock.ID;
}

void destroyBlockFunc
(
    const SelectedBlock& targetedBlock,
    std::atomic<Pass>& currentPass, 
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, 
    std::mutex& ChunksMutex,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshingHighPriority,
    std::mutex& needsMeshingHPMutex,
    std::condition_variable& meshingThreadCV
)
{
    destroyBlock = false;

    if (targetedBlock.ID != Block::AIR)
    {
        std::unordered_set<glm::vec3, std::hash<glm::vec3>> affectedChunk = tryReplaceBlock(targetedBlock.pos, Block::AIR, Chunks, ChunksMutex);

        if (currentPass == Pass::IDLING)
        {
            currentPass = Pass::MESHING;
        }

        std::scoped_lock lock{ needsMeshingHPMutex };

        needsMeshingHighPriority.merge(affectedChunk);

        meshingThreadCV.notify_all();
    }
}

void placeBlockFunc
(
    const SelectedBlock& targetedBlock,
    std::atomic<Pass>& currentPass,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshingHighPriority,
    std::mutex& needsMeshingHPMutex,
    std::condition_variable& meshingThreadCV,
    Actor& playerActor
)
{
    placeBlock = false;

    glm::vec3 newBlockPos = targetedBlock.pos + targetedBlock.face;

    if (targetedBlock.ID != Block::AIR && !blockInAABB(newBlockPos, playerActor))
    {
        std::unordered_set<glm::vec3, std::hash<glm::vec3>> affectedChunk = tryReplaceBlock(newBlockPos, playerActor.hotbar[selectedItem], Chunks, ChunksMutex);

        if (currentPass == Pass::IDLING)
        {
            currentPass = Pass::MESHING;
        }

        std::scoped_lock lock{ needsMeshingHPMutex };

        needsMeshingHighPriority.merge(affectedChunk);

        meshingThreadCV.notify_all();
    }
}

void drawHud(GLuint texturedQuadVAO, GLuint hudAtlas, Shader& hudShader, GLuint blockAtlas, const Actor& playerActor)
{
    glBindVertexArray(texturedQuadVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hudAtlas);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    hudShader.use();

    float zoom = 1.0f;
    float view_width = (WIDTH / zoom);
    float view_height = (HEIGHT / zoom);
    float left = (WIDTH - view_width) / 2.0f;
    float right = left + view_width;
    float top = (HEIGHT - view_height) / 2.0f;
    float bottom = top + view_height;
    glm::mat4 hudProjection = glm::ortho(left, right, bottom, top, -1.0f, 1.0f);
    hudShader.setMat4("projection", hudProjection);

    float hudScale;
    if (HEIGHT < 480.0f) hudScale = 0.5f;
    else if (HEIGHT < 720.0f) hudScale = 1.0f;
    else hudScale = 1.5f;

    // Crosshair
    glm::vec3 renderPos(WIDTH / 2.0f, HEIGHT / 2.0f, 0.0f);
    glm::vec3 renderScale(16.0f * hudScale, 16.0f * hudScale, 1.0f);
    glm::vec2 atlasCutout(16.0f / 256.0f, 16.0f / 256.0f);
    glm::vec2 atlasOrigin(240.0f / 256.0f, 0.0f / 256.0f);
    glEnable(GL_COLOR_LOGIC_OP);
    drawQuadHudShader(hudShader, renderPos, renderScale, atlasCutout, atlasOrigin);
    glDisable(GL_COLOR_LOGIC_OP);

    // Hotbar
    renderPos = glm::vec3(WIDTH / 2.0f, HEIGHT - (22.0f * hudScale), 0.0f);
    renderScale = glm::vec3(182.0f * hudScale, 22.0f * hudScale, 1.0f);
    atlasCutout = glm::vec2(182.0f / 256.0f, 22.0f / 256.0f);
    atlasOrigin = glm::vec2(0.0f / 256.0f, 0.0f / 256.0f);
    glEnable(GL_BLEND);
    drawQuadHudShader(hudShader, renderPos, renderScale, atlasCutout, atlasOrigin);
    glDisable(GL_BLEND);

    // Selected hotbar item outline
    renderPos = glm::vec3(WIDTH / 2.0f + ((selectedItem - 4) * 40.0f * hudScale), HEIGHT - (22.0f * hudScale), 0.0f);
    renderScale = glm::vec3(24.0f * hudScale, 24.0f * hudScale, 1.0f);
    atlasCutout = glm::vec2(24.0f / 256.0f, 24.0f / 256.0f);
    atlasOrigin = glm::vec2(0.0f / 256.0f, 22.0f / 256.0f);
    drawQuadHudShader(hudShader, renderPos, renderScale, atlasCutout, atlasOrigin);

    // Hotbar item icons
    glBindTexture(GL_TEXTURE_2D, blockAtlas);
    for (int i = 0; i < 9; ++i)
    {
        if (playerActor.hotbar[i] != Block::AIR)
        {
            renderPos = glm::vec3(WIDTH / 2.0f + ((i - 4) * 40.0f * hudScale), HEIGHT - (22.0f * hudScale), 0.0f);
            renderScale = glm::vec3(16.0f * hudScale, 16.0f * hudScale, 1.0f);
            atlasCutout = glm::vec2(16.0f / 256.0f, 16.0f / 256.0f);
            atlasOrigin = getSideFaceTextureAtlas(playerActor.hotbar[i]) / 16.0f;
            drawQuadHudShader(hudShader, renderPos, renderScale, atlasCutout, atlasOrigin);
        }
    }

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

// Assumes appropriate state already bound
void drawQuadHudShader(Shader& hudShader, const glm::vec3& renderPos, const glm::vec3& renderScale, const glm::vec2& atlasCutout, const glm::vec2& atlasOrigin)
{
    glm::mat4 model(1.0f);
    model = glm::translate(model, renderPos);
    model = glm::scale(model, renderScale);
    hudShader.setMat4("model", model);
    hudShader.setVec2("atlasCutout", atlasCutout);
    hudShader.setVec2("atlasOrigin", atlasOrigin);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    ++drawCallsThisFrame;
}

void mergeFramebuffers(GLuint nonWaterColor, GLuint nonWaterDepth, GLuint waterColor, GLuint waterDepth, Shader& fboShader, GLuint texturedQuadVAO)
{
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, nonWaterColor);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, nonWaterDepth);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, waterColor);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, waterDepth);

    fboShader.use();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindVertexArray(texturedQuadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    ++drawCallsThisFrame;
}

void drawClouds(GLuint nonWaterFBO, GLuint cloudTexture, GLuint texturedQuadVAO, Shader& cloudShader, const glm::mat4& view, const glm::mat4& projection)
{
    float mult = 9.0f;
    float scale = 528.0f;
    unsigned int sideLen = scale * 2.0f;

    int xOffset = camera.Position.x / sideLen;
    int zOffset = camera.Position.z / sideLen;

    glm::vec3 gridOffset = glm::vec3(static_cast<float>(xOffset), 0.0f, static_cast<float>(zOffset)) * static_cast<float>(sideLen);

    glBindFramebuffer(GL_FRAMEBUFFER, nonWaterFBO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, cloudTexture);
    glBindVertexArray(texturedQuadVAO);
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(gridOffset.x, 66.0f, gridOffset.z));
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::scale(model, glm::vec3(scale * mult, scale * mult, 1.0f));
    cloudShader.use();
    cloudShader.setFloat("uvOffset", glfwGetTime() / 8192.0f);
    cloudShader.setFloat("uvMult", mult);
    cloudShader.setMat4("model", model);
    cloudShader.setMat4("view", view);
    cloudShader.setMat4("projection", projection);
    glDisable(GL_CULL_FACE);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glEnable(GL_CULL_FACE);

    ++drawCallsThisFrame;
}

void drawChunks
(
    const std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    Shader& chunkShader,
    Frustum& frustum,
    GLuint atlas,
    GLuint nonWaterFBO,
    GLuint waterFBO,
    Shader& fboShader,
    GLuint screenQuadVAO,
    GLuint nonWaterColor,
    GLuint nonWaterDepth,
    GLuint waterColor,
    GLuint waterDepth,
    const glm::mat4& view,
    const glm::mat4& projection
)
{
    static constexpr glm::vec3 chunkDims = { CHUNK_W, CHUNK_H, CHUNK_L };

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, atlas);

    chunkShader.use();
    chunkShader.setMat4("view", view);
    chunkShader.setMat4("projection", projection);
    frustum.updatePlanes(projection * view);

    std::vector<const Chunk*> chunksInFrustum{};

    glBindFramebuffer(GL_FRAMEBUFFER, nonWaterFBO);
    for (const auto& [chunkOrigin, chunk] : Chunks)
    {
        if (frustum.inFrustum(chunkDims, chunkOrigin))
        {
            chunksInFrustum.emplace_back(&chunk);

            if (chunk.chunkMeshHandles.nonWaterMesh.VAO != 0 && chunk.chunkMeshHandles.nonWaterMesh.numIndices > 0)
            {
                chunkShader.setMat4("model", glm::translate(glm::mat4(1.0f), chunkOrigin));
                draw(chunk.chunkMeshHandles.nonWaterMesh);
            }
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, waterFBO);
    glDisable(GL_CULL_FACE);
    for (const auto& chunk : chunksInFrustum)
    {
        if (chunk->chunkMeshHandles.waterMesh.VAO != 0 && chunk->chunkMeshHandles.waterMesh.numIndices > 0)
        {
            chunkShader.setMat4("model", glm::translate(glm::mat4(1.0f), chunk->origin));
            draw(chunk->chunkMeshHandles.waterMesh);
        }
    }
    glEnable(GL_CULL_FACE);
}

void processInput(GLFWwindow* window, Actor& playerActor, float deltaTime)
{
    static bool lmbHeld = false;
    static bool rmbHeld = false;
    static bool mmbHeld = false;

    if (!flyMode)
    {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            float newVelocity = playerActor.forwardMag + (playerActor.lateralAccel * deltaTime);
            playerActor.forwardMag = newVelocity > playerActor.lateralMagLimits[1] ? playerActor.lateralMagLimits[1] : newVelocity;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            float newVelocity = playerActor.forwardMag - (playerActor.lateralAccel * deltaTime);
            playerActor.forwardMag = newVelocity < playerActor.lateralMagLimits[0] ? playerActor.lateralMagLimits[0] : newVelocity;
        }
        else if (glfwGetKey(window, GLFW_KEY_W) != GLFW_PRESS && glfwGetKey(window, GLFW_KEY_S) != GLFW_PRESS)
        {
            playerActor.forwardMag = 0.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            float newVelocity = playerActor.sideMag - (playerActor.lateralAccel * deltaTime);
            playerActor.sideMag = newVelocity < playerActor.lateralMagLimits[0] ? playerActor.lateralMagLimits[0] : newVelocity;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            float newVelocity = playerActor.sideMag + (playerActor.lateralAccel * deltaTime);
            playerActor.sideMag = newVelocity > playerActor.lateralMagLimits[1] ? playerActor.lateralMagLimits[1] : newVelocity;
        }
        else if (glfwGetKey(window, GLFW_KEY_A) != GLFW_PRESS && glfwGetKey(window, GLFW_KEY_D) != GLFW_PRESS)
        {
            playerActor.sideMag = 0.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && playerActor.onGround)
        {
            playerActor.upMag = playerActor.jumpAccel;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        {
            // crouch
        }
    }
    else
    {
        playerActor.forwardMag = 0.0f;
        playerActor.sideMag = 0.0f;
        playerActor.upMag = 0.0f;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            playerActor.forwardMag = 100.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            playerActor.forwardMag = -100.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            playerActor.sideMag = -100.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            playerActor.sideMag = 100.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        {
            playerActor.upMag = 100.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        {
            playerActor.upMag = -100.0f;
        }
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        if (!lmbHeld)
        {
            lmbHeld = true;
            destroyBlock = true;
        }
    }
    if (lmbHeld && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        lmbHeld = false;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        if (!rmbHeld)
        {
            rmbHeld = true;
            placeBlock = true;
        }
    }
    if (rmbHeld && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
    {
        rmbHeld = false;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
        if (!mmbHeld)
        {
            mmbHeld = true;
            selectBlock = true;
        }
    }
    if (mmbHeld && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_RELEASE)
    {
        mmbHeld = false;
    }

    debugKeys(window, playerActor, deltaTime);
}

void processInputFreecam(GLFWwindow* window, Actor& playerActor, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(FORWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(RIGHT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(UP, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(DOWN, deltaTime);
    }

    debugKeys(window, playerActor, deltaTime);
}

void debugKeys(GLFWwindow* window, Actor& playerActor, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }

    static bool qHeld = false;
    static bool jHeld = false;
    static bool fHeld = false;
    if (!qHeld && glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        qHeld = true;
        cameraMode = cameraMode == CameraMode::FREECAM ? CameraMode::FIRST_PERSON : CameraMode::FREECAM;
    }
    if (qHeld && glfwGetKey(window, GLFW_KEY_Q) == GLFW_RELEASE)
    {
        qHeld = false;
    }
    if (!jHeld && glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
    {
        jHeld = true;
        playerActor.pos -= 32.0f;
    }
    if (jHeld && glfwGetKey(window, GLFW_KEY_J) == GLFW_RELEASE)
    {
        jHeld = false;
    }
    if (!fHeld && glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    {
        fHeld = true;
        flyMode = !flyMode;
    }
    if (fHeld && glfwGetKey(window, GLFW_KEY_F) == GLFW_RELEASE)
    {
        fHeld = false;
    }
}

void simulateActor(Actor& actor, std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, std::mutex& ChunksMutex, float deltaTime)
{
    actor.forward = glm::vec3(std::cos(glm::radians(actor.yaw)), 0.0f, std::sin(glm::radians(actor.yaw)));
    actor.up = glm::vec3(0.0f, 1.0f, 0.0f);
    actor.side = glm::normalize(glm::cross(actor.forward, actor.up));

    glm::vec3 motionDelta = actor.forward * actor.forwardMag + actor.side * actor.sideMag;
    if (!flyMode)
    {
        if (glm::length(motionDelta) > actor.lateralMagLimits[1])
        {
            motionDelta = glm::normalize(motionDelta);
            motionDelta *= actor.lateralMagLimits[1];
        }
        else if (glm::length(motionDelta) < actor.lateralMagLimits[0])
        {
            motionDelta = glm::normalize(motionDelta);
            motionDelta *= actor.lateralMagLimits[0];
        }
    }
    motionDelta += actor.up * actor.upMag;

    glm::vec3 candidatePos = actor.pos + (motionDelta * deltaTime);

    glm::vec3 xStep = glm::vec3(candidatePos.x, actor.pos.y, actor.pos.z);
    if (!aabbIsColliding(xStep, actor.aabbDims, Chunks, ChunksMutex))
    {
        actor.pos.x = candidatePos.x;
    }

    glm::vec3 yStep = glm::vec3(actor.pos.x, candidatePos.y, actor.pos.z);
    if (!aabbIsColliding(yStep, actor.aabbDims, Chunks, ChunksMutex))
    {
        actor.onGround = false;

        actor.pos.y = candidatePos.y;

        if (!flyMode)
        {
            float newVertSpeed = actor.upMag - (actor.gravityAccel * deltaTime);
            actor.upMag = newVertSpeed < actor.verticalMagLimits[0] ? actor.verticalMagLimits[0] : newVertSpeed;
        }
    }
    else
    {
        if (yStep.y <= actor.pos.y)
        {
            actor.onGround = true;
        }

        actor.upMag = 0;
    }

    glm::vec3 zStep = glm::vec3(actor.pos.x, actor.pos.y, candidatePos.z);
    if (!aabbIsColliding(zStep, actor.aabbDims, Chunks, ChunksMutex))
    {
        actor.pos.z = candidatePos.z;
    }
}

bool aabbIsColliding(glm::vec3 pos, glm::vec3 aabbDims, std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, std::mutex& ChunksMutex)
{
    glm::vec3 llb = pos - (aabbDims / 2.0f);
    glm::vec3 rtf = pos + (aabbDims / 2.0f);

    int startX = static_cast<int>(std::floor(llb.x));
    int endX = static_cast<int>(std::ceil(rtf.x));

    int startY = static_cast<int>(std::floor(llb.y));
    int endY = static_cast<int>(std::ceil(rtf.y));

    int startZ = static_cast<int>(std::floor(llb.z));
    int endZ = static_cast<int>(std::ceil(rtf.z));

    for (int x = startX; x < endX; ++x)
    {
        for (int y = startY; y < endY; ++y)
        {
            for (int z = startZ; z < endZ; ++z)
            {
                if (isSolid(tryReadBlock(glm::vec3(x, y, z), Chunks, ChunksMutex)))
                {
                    return true;
                }
            }
        }
    }

    return false;
}

bool blockInAABB(glm::vec3 blockPos, const Actor& actor)
{
    glm::vec3 min = actor.pos - (actor.aabbDims / 2.0f);
    glm::vec3 max = actor.pos + (actor.aabbDims / 2.0f);

    return !(blockPos.x + 1 < min.x || blockPos.x > max.x) && !(blockPos.y + 1 < min.y || blockPos.y > max.y) && !(blockPos.z + 1 < min.z || blockPos.z > max.z);
}

SelectedBlock castRayToBlock
(
    glm::vec3 rayOrigin, 
    glm::vec3 rayDirection, 
    float stepSize, 
    float maxRayLength, 
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, 
    std::mutex& ChunksMutex
)
{
    glm::vec3 rayStep = rayDirection * (stepSize / glm::length(rayDirection));
    
    glm::vec3 raySample{};
    Block selectedBlock{};

    glm::vec3 blockTraversalHistory[2] = { { glm::floor(rayOrigin) }, { glm::floor(rayOrigin) } }; // 0 = most recent

    glm::vec3 lastSample = rayOrigin;

    for (raySample = rayOrigin; glm::length(raySample - rayOrigin) < maxRayLength; raySample += rayStep)
    {
        glm::vec3 currentVoxel = glm::floor(raySample);
        glm::vec3 prevVoxel = glm::floor(lastSample);

        if ((int)(currentVoxel.x != prevVoxel.x) + (int)(currentVoxel.y != prevVoxel.y) + (int)(currentVoxel.z != prevVoxel.z) > 1)
        {
            glm::vec3 subVoxel = prevVoxel;

            for (int i = 0; i < 3; ++i)
            {
                if (subVoxel[i] != currentVoxel[i])
                {
                    subVoxel[i] = currentVoxel[i];

                    if (blockTraversalHistory[0] != subVoxel)
                    {
                        blockTraversalHistory[1] = blockTraversalHistory[0];
                        blockTraversalHistory[0] = subVoxel;
                    }

                    selectedBlock = tryReadBlock(subVoxel, Chunks, ChunksMutex);

                    if (selectedBlock != Block::AIR)
                    {
                        currentVoxel = subVoxel;
                        break;
                    }
                }
            }
        }

        if (blockTraversalHistory[0] != currentVoxel)
        {
            blockTraversalHistory[1] = blockTraversalHistory[0];
            blockTraversalHistory[0] = currentVoxel;
        }

        selectedBlock = tryReadBlock(currentVoxel, Chunks, ChunksMutex);

        lastSample = raySample;

        if (selectedBlock != Block::AIR)
        {
            glm::vec3 enteredFace;

            if (blockTraversalHistory[0].x > blockTraversalHistory[1].x)
            {
                enteredFace = LEFT_FACE;
            }
            else if (blockTraversalHistory[0].y > blockTraversalHistory[1].y)
            {
                enteredFace = BOTTOM_FACE;
            }
            else if (blockTraversalHistory[0].z > blockTraversalHistory[1].z)
            {
                enteredFace = BACK_FACE;
            }
            else if (blockTraversalHistory[0].x < blockTraversalHistory[1].x)
            {
                enteredFace = RIGHT_FACE;
            }
            else if (blockTraversalHistory[0].y < blockTraversalHistory[1].y)
            {
                enteredFace = TOP_FACE;
            }
            else if (blockTraversalHistory[0].z < blockTraversalHistory[1].z)
            {
                enteredFace = FRONT_FACE;
            }
            else
            {
                enteredFace = INSIDE_FACE;
            }


            return { selectedBlock, currentVoxel, enteredFace };
        }
    }

    return { selectedBlock, glm::floor(raySample) };
}

void bufferChunksFunc
(
    std::mutex& needsBufferingMutex, 
    std::mutex& ChunksMutex, 
    std::unordered_set<ChunkMeshes, ChunkMeshesHasher>& needsBuffering, 
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks
)
{
    std::scoped_lock lock{ needsBufferingMutex, ChunksMutex };

    while (!needsBuffering.empty())
    {
        ChunkMeshes unbufferedMeshes = *needsBuffering.begin();

        MeshHandle bufferedNonWaterMesh = bufferMesh(unbufferedMeshes.nonWaterMesh);
        MeshHandle bufferedWaterMesh = bufferMesh(unbufferedMeshes.waterMesh);

        Chunk& chunkObject = Chunks.at(unbufferedMeshes.origin);

        if (chunkObject.chunkMeshHandles.nonWaterMesh.VAO != 0)
        {
            glDeleteVertexArrays(1, &chunkObject.chunkMeshHandles.nonWaterMesh.VAO);
        }
        if (chunkObject.chunkMeshHandles.nonWaterMesh.VBO != 0)
        {
            glDeleteBuffers(1, &chunkObject.chunkMeshHandles.nonWaterMesh.VBO);
        }

        if (chunkObject.chunkMeshHandles.waterMesh.VAO != 0)
        {
            glDeleteVertexArrays(1, &chunkObject.chunkMeshHandles.waterMesh.VAO);
        }
        if (chunkObject.chunkMeshHandles.waterMesh.VBO != 0)
        {
            glDeleteBuffers(1, &chunkObject.chunkMeshHandles.waterMesh.VBO);
        }

        chunkObject.chunkMeshHandles.nonWaterMesh.VAO = bufferedNonWaterMesh.VAO;
        chunkObject.chunkMeshHandles.nonWaterMesh.VBO = bufferedNonWaterMesh.VBO;
        chunkObject.chunkMeshHandles.nonWaterMesh.numIndices = bufferedNonWaterMesh.numIndices;

        chunkObject.chunkMeshHandles.waterMesh.VAO = bufferedWaterMesh.VAO;
        chunkObject.chunkMeshHandles.waterMesh.VBO = bufferedWaterMesh.VBO;
        chunkObject.chunkMeshHandles.waterMesh.numIndices = bufferedWaterMesh.numIndices;

        needsBuffering.erase(needsBuffering.begin());
    }
}

void reloadChunksMinimal
(
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex,
    std::multiset<glm::vec3, decltype(&compareVec3)>& needsGenerating,
    std::mutex& needsGeneratingMutex,
    std::atomic<bool>& suspendMeshingThreads,
    std::atomic<uint8_t>& numMeshingThreadsWaiting,
    std::mutex& mainThreadCVMutex,
    std::condition_variable& mainThreadCV,
    std::condition_variable& meshingThreadCV,
    glm::vec3 playerPos,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsDecorating,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshing,
    std::unordered_set<ChunkMeshes, ChunkMeshesHasher>& needsBuffering
)
{
    std::cout << "[LOG] Called reloadChunksMinimal\n";

    static constexpr glm::vec3 chunkNeighbours[6] =
    {
        LEFT_FACE * 16.0f,
        RIGHT_FACE * 16.0f,
        BOTTOM_FACE * 16.0f,
        TOP_FACE * 16.0f,
        BACK_FACE * 16.0f,
        FRONT_FACE * 16.0f
    };

    playerChunkSnapshot = getChunkOfPositionVector(playerPos);
    std::unordered_set<glm::vec3, std::hash<glm::vec3>> newRenderingVolume = generateSphericalRenderingVolume(playerPos);

    suspendMeshingThreads = true;
    if (numMeshingThreadsWaiting.load() != NUM_MESHING_THREADS)
    {
        std::unique_lock<std::mutex> lock(mainThreadCVMutex);
        mainThreadCV.wait(lock, [&numMeshingThreadsWaiting] { return numMeshingThreadsWaiting.load() == NUM_MESHING_THREADS; });
    }

    {
        std::scoped_lock lock{ ChunksMutex };

        for (auto it = Chunks.begin(); it != Chunks.end();)
        {
            glm::vec3 origin = it->first;

            if (!newRenderingVolume.contains(origin))
            {
                Chunk& handle = it->second;

                if (handle.chunkMeshHandles.nonWaterMesh.VAO != 0)
                {
                    glDeleteVertexArrays(1, &handle.chunkMeshHandles.nonWaterMesh.VAO);
                }
                if (handle.chunkMeshHandles.nonWaterMesh.VBO != 0)
                {
                    glDeleteBuffers(1, &handle.chunkMeshHandles.nonWaterMesh.VBO);
                }

                if (handle.chunkMeshHandles.waterMesh.VAO != 0)
                {
                    glDeleteVertexArrays(1, &handle.chunkMeshHandles.waterMesh.VAO);
                }
                if (handle.chunkMeshHandles.waterMesh.VBO != 0)
                {
                    glDeleteBuffers(1, &handle.chunkMeshHandles.waterMesh.VBO);
                }

                it = Chunks.erase(it);

                for (size_t i = 0; i < 6; ++i)
                {
                    glm::vec3 neighbourOrigin = origin + chunkNeighbours[i];

                    if (Chunks.contains(neighbourOrigin))
                    {
                        needsMeshing.emplace(neighbourOrigin);
                    }
                }

                needsDecorating.erase(origin);
                needsMeshing.erase(origin);
                needsBuffering.erase({ {}, {}, origin });
            }
            else
            {
                ++it;
            }
        }
    }

    {
        std::scoped_lock lock{ needsGeneratingMutex };

        for (auto it = needsGenerating.begin(); it != needsGenerating.end();)
        {
            glm::vec3 origin = *it;

            if (!newRenderingVolume.contains(origin))
            {
                it = needsGenerating.erase(it);
            }
            else
            {
                ++it;
            }
        }

        for (auto& origin : newRenderingVolume)
        {
            if (!Chunks.contains(origin))
            {
                needsGenerating.emplace(origin);

                for (size_t i = 0; i < 6; ++i)
                {
                    glm::vec3 neighbourOrigin = origin + chunkNeighbours[i];

                    if (Chunks.contains(neighbourOrigin))
                    {
                        needsMeshing.emplace(neighbourOrigin);
                    }
                }
            }
        }
    }
    
    suspendMeshingThreads = false;
    meshingThreadCV.notify_all();
}

std::unordered_set<glm::vec3, std::hash<glm::vec3>> generateSphericalRenderingVolume(glm::vec3 playerPos)
{
    playerChunkSnapshot = getChunkOfPositionVector(playerPos);

    glm::vec3 renderVolumeMin =
    {
        playerChunkSnapshot.x - (RENDER_DIST * CHUNK_W),
        playerChunkSnapshot.y - (RENDER_DIST * CHUNK_H),
        playerChunkSnapshot.z - (RENDER_DIST * CHUNK_L)
    };

    glm::vec3 renderVolumeMax =
    {
        playerChunkSnapshot.x + (RENDER_DIST * CHUNK_W),
        playerChunkSnapshot.y + (RENDER_DIST * CHUNK_H),
        playerChunkSnapshot.z + (RENDER_DIST * CHUNK_L)
    };

    std::unordered_set<glm::vec3, std::hash<glm::vec3>> chunksInRenderingVolume{};

    for (int x = renderVolumeMin.x; x < renderVolumeMax.x; x += CHUNK_W)
    {
        for (int y = renderVolumeMin.y; y < renderVolumeMax.y; y += CHUNK_H)
        {
            for (int z = renderVolumeMin.z; z < renderVolumeMax.z; z += CHUNK_L)
            {
                glm::vec3 chunkOrigin = glm::vec3(x, y, z);
                glm::vec3 distVec = chunkOrigin - playerChunkSnapshot;

                float squaredDist = glm::dot(distVec, distVec);

                if (squaredDist <= (RENDER_DIST * CHUNK_W) * (RENDER_DIST * CHUNK_W))
                {
                    chunksInRenderingVolume.emplace(chunkOrigin);
                }
            }
        }
    }

    return chunksInRenderingVolume;
}

void clearMeshes(std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks)
{
    for (auto it = Chunks.begin(); it != Chunks.end(); )
    {
        Chunk& handle = it->second;

        if (handle.chunkMeshHandles.nonWaterMesh.VAO != 0)
        {
            glDeleteVertexArrays(1, &handle.chunkMeshHandles.nonWaterMesh.VAO);
        }
        if (handle.chunkMeshHandles.nonWaterMesh.VBO != 0)
        {
            glDeleteBuffers(1, &handle.chunkMeshHandles.nonWaterMesh.VBO);
        }

        if (handle.chunkMeshHandles.waterMesh.VAO != 0)
        {
            glDeleteVertexArrays(1, &handle.chunkMeshHandles.waterMesh.VAO);
        }
        if (handle.chunkMeshHandles.waterMesh.VBO != 0)
        {
            glDeleteBuffers(1, &handle.chunkMeshHandles.waterMesh.VBO);
        }

        it = Chunks.erase(it);
    }
}

void needsGeneratingAllSphere
(
    std::multiset<glm::vec3, decltype(&compareVec3)>& needsGenerating,
    glm::vec3 playerPos
)
{
    playerChunkSnapshot = getChunkOfPositionVector(playerPos);

    glm::vec3 renderVolumeMin =
    {
        playerChunkSnapshot.x - (RENDER_DIST * CHUNK_W),
        playerChunkSnapshot.y - (RENDER_DIST * CHUNK_H),
        playerChunkSnapshot.z - (RENDER_DIST * CHUNK_L)
    };

    glm::vec3 renderVolumeMax =
    {
        playerChunkSnapshot.x + (RENDER_DIST * CHUNK_W),
        playerChunkSnapshot.y + (RENDER_DIST * CHUNK_H),
        playerChunkSnapshot.z + (RENDER_DIST * CHUNK_L)
    };

    for (int x = renderVolumeMin.x; x < renderVolumeMax.x; x += CHUNK_W)
    {
        for (int y = renderVolumeMin.y; y < renderVolumeMax.y; y += CHUNK_H)
        {
            for (int z = renderVolumeMin.z; z < renderVolumeMax.z; z += CHUNK_L)
            {
                glm::vec3 chunkOrigin = glm::vec3(x, y, z);
                glm::vec3 distVec = chunkOrigin - playerChunkSnapshot;

                float squaredDist = glm::dot(distVec, distVec);

                if (squaredDist <= (RENDER_DIST * CHUNK_W) * (RENDER_DIST * CHUNK_W))
                {
                    needsGenerating.emplace(chunkOrigin);
                }
            }
        }
    }
}

void meshingThread
(
    std::atomic<bool>& suspendMeshingThreads, 
    std::atomic<uint8_t>& numMeshingThreadsWaiting, 
    std::condition_variable& mainThreadCV, 
    std::mutex& ChunksMutex, 
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, 
    std::unordered_set<ChunkMeshes, ChunkMeshesHasher>& needsBuffering, 
    std::multiset<glm::vec3, decltype(&compareVec3)>& needsGenerating, 
    std::mutex& needsGeneratingMutex, 
    std::mutex& needsBufferingMutex, 
    std::condition_variable& meshingThreadCV, 
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshing, 
    std::atomic<Pass>& currentPass,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsMeshingHighPriority,
    std::mutex& needsMeshingHPMutex,
    std::unordered_set<glm::vec3, std::hash<glm::vec3>>& needsDecorating,
    std::mutex& needsDecoratingMutex,
    GLFWwindow* window
)
{
    bool thisThreadIsSuspended = false;
    static std::mutex meshingThreadCVMutex{};
    static std::mutex needsMeshingMutex{};

    static FastNoiseLite HeightGen;
    HeightGen.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    HeightGen.SetSeed(1337); // 1337 is the default FastNoiseLite seed
    HeightGen.SetFrequency(0.020);
    HeightGen.SetFractalType(FastNoiseLite::FractalType_FBm);
    HeightGen.SetFractalOctaves(5);
    HeightGen.SetFractalLacunarity(2.00);
    HeightGen.SetFractalGain(0.50);
    HeightGen.SetFractalWeightedStrength(0.00);

    static FastNoiseLite ErosionGen;
    ErosionGen.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    ErosionGen.SetSeed(1337);
    ErosionGen.SetFrequency(0.005);
    ErosionGen.SetFractalType(FastNoiseLite::FractalType_FBm);
    ErosionGen.SetFractalOctaves(4);
    ErosionGen.SetFractalLacunarity(1.00);
    ErosionGen.SetFractalGain(1.00);
    ErosionGen.SetFractalWeightedStrength(0.00);


    while (!glfwWindowShouldClose(window))
    {
        std::optional<glm::vec3> chunkOrigin;

        {
            std::unique_lock<std::mutex> lock(meshingThreadCVMutex);

            meshingThreadCV.wait(lock,
                [&]
                {
                    bool predicate = ((!needsGenerating.empty() || !needsDecorating.empty() || !needsMeshing.empty() || !needsMeshingHighPriority.empty()) && !suspendMeshingThreads) || programEnded;
                    if (!predicate && !thisThreadIsSuspended)
                    {
                        thisThreadIsSuspended = true;
                        ++numMeshingThreadsWaiting;

                        currentPass = Pass::IDLING;

                        if (numMeshingThreadsWaiting == NUM_MESHING_THREADS)
                        {
                            mainThreadCV.notify_all();
                        }
                    }
                    if (predicate && thisThreadIsSuspended)
                    {
                        thisThreadIsSuspended = false;

                        if (currentPass == Pass::IDLING)
                        {
                            // Thread notified and pass not manually set - defaults to first pass (Generating)
                            currentPass = Pass::GENERATING;
                        }

                        --numMeshingThreadsWaiting;
                    }
                    return predicate;
                });
        }

        if (programEnded)
        {
            break;
        }

        if (!needsMeshingHighPriority.empty()) // Currently only used for block placing/destruction
        {
            chunkOrigin.reset();

            {
                std::scoped_lock lock{ needsMeshingHPMutex };

                if (!needsMeshingHighPriority.empty())
                {
                    auto it = needsMeshingHighPriority.begin();
                    chunkOrigin = *it;
                    needsMeshingHighPriority.erase(it);
                }
            }

            if (chunkOrigin)
            {
                ChunkMeshes chunkMeshes = buildMeshFromChunkData(Chunks, Chunks.at(*chunkOrigin).chunkData, *chunkOrigin);;

                {
                    std::scoped_lock lock{ needsBufferingMutex };

                    needsBuffering.emplace(chunkMeshes);
                }
            }
        }

        if (currentPass == Pass::GENERATING && !needsGenerating.empty())
        {
            chunkOrigin.reset();

            {
                std::scoped_lock lock{ needsGeneratingMutex };

                if (currentPass == Pass::GENERATING && !needsGenerating.empty())
                {
                    auto it = needsGenerating.begin();
                    chunkOrigin = *it;
                    needsGenerating.erase(it);
                }
            }

            if (chunkOrigin)
            {
                std::array<Block, CHUNK_W * CHUNK_H * CHUNK_L> terrain = normalTerrain(*chunkOrigin, HeightGen, ErosionGen);

                {
                    std::scoped_lock lock{ ChunksMutex };

                    Chunks.try_emplace(*chunkOrigin, Chunk{ {}, *chunkOrigin, terrain });
                }

                {
                    std::scoped_lock lock{ needsDecoratingMutex };

                    needsDecorating.emplace(*chunkOrigin);
                }

                if (needsGenerating.empty() && currentPass == Pass::GENERATING)
                {
                    currentPass = Pass::DECORATING;
                }
            }
        }

        if (currentPass == Pass::DECORATING && !needsDecorating.empty())
        {
            chunkOrigin.reset();

            {
                std::scoped_lock lock{ needsDecoratingMutex };

                if (currentPass == Pass::DECORATING && !needsDecorating.empty())
                {
                    auto it = needsDecorating.begin();
                    chunkOrigin = *it;
                    needsDecorating.erase(it);
                }
            }

            if (chunkOrigin)
            {
                std::unordered_set<glm::vec3, std::hash<glm::vec3>> affectedChunks = decorateChunkNormalTerrain(Chunks.at(*chunkOrigin).chunkData, Chunks, ChunksMutex, HeightGen, ErosionGen, *chunkOrigin);

                {
                    std::scoped_lock lock{ needsMeshingMutex };

                    needsMeshing.merge(affectedChunks);
                }

                if (needsDecorating.empty() && currentPass == Pass::DECORATING)
                {
                    currentPass = Pass::MESHING;
                }
            }
        }

        if (currentPass == Pass::MESHING && !needsMeshing.empty())
        {
            chunkOrigin.reset();

            {
                std::scoped_lock lock{ needsMeshingMutex };

                if (currentPass == Pass::MESHING && !needsMeshing.empty())
                {
                    auto it = needsMeshing.begin();
                    chunkOrigin = *it;
                    needsMeshing.erase(it);
                }
            }

            if (chunkOrigin)
            {
                ChunkMeshes chunkMeshes = buildMeshFromChunkData(Chunks, Chunks.at(*chunkOrigin).chunkData, *chunkOrigin);

                {
                    std::scoped_lock lock{ needsBufferingMutex };

                    needsBuffering.emplace(chunkMeshes);
                }

                if (needsMeshing.empty() && needsDecorating.empty() && currentPass == Pass::MESHING)
                {
                    currentPass = Pass::IDLING;
                }
            }
        }
    }
}

ChunkMeshes buildMeshFromChunkData
(
    const std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, 
    const std::array<Block, CHUNK_W * CHUNK_H * CHUNK_L>& chunkData, 
    const glm::vec3 chunkOrigin
)
{
    ChunkMeshes chunkMeshes{};
    chunkMeshes.origin = chunkOrigin;

    chunkMeshes.nonWaterMesh.vertices.reserve(CHUNK_W * CHUNK_H * CHUNK_L * 6 * 4); // 6 faces per cube, 4 vertices per face
    size_t nonWaterMeshIndex = 0;

    chunkMeshes.waterMesh.vertices.reserve(CHUNK_W * CHUNK_H * CHUNK_L * 6 * 4);
    size_t waterMeshIndex = 0;

    for (uint8_t x = 0; x < CHUNK_W; ++x)
    {
        for (uint8_t z = 0; z < CHUNK_L; ++z)
        {
            for (uint8_t y = 0; y < CHUNK_H; ++y)
            {
                glm::vec3 localChunkPos(x, y, z);
                size_t index = localChunkPosToIndex16cubed(localChunkPos);
                Block blockEnum = chunkData.at(index);

                if (blockEnum == Block::WATER)
                {
                    std::bitset<6> occludedFaces = getOccludedFaces(Chunks, localChunkPos, chunkOrigin, chunkData);

                    addVisibleFaces(occludedFaces, chunkMeshes.waterMesh, waterMeshIndex, localChunkPos, blockEnum, chunkOrigin);
                }
                else if (isBillboard(blockEnum))
                {
                    addBillboard(chunkMeshes.nonWaterMesh, nonWaterMeshIndex, localChunkPos, chunkOrigin, blockEnum, getSideFaceTexture(blockEnum));
                }
                else if (blockEnum != Block::AIR)
                {
                    std::bitset<6> occludedFaces = getOccludedFaces(Chunks, localChunkPos, chunkOrigin, chunkData);

                    addVisibleFaces(occludedFaces, chunkMeshes.nonWaterMesh, nonWaterMeshIndex, localChunkPos, blockEnum, chunkOrigin);
                }
            }
        }
    }

    return chunkMeshes;
}

constexpr uint32_t packAttributes32(int posX, int posY, int posZ, int brightnessLookup, int uvX, int uvY, int uvZ)
{
    uint32_t packedAttrib = 0;

    packedAttrib |= (posX & 0x1F);                   // 0-4 bits
    packedAttrib |= (posY & 0x1F) << 5;              // 5-9 bits
    packedAttrib |= (posZ & 0x1F) << 10;             // 10-14 bits
    packedAttrib |= (brightnessLookup & 0x03) << 15; // 15-16 bits
    packedAttrib |= (uvX & 0x01) << 17;              // 17th bit
    packedAttrib |= (uvY & 0x01) << 18;              // 18th bit
    packedAttrib |= (uvZ & 0xFF) << 19;              // 19-26 bits

    return packedAttrib;
}

void addBillboard(Mesh& mesh, size_t& currentMeshIndex, const glm::vec3& localChunkPos, const glm::vec3& chunkOrigin, const Block blockEnum, const uint8_t texture2DArrayDepth)
{
    static constexpr int faces[2][4][7] =
    {
        {
            { 0, 0, 1,  3,  0, 0, 0 },
            { 1, 0, 0,  3,  1, 0, 0 },
            { 0, 1, 1,  3,  0, 1, 0 },
            { 1, 1, 0,  3,  1, 1, 0 }
        },                 
                           
        {                  
            { 0, 0, 0,  3,  0, 0, 0 },
            { 1, 0, 1,  3,  1, 0, 0 },
            { 0, 1, 0,  3,  0, 1, 0 },
            { 1, 1, 1,  3,  1, 1, 0 }
        }
    };

    static constexpr int order[] = 
    { 
        0, 1, 3,
        3, 2, 0,
        3, 1, 0,
        0, 2, 3
    };

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 12; ++j)
        {
            size_t idx = order[j];
            const int* v = faces[i][idx];

            mesh.vertices.emplace_back
            (
                packAttributes32
                (
                    v[0] + localChunkPos.x,
                    v[1] + localChunkPos.y,
                    v[2] + localChunkPos.z,
                    v[3],
                    v[4],
                    v[5],
                    texture2DArrayDepth
                )
            );
        }
    }
}

/* 
* Bit 0 = left    (-x)
* Bit 1 = below   (-y)
* Bit 2 = behind  (-z)
* Bit 3 = right   (+x)
* Bit 4 = above   (+y)
* Bit 5 = infront (+z) */
void addVisibleFaces(const std::bitset<6>& occludedFaces, Mesh& mesh, size_t& meshIndex, const glm::vec3& localChunkPos, const Block blockEnum, const glm::vec3& chunkOrigin)
{
    if (!occludedFaces[0])
    {
        addFace(mesh, meshIndex, localChunkPos, getSideFaceTexture(blockEnum), FaceDirection::LEFT, chunkOrigin);
    }
    if (!occludedFaces[1])
    {
        addFace(mesh, meshIndex, localChunkPos, getBottomFaceTexture(blockEnum), FaceDirection::BOTTOM, chunkOrigin);
    }
    if (!occludedFaces[2])
    {
        addFace(mesh, meshIndex, localChunkPos, getSideFaceTexture(blockEnum), FaceDirection::BACK, chunkOrigin);
    }
    if (!occludedFaces[3])
    {
        addFace(mesh, meshIndex, localChunkPos, getSideFaceTexture(blockEnum), FaceDirection::RIGHT, chunkOrigin);
    }
    if (!occludedFaces[4])
    {
        addFace(mesh, meshIndex, localChunkPos, getTopFaceTexture(blockEnum), FaceDirection::TOP, chunkOrigin);
    }
    if (!occludedFaces[5])
    {
        addFace(mesh, meshIndex, localChunkPos, getSideFaceTexture(blockEnum), FaceDirection::FRONT, chunkOrigin);
    }
}

void addFace(Mesh& mesh, size_t& currentMeshIndex, const glm::vec3& pos, const uint8_t texture2DArrayDepth, FaceDirection dir, const glm::vec3& chunkOrigin)
{
    static constexpr int faces[6][4][7] =
    {
        {
            { 0, 0, 1,  2,  0, 0, 0 },
            { 0, 0, 0,  2,  1, 0, 0 },
            { 0, 1, 1,  2,  0, 1, 0 },
            { 0, 1, 0,  2,  1, 1, 0 }
        },

        {
            { 1, 0, 0,  2,  0, 0, 0 },
            { 1, 0, 1,  2,  1, 0, 0 },
            { 1, 1, 0,  2,  0, 1, 0 },
            { 1, 1, 1,  2,  1, 1, 0 }
        },

        {
            { 1, 0, 0,  0,  1, 1, 0 },
            { 0, 0, 0,  0,  0, 1, 0 },
            { 1, 0, 1,  0,  1, 0, 0 },
            { 0, 0, 1,  0,  0, 0, 0 }
        },

        {
            { 1, 1, 1,  3,  1, 0, 0 },
            { 0, 1, 1,  3,  0, 0, 0 },
            { 1, 1, 0,  3,  1, 1, 0 },
            { 0, 1, 0,  3,  0, 1, 0 }
        },

        {
            { 0, 0, 0,  1,  0, 0, 0 },
            { 1, 0, 0,  1,  1, 0, 0 },
            { 0, 1, 0,  1,  0, 1, 0 },
            { 1, 1, 0,  1,  1, 1, 0 }
        },

        {
            { 1, 0, 1,  1,  0, 0, 0 },
            { 0, 0, 1,  1,  1, 0, 0 },
            { 1, 1, 1,  1,  0, 1, 0 },
            { 0, 1, 1,  1,  1, 1, 0 }
        }
    };

    static constexpr int order[] =
    {
        0, 1, 3,
        3, 2, 0
    };

    for (size_t i = 0; i < 6; ++i)
    {
        size_t idx = order[i];
        const int* v = faces[(size_t)dir][idx];

        mesh.vertices.emplace_back
        (
            packAttributes32
            (
                v[0] + pos.x,
                v[1] + pos.y,
                v[2] + pos.z,
                v[3],
                v[4],
                v[5],
                texture2DArrayDepth
            )
        );
    }
}

std::bitset<6> getOccludedFaces
(
    const std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    const glm::vec3& localCoords,
    const glm::vec3& origin,
    const std::array<Block, CHUNK_W* CHUNK_H* CHUNK_L>& chunkData
)
{
    std::bitset<6> neighbours;
    static const glm::vec3 chunkDims = { CHUNK_W, CHUNK_H, CHUNK_L };

    Block thisBlock = chunkData.at(localChunkPosToIndex16cubed(localCoords));

    for (int j = 0; j < 3; ++j)
    {
        if (localCoords[j] == 0)
        {
            glm::vec3 o = origin;
            o[j] -= chunkDims[j];

            glm::vec3 l = localCoords;
            l[j] = chunkDims[j] - 1;
            int idx = localChunkPosToIndex16cubed(l);
            
            auto it = Chunks.find(o);
            Block neighbourBlock;
            bool inRange = it != Chunks.end();

            if (inRange)
            {
                neighbourBlock = it->second.chunkData.at(idx);
            }

            neighbours[j] = inRange && (!isTransparent(neighbourBlock) || (isSelfOccludingBlock(neighbourBlock) && thisBlock == neighbourBlock));
        }
        else
        {
            glm::vec3 l = localCoords;
            --l[j];
            int idx = localChunkPosToIndex16cubed(l);
            Block neighbourBlock = chunkData.at(idx);

            neighbours[j] = !isTransparent(neighbourBlock) || (isSelfOccludingBlock(neighbourBlock) && thisBlock == neighbourBlock);
        }
    }

    for (int j = 0; j < 3; ++j)
    {
        if (localCoords[j] == chunkDims[j] - 1)
        {
            glm::vec3 o = origin;
            o[j] += chunkDims[j];

            glm::vec3 l = localCoords;
            l[j] = 0;
            int idx = localChunkPosToIndex16cubed(l);

            auto it = Chunks.find(o);
            Block neighbourBlock;
            bool inRange = it != Chunks.end();

            if (inRange)
            {
                neighbourBlock = it->second.chunkData.at(idx);
            }

            neighbours[j + 3] = inRange && (!isTransparent(neighbourBlock) || (isSelfOccludingBlock(neighbourBlock) && thisBlock == neighbourBlock));
        }
        else
        {
            glm::vec3 l = localCoords;
            ++l[j];
            int idx = localChunkPosToIndex16cubed(l);
            Block neighbourBlock = chunkData.at(idx);

            neighbours[j + 3] = !isTransparent(neighbourBlock) || (isSelfOccludingBlock(neighbourBlock) && thisBlock == neighbourBlock);
        }
    }

    return neighbours;
}

int localChunkPosToIndex16cubed(const glm::vec3& v)
{
    return (int)v.y | (int)v.z << 4 | (int)v.x << 8;
}

MeshHandle bufferMesh(const Mesh& mesh)
{
    GLuint VAO = 0;
    GLuint VBO = 0;

    if (mesh.vertices.size() > 0)
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glEnableVertexAttribArray(0);
        glVertexAttribIPointer(0, 1, GL_UNSIGNED_INT, 0, 0);

        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(Vertex), mesh.vertices.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    return { VAO, VBO, mesh.vertices.size() };
}

void draw(const MeshHandle& meshHandle)
{
    glBindVertexArray(meshHandle.VAO);
    glDrawArrays(GL_TRIANGLES, 0, meshHandle.numIndices);
    glBindVertexArray(0);

    ++drawCallsThisFrame;
}

bool compareVec3(const glm::vec3& a, const glm::vec3& b)
{
    float squaredDistanceA =
        (
            ((a.x - playerChunkSnapshot.x) * (a.x - playerChunkSnapshot.x))
            + ((a.y - playerChunkSnapshot.y) * (a.y - playerChunkSnapshot.y))
            + ((a.z - playerChunkSnapshot.z) * (a.z - playerChunkSnapshot.z))
        );
    float squaredDistanceB =
        (
            ((b.x - playerChunkSnapshot.x) * (b.x - playerChunkSnapshot.x))
            + ((b.y - playerChunkSnapshot.y) * (b.y - playerChunkSnapshot.y))
            + ((b.z - playerChunkSnapshot.z) * (b.z - playerChunkSnapshot.z))
        );
    return squaredDistanceA < squaredDistanceB;
}

GLuint loadTexture(char const* path, GLenum wrapParam)
{
    GLuint textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1) format = GL_RED;
        else if (nrComponents == 3) format = GL_RGB;
        else if (nrComponents == 4) format = GL_RGBA;
        else
        {
            std::cerr << "Unsupported number of channels: " << nrComponents << std::endl;
            return 0; // Return invalid texture ID
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapParam);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapParam);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

GLuint loadTexturesIntoArray(const std::vector<const char*>& paths)
{
    stbi_set_flip_vertically_on_load(true);

    if (paths.empty())
    {
        std::cerr << "No paths provided!" << std::endl;
        return 0;
    }

    GLuint textureArrayID;
    glGenTextures(1, &textureArrayID);
    glBindTexture(GL_TEXTURE_2D_ARRAY, textureArrayID);

    int width, height, nrComponents;

    unsigned char* firstImage = stbi_load(paths[0], &width, &height, &nrComponents, 0);
    if (!firstImage)
    {
        std::cerr << "First texture failed to load at path: " << paths[0] << std::endl;
        return 0;
    }
    stbi_image_free(firstImage);

    GLenum format;
    if (nrComponents == 1) format = GL_RED;
    else if (nrComponents == 3) format = GL_RGB;
    else if (nrComponents == 4) format = GL_RGBA;
    else
    {
        std::cerr << "Unsupported number of channels: " << nrComponents << std::endl;
        return 0;
    }

    // Preallocate storage for the Texture2DArray
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, format, width, height, paths.size(), 0, format, GL_UNSIGNED_BYTE, NULL);

    for (size_t i = 0; i < paths.size(); ++i)
    {
        unsigned char* data = stbi_load(paths[i], &width, &height, &nrComponents, 0);
        if (data)
        {
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, width, height, 1, format, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            std::cerr << "Texture failed to load at path: " << paths[i] << std::endl;
        }
    }

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

    return textureArrayID;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    viewportSizeChanged = true;

    WIDTH = width;
    HEIGHT = height;

    glViewport(0, 0, width, height);
}

void mouseCallback(GLFWwindow* window, double xposIn, double yposIn)
{
    if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL)
    {
        return;
    }

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    selectedItem = (selectedItem + 9 - static_cast<int>(yoffset)) % 9;
}

GLFWwindow* initOpenGL()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Minecraft Clone", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSwapInterval(0);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(-1);
    }

    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);

    glLogicOp(GL_XOR);

    glClearColor(skyClearColor.r, skyClearColor.g, skyClearColor.b, skyClearColor.a);

    return window;
}

std::array<Block, CHUNK_W * CHUNK_H * CHUNK_L> normalTerrain(const glm::vec3& chunkOrigin, FastNoiseLite& HeightGen, FastNoiseLite& ErosionGen)
{
    std::array<Block, CHUNK_W* CHUNK_H* CHUNK_L> chunkData;
    glm::vec3 worldPos = chunkOrigin;
    static constexpr int seaLevel = -12;

    for (int x = 0; x < CHUNK_W; ++x)
    {
        worldPos.x = chunkOrigin.x + x;

        for (int z = 0; z < CHUNK_L; ++z)
        {
            worldPos.z = chunkOrigin.z + z;
            
            int terrainHeight = getTerrainHeight(worldPos.x, worldPos.z, HeightGen, ErosionGen);

            for (int y = 0; y < CHUNK_H; ++y)
            {
                worldPos.y = chunkOrigin.y + y;

                size_t idx = localChunkPosToIndex16cubed({ x, y, z });

                int wpy = worldPos.y;

                if (wpy < terrainHeight - 1)
                {
                    chunkData.at(idx) = Block::STONE;
                }
                else if (wpy < terrainHeight)
                {
                    if (terrainHeight > seaLevel)
                    {
                        chunkData.at(idx) = Block::DIRT;
                    }
                    else
                    {
                        chunkData.at(idx) = Block::SAND;
                    }
                }
                else if (wpy == terrainHeight)
                {
                    if (wpy < seaLevel - 1)
                    {
                        if (randomChance(0.8))
                        {
                            chunkData.at(idx) = Block::SAND;
                        }
                        else
                        {
                            chunkData.at(idx) = Block::GRAVEL;
                        }
                    }
                    else if (wpy < seaLevel + 2)
                    {
                        chunkData.at(idx) = Block::SAND;
                    }
                    else
                    {
                        chunkData.at(idx) = Block::GRASS;
                    }
                }
                else if (wpy < seaLevel)
                {
                    chunkData.at(idx) = Block::WATER;
                }
                else
                {
                    chunkData.at(idx) = Block::AIR;
                }
            }
        }
    }

    return chunkData;
}

int getTerrainHeight(int x, int z, FastNoiseLite& HeightGen, FastNoiseLite& ErosionGen)
{
    int terrainHeight = HeightGen.GetNoise((float)x, (float)z) * 50;
    float erosion = ErosionGen.GetNoise((float)x, (float)z);

    if (erosion > 0.3f)
    {
        terrainHeight += (erosion - 0.3f) * 256.0f;
    }
    else if (erosion < -0.4f)
    {
        terrainHeight -= (erosion + 0.4f) * 512.0f;
    }

    return terrainHeight;
}

glm::vec3 getChunkOfPositionVector(glm::vec3 pos)
{
    glm::vec3 normalizedChunkOrigin = glm::floor(glm::vec3((pos.x / CHUNK_W), (pos.y / CHUNK_H), (pos.z / CHUNK_L)));
    glm::vec3 chunkOrigin = { CHUNK_W * normalizedChunkOrigin.x, CHUNK_H * normalizedChunkOrigin.y, CHUNK_L * normalizedChunkOrigin.z };
    return chunkOrigin;
}

// Onus on the caller to queue affected chunks for meshing
std::unordered_set<glm::vec3, std::hash<glm::vec3>> tryReplaceBlock
(
    const glm::vec3& targetBlockWorldPos, 
    const Block targetBlockNewValue, 
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, 
    std::mutex& ChunksMutex
)
{
    std::unordered_set<glm::vec3> affectedChunks{};

    const glm::vec3 targetChunkPos = getChunkOfPositionVector(targetBlockWorldPos);
    const glm::uvec3 targetBlockLocalPos = targetBlockWorldPos - targetChunkPos;
    const int targetBlockIndex = localChunkPosToIndex16cubed(targetBlockLocalPos);

    {
        std::scoped_lock lock{ ChunksMutex };

        if (Chunks.contains(targetChunkPos))
        {
            Chunks.at(targetChunkPos).chunkData.at(targetBlockIndex) = targetBlockNewValue;
        }
    }

    if (Chunks.contains(targetChunkPos))
    {
        affectedChunks.insert(targetChunkPos);

        glm::vec3 leftChunk = { targetChunkPos.x - CHUNK_W, targetChunkPos.y, targetChunkPos.z };
        glm::vec3 rightChunk = { targetChunkPos.x + CHUNK_W, targetChunkPos.y, targetChunkPos.z };
        glm::vec3 belowChunk = { targetChunkPos.x, targetChunkPos.y - CHUNK_H, targetChunkPos.z };
        glm::vec3 aboveChunk = { targetChunkPos.x, targetChunkPos.y + CHUNK_H, targetChunkPos.z };
        glm::vec3 behindChunk = { targetChunkPos.x, targetChunkPos.y, targetChunkPos.z - CHUNK_L };
        glm::vec3 infrontChunk = { targetChunkPos.x, targetChunkPos.y, targetChunkPos.z + CHUNK_L };

        if (targetBlockLocalPos.x == 0 && Chunks.contains(leftChunk))
        {
            affectedChunks.insert(leftChunk);
        }
        else if (targetBlockLocalPos.x == CHUNK_W - 1 && Chunks.contains(rightChunk))
        {
            affectedChunks.insert(rightChunk);
        }

        if (targetBlockLocalPos.y == 0 && Chunks.contains(belowChunk))
        {
            affectedChunks.insert(belowChunk);
        }
        else if (targetBlockLocalPos.y == CHUNK_H - 1 && Chunks.contains(aboveChunk))
        {
            affectedChunks.insert(aboveChunk);
        }

        if (targetBlockLocalPos.z == 0 && Chunks.contains(behindChunk))
        {
            affectedChunks.insert(behindChunk);
        }
        else if (targetBlockLocalPos.z == CHUNK_L - 1 && Chunks.contains(infrontChunk))
        {
            affectedChunks.insert(infrontChunk);
        }
    }

    return affectedChunks;
}

Block tryReadBlock(const glm::vec3& targetBlockWorldPos, std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks, std::mutex& ChunksMutex)
{
    glm::vec3 targetBlock = glm::floor(targetBlockWorldPos);
    const glm::vec3 targetChunkPos = getChunkOfPositionVector(targetBlock);

    {
        std::scoped_lock lock{ ChunksMutex };

        if (Chunks.contains(targetChunkPos))
        {
            Chunk& targetChunk = Chunks.at(targetChunkPos);

            glm::uvec3 targetBlockLocalPos = targetBlock - targetChunkPos;

            int targetBlockIndex = localChunkPosToIndex16cubed(targetBlockLocalPos);
            return targetChunk.chunkData.at(targetBlockIndex);
        }
    }

    return Block::AIR;
}

int randomIntInRange(int min, int max)
{
    return std::uniform_int_distribution<>{min, max}(gen);
}

float randomFloatInRange(float min, float max)
{
    return std::uniform_real_distribution<float>{min, max}(gen);
}

bool randomChance(float decimalProbability)
{
    return std::uniform_real_distribution<float>{0.0f, 1.0f}(gen) <= decimalProbability;
}

// Onus on the caller to queue affected chunks for meshing
std::unordered_set<glm::vec3, std::hash<glm::vec3>> createTree
(
    const glm::vec3& targetBlockWorldPos, 
    std::unordered_map<glm::vec3, Chunk, 
    std::hash<glm::vec3>>& Chunks, 
    std::mutex& ChunksMutex
)
{
    std::unordered_set<glm::vec3, std::hash<glm::vec3>> affectedChunks{};

    int treeTrunkHeight = randomIntInRange(3, 5);
    int leavesHeight = randomIntInRange(2, 3);

    // Lower tier
    for (int x = (targetBlockWorldPos.x - 2); x <= (targetBlockWorldPos.x + 2); ++x)
    {
        for (int z = (targetBlockWorldPos.z - 2); z <= (targetBlockWorldPos.z + 2); ++z)
        {
            for (int y = (targetBlockWorldPos.y + treeTrunkHeight); y <= (targetBlockWorldPos.y + treeTrunkHeight + 1); ++y)
            {
                bool corner = (std::abs(x - targetBlockWorldPos.x) == 2 && std::abs(z - targetBlockWorldPos.z) == 2);

                if ((!(x == targetBlockWorldPos.x && z == targetBlockWorldPos.z) || y > (targetBlockWorldPos.y + treeTrunkHeight)) && !(corner && y == (targetBlockWorldPos.y + treeTrunkHeight + 1) && randomChance(0.4)))
                {
                    affectedChunks.merge(tryReplaceBlock({ x, y, z }, Block::OAK_LEAVES, Chunks, ChunksMutex));
                }
            }
        }
    }

    // Upper tier
    for (int x = (targetBlockWorldPos.x - 1); x <= (targetBlockWorldPos.x + 1); ++x)
    {
        for (int z = (targetBlockWorldPos.z - 1); z <= (targetBlockWorldPos.z + 1); ++z)
        {
            for (int y = (targetBlockWorldPos.y + treeTrunkHeight + 2); y <= (targetBlockWorldPos.y + treeTrunkHeight + leavesHeight); ++y)
            {
                bool corner = (std::abs(x - targetBlockWorldPos.x) == 1 && std::abs(z - targetBlockWorldPos.z) == 1);

                if (!(corner && y == (targetBlockWorldPos.y + treeTrunkHeight + leavesHeight) && randomChance(0.8)))
                {
                    affectedChunks.merge(tryReplaceBlock({ x, y, z }, Block::OAK_LEAVES, Chunks, ChunksMutex));
                }
            }
        }
    }

    // Trunk
    int trunkTop = leavesHeight - 1;
    for (int y = targetBlockWorldPos.y; y < (targetBlockWorldPos.y + treeTrunkHeight + trunkTop); ++y)
    {
        affectedChunks.merge(tryReplaceBlock({ targetBlockWorldPos.x, y, targetBlockWorldPos.z }, Block::OAK, Chunks, ChunksMutex));
    }

    return affectedChunks;
}

std::unordered_set<glm::vec3, std::hash<glm::vec3>> decorateChunkNormalTerrain
(
    std::array<Block, CHUNK_W* CHUNK_H* CHUNK_L>& chunkData,
    std::unordered_map<glm::vec3, Chunk, std::hash<glm::vec3>>& Chunks,
    std::mutex& ChunksMutex,
    FastNoiseLite& HeightGen,
    FastNoiseLite& ErosionGen,
    const glm::vec3& sourceChunkOrigin
)
{
    std::unordered_set<glm::vec3, std::hash<glm::vec3>> affectedChunks{};

    affectedChunks.insert(sourceChunkOrigin); // Every chunk in needsDecorating must advance to needsMeshing

    for (int x = 0; x < CHUNK_W; ++x)
    {
        for (int z = 0; z < CHUNK_L; ++z)
        {
            int terrainHeight = getTerrainHeight(x + sourceChunkOrigin.x, z + sourceChunkOrigin.z, HeightGen, ErosionGen);

            if (terrainHeight > sourceChunkOrigin.y + CHUNK_H || terrainHeight < sourceChunkOrigin.y) // Only decorate the surface
            {
                continue;
            }

            if (randomChance(0.005) && tryReadBlock({x + sourceChunkOrigin.x, terrainHeight, z + sourceChunkOrigin.z }, Chunks, ChunksMutex) == Block::GRASS)
            {
                glm::vec3 worldPos = glm::vec3(x + sourceChunkOrigin.x, terrainHeight + 1, z + sourceChunkOrigin.z);
                affectedChunks.merge(createTree(worldPos, Chunks, ChunksMutex));
            }
            else if (tryReadBlock({ x + sourceChunkOrigin.x, terrainHeight, z + sourceChunkOrigin.z }, Chunks, ChunksMutex) == Block::GRASS && tryReadBlock({ x + sourceChunkOrigin.x, terrainHeight + 1, z + sourceChunkOrigin.z }, Chunks, ChunksMutex) == Block::AIR)
            {
                float randomReal = randomFloatInRange(0, 1);
                glm::vec3 worldPos = glm::vec3(x + sourceChunkOrigin.x, terrainHeight + 1, z + sourceChunkOrigin.z);

                if (randomReal < 0.25f / 16.0f)
                {
                    affectedChunks.merge(tryReplaceBlock(worldPos, Block::DAFFODIL, Chunks, ChunksMutex));
                }
                else if (randomReal < 0.50f / 16.0f)
                {
                    affectedChunks.merge(tryReplaceBlock(worldPos, Block::ROSE, Chunks, ChunksMutex));
                }
                else if (randomReal < 0.75f / 16.0f)
                {
                     affectedChunks.merge(tryReplaceBlock(worldPos, Block::RED_MUSHROOM, Chunks, ChunksMutex));
                }
                else if (randomReal < 1.0f / 16.0f)
                {
                    affectedChunks.merge(tryReplaceBlock(worldPos, Block::MUSHROOM, Chunks, ChunksMutex));
                }
            }
        }
    }

    return affectedChunks;
}

uint8_t getSideFaceTexture(Block blockEnum)
{
    static constexpr uint8_t textures[] =
    {
       11, // GRASS = 1,
       25, // STONE = 2,
       7,  // DIRT = 3,
       19, // OAK = 4,
       4,  // COBBLE = 5,
       0,  // BEDROCK = 6,
       13, // GRAVEL = 7,
       23, // SAND = 8,
       16, // OAK_LEAVES = 9
       26, // WATER = 10
       22, // ROSE = 11,
       5,  // DAFFODIL = 12,
       27, // RED_MUSHROOM = 13,
       28, // MUSHROOM = 14,
       29  // SUGAR_CANE = 15
    };

    return textures[(size_t)blockEnum - 1];
}

uint8_t getTopFaceTexture(Block blockEnum)
{
    static constexpr uint8_t textures[] =
    {
       12, // GRASS = 1,
       25, // STONE = 2,
       7,  // DIRT = 3,
       20, // OAK = 4,
       4,  // COBBLE = 5,
       0,  // BEDROCK = 6,
       13, // GRAVEL = 7,
       23, // SAND = 8,
       16, // OAK_LEAVES = 9
       26, // WATER = 10
       22, // ROSE = 11,
       5,  // DAFFODIL = 12,
       27, // RED_MUSHROOM = 13,
       28, // MUSHROOM = 14,
       29  // SUGAR_CANE = 15
    };

    return textures[(size_t)blockEnum - 1];
}

uint8_t getBottomFaceTexture(Block blockEnum)
{
    static constexpr uint8_t textures[] =
    {
       7,  // GRASS = 1,
       25, // STONE = 2,
       7,  // DIRT = 3,
       20, // OAK = 4,
       4,  // COBBLE = 5,
       0,  // BEDROCK = 6,
       13, // GRAVEL = 7,
       23, // SAND = 8,
       16, // OAK_LEAVES = 9
       26, // WATER = 10
       22, // ROSE = 11,
       5,  // DAFFODIL = 12,
       27, // RED_MUSHROOM = 13,
       28, // MUSHROOM = 14,
       29  // SUGAR_CANE = 15
    };

    return textures[(size_t)blockEnum - 1];
}

// Used for HUD
glm::vec2 getSideFaceTextureAtlas(Block blockEnum)
{
    static constexpr glm::vec2 textures[] =
    {
        {3.0f, 0.0f},   // GRASS = 1,
        {1.0f, 0.0f},   // STONE = 2,
        {2.0f, 0.0f},   // DIRT = 3,
        {4.0f, 1.0f},   // OAK = 4,
        {0.0f, 1.0f},   // COBBLE = 5,
        {1.0f, 1.0f},   // BEDROCK = 6,
        {3.0f, 1.0f},   // GRAVEL = 7,
        {2.0f, 1.0f},   // SAND = 8,
        {4.0f, 3.0f},   // OAK_LEAVES = 9
        {14.0f, 12.0f}, // WATER = 10
        {12.0f, 0.0f},  // ROSE = 11,
        {13.0f, 0.0f},  // DAFFODIL = 12,
        {12.0f, 1.0f},  // RED_MUSHROOM = 13,
        {13.0f, 1.0f},  // MUSHROOM = 14,
        {9.0f, 4.0f}    // SUGAR_CANE = 15
    };

    return textures[(size_t)blockEnum - 1];
}

// Transparent blocks do not occlude the faces of their neighbours by default
bool isTransparent(Block blockEnum)
{
    static constexpr bool transparentBlocks[] =
    {
        true,  // AIR = 0
        false, // GRASS = 1,
        false, // STONE = 2,
        false, // DIRT = 3,
        false, // OAK = 4,
        false, // COBBLE = 5,
        false, // BEDROCK = 6,
        false, // GRAVEL = 7,
        false, // SAND = 8,
        true,  // OAK_LEAVES = 9
        true,  // WATER = 10
        true,  // ROSE = 11,
        true,  // DAFFODIL = 12,
        true,  // RED_MUSHROOM = 13,
        true,  // MUSHROOM = 14,
        true   // SUGAR_CANE = 15
    };

    return transparentBlocks[(size_t)blockEnum];
}

// When a "self occluding block" is placed next to itself, you cannot see the joining face, but when it is placed next to a different block, you can.
// This is only relevant for transparent blocks
bool isSelfOccludingBlock(Block blockEnum)
{
    static constexpr bool selfOccludingBlocks[] =
    {
        false, // AIR = 0
        false, // GRASS = 1,
        false, // STONE = 2,
        false, // DIRT = 3,
        false, // OAK = 4,
        false, // COBBLE = 5,
        false, // BEDROCK = 6,
        false, // GRAVEL = 7,
        false, // SAND = 8,
        false, // OAK_LEAVES = 9
        true,  // WATER = 10
        false, // ROSE = 11,
        false, // DAFFODIL = 12,
        false, // RED_MUSHROOM = 13,
        false, // MUSHROOM = 14,
        false  // SUGAR_CANE = 15
    };         

    return selfOccludingBlocks[(size_t)blockEnum];
}

bool isBillboard(Block blockEnum)
{
    static constexpr bool billboards[] =
    {
        false, // AIR = 0
        false, // GRASS = 1,
        false, // STONE = 2,
        false, // DIRT = 3,
        false, // OAK = 4,
        false, // COBBLE = 5,
        false, // BEDROCK = 6,
        false, // GRAVEL = 7,
        false, // SAND = 8,
        false, // OAK_LEAVES = 9
        false, // WATER = 10
        true,  // ROSE = 11,
        true,  // DAFFODIL = 12,
        true,  // RED_MUSHROOM = 13,
        true,  // MUSHROOM = 14,
        true   // SUGAR_CANE = 15
    };

    return billboards[(size_t)blockEnum];
}

bool isSolid(Block blockEnum)
{
    static constexpr bool solidBlocks[] =
    {
        false, // AIR = 0
        true,  // GRASS = 1,
        true,  // STONE = 2,
        true,  // DIRT = 3,
        true,  // OAK = 4,
        true,  // COBBLE = 5,
        true,  // BEDROCK = 6,
        true,  // GRAVEL = 7,
        true,  // SAND = 8,
        true,  // OAK_LEAVES = 9
        false, // WATER = 10
        false, // ROSE = 11,
        false, // DAFFODIL = 12,
        false, // RED_MUSHROOM = 13,
        false, // MUSHROOM = 14,
        false  // SUGAR_CANE = 15
    };

    return solidBlocks[(size_t)blockEnum];
}

GLuint genFramebufferTextureAttachment(GLenum internalFormat, GLenum format, GLenum type)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, WIDTH, HEIGHT, 0, format, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;
}

GLuint genFramebufferWithDepthAndColor(GLuint colorTexture, GLuint depthTexture)
{
    GLuint FBO;
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "genFramebufferWithDepthAndColor not GL_FRAMEBUFFER_COMPLETE" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return FBO;
}

GLuint genTexturedQuadVAO()
{
    GLfloat quadVertices[] =
    {
        // pos         // uv
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    GLuint quadIndices[] =
    {
        2, 1, 0,
        3, 2, 0
    };

    GLuint screenQuadVAO, screenQuadVBO, screenQuadEBO;
    glGenVertexArrays(1, &screenQuadVAO);
    glGenBuffers(1, &screenQuadVBO);
    glGenBuffers(1, &screenQuadEBO);
    glBindVertexArray(screenQuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, screenQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, screenQuadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    return screenQuadVAO;
}

void resizeFBOTexture(GLuint texture, GLenum internalFormat, GLenum format, GLenum type)
{
    viewportSizeChanged = false;

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, WIDTH, HEIGHT, 0, format, type, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}