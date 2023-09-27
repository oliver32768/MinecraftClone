#version 460 core

layout (location = 0) in uint aPacked;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out float brightness;
out vec3 viewPos;
out vec3 fragTexCoord;

float brightnessLevels[4] = { 0.45f, 0.60f, 0.775f, 1.0f };

void main()
{
    int posX = int((aPacked & 0x1F));                
    int posY = int((aPacked >> 5) & 0x1F);           
    int posZ = int((aPacked >> 10) & 0x1F);          
    int brightnessLookup = int((aPacked >> 15) & 0x03);
    int uvX = int((aPacked >> 17) & 0x01);
    int uvY = int((aPacked >> 18) & 0x01);
    int uvZ = int((aPacked >> 19) & 0xFF);

    vec3 aPos = vec3(posX, posY, posZ);
    float aFakeNormal = brightnessLevels[brightnessLookup];
    vec3 aUV = vec3(uvX, uvY, uvZ);

    gl_Position = projection * view * model * vec4(aPos, 1.0);

    brightness = aFakeNormal;

    fragTexCoord = aUV;

    viewPos = (view * model * vec4(aPos, 1.0)).xyz;
}