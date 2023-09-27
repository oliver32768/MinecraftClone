#version 460 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;

uniform mat4 model;
uniform mat4 projection;

uniform vec2 atlasCutout;
uniform vec2 atlasOrigin; // bottom left of quad

out vec2 uv;

void main()
{
    gl_Position = projection * model * vec4(aPos, 0.0, 1.0);

    uv = aUV * atlasCutout;
    uv += atlasOrigin;
}