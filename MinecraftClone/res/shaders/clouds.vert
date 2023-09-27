#version 460 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float uvOffset;
uniform float uvMult;

out vec2 uv;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 0.0, 1.0);

    uv = vec2(aUV.x + uvOffset, aUV.y) * uvMult;
}