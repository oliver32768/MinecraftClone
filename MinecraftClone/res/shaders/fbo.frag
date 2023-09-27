#version 460 core

uniform sampler2D fbo1ColorTexture;
uniform sampler2D fbo1DepthTexture;

uniform sampler2D fbo2ColorTexture;
uniform sampler2D fbo2DepthTexture;

out vec4 fragColor;

in vec2 fragTexCoord;

void main()
{
    float depth1 = texture(fbo1DepthTexture, fragTexCoord).r;
    float depth2 = texture(fbo2DepthTexture, fragTexCoord).r;

    int waterOntop = int(depth2 < depth1);

    vec4 color1 = texture(fbo1ColorTexture, fragTexCoord);
    vec4 color2 = texture(fbo2ColorTexture, fragTexCoord);

    fragColor = mix(color1, color2, waterOntop * 0.6);
}
