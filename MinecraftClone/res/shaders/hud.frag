#version 460 core

in vec2 uv;

uniform sampler2D texture1;

out vec4 FragColor;

void main()
{
    vec4 tex_color = texture(texture1, uv);
    
    if (tex_color.a == 0.0)
    {
        discard;
    }
    
    FragColor = vec4(tex_color.xyz, tex_color.a);
}