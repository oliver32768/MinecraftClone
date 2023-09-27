#version 460 core

in float brightness;
in vec3 viewPos;
in vec3 fragTexCoord;

uniform sampler2DArray texture1;

out vec4 FragColor;

void main()
{
    vec4 tex_color = texture(texture1, fragTexCoord);
    
    if (tex_color.a < 0.5)
    {
        discard;
    }
    
    FragColor = vec4(tex_color.xyz * brightness, tex_color.a);// * (1.0 - smoothstep(224.0f, 240.0f, length(viewPos))));
}