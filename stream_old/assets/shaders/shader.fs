#version 330 core

in vec2 TexCoord;

out vec4 color;

uniform sampler2D tex;

void main()
{
	color = texture(tex, TexCoord);
	// color = vec4(1.f, 0.5f, 0.f, 1.f); // FULL ORANGE FOR DEBUG !!
}