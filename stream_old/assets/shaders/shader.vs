#version 330 core
	
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;

out vec2 TexCoord;

// uniform mat4 model;
	
void main()
{
	// gl_Position = model * vec4(position, 1.f);
	gl_Position = vec4(position, 1.f);
	TexCoord = vec2(texCoord.x, 1.f - texCoord.y); // on inverse l'axe des y pour avoir notre texture ds le bon sens !
}