# MinecraftClone
'Pure' C++/OpenGL Minecraft clone, i.e. made without the use of a game engine. Uses multithreading to compute terrain data and chunk geometry on multiple hardware threads, concurrently with the rendering thread. Chunks are computed in multiple stages (generating, meshing, decorating, buffering) in order to keep threads from blocking on contended resources. All resource access synchronization is done by hand using the STL due to the absence of any thread safety in OpenGL.

![Demo](videos/minecraft-clone-demo-compressed.mp4)

## Usage

Open the solution file in Visual Studio (tested on Visual Studio 2022) and change to Release build

Download [GLFW3](https://www.glfw.org/download), [Glad](https://glad.dav1d.de/) (GL 4.6, Profile Core), [GLM](https://github.com/g-truc/glm) and [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)

After extracting any downloaded zip files, add the path to the following directories under Project Properties -> C/C++ -> General -> Additional Include Directories: 
* glad/ (from inside folder 'include/' in glad.zip)
* glm/
* glfw-x.bin.WIN64/include/
* The directory you saved stb_image.h to

Add the following to Project Properties -> Linker -> General -> Additional Library Directories:
* glfw-x.bin.WIN64/lib-vc2022/

Add the following to Project Properties -> Linker -> Input -> Additional Dependencies:
* glfw3.lib
* glfw3dll.lib
* opengl32.lib

Build the binary: Build -> Build Solution. Copy MinecraftClone.exe from MinecraftClone/x64/Release to MinecraftClone/MinecraftClone/ and run the binary.