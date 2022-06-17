# CMake

## 简介

用不同语言或编译器开发同一个项目，最终输出一个可执行文件或者共享库

## Hello World

`main.cpp`

```cpp
#include <iostream>

int main()
{
	std::cout << "Hello World" << std::endl;
}
```

新建`CMakeLists.txt`（严格区分大小写）

```cmake
project(HELLO)
set(SRC_LIST main.cpp)
message(STATUS "This is BINARY dir" ${HELLO_BINARY_DIR})
massage(STATUS "This is SOURCE dir" ${HELLO_SOURCE_DIR})
add_executable(hello ${SRC_LIST})
```

在命令窗输入`cmake .`生成`makefile`文件（建议新建`build`文件夹并进行`cmake ..`）

```shell
<project filepath>> cmake .
#Windows安装了VS或MSCV默认生成.sln等一系列适用VS或MSCV的文件
#使用MinGW编译，以方便跨平台使用
#每次使用命令窗第一次输入cmake -G"MinGW Makefiles" .
-- The C compiler identification is GNU 11.2.0
-- The CXX compiler identification is GNU 11.2.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: <mingw64 install filespath>/bin/gcc.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: <mingw64 install filespath>/bin/g++.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- This is BINARY dir<project filepath>/
-- This is SOURCE dir<project filepath>
-- Configuring done
-- Generating done
-- Build files have been written to: <project filepath>
```

目录生成了`-CMakeFiles`，`CMakeCache.txt`，`cmake_install.cmake`以及`Makefiles`

输入`make`进行编译

```shell
<project filepath>> make
[ 50%] Building CXX object CMakeFiles/hello.dir/main.cpp.obj
[100%] Linking CXX executable hello.exe
[100%] Built target hello
```

最终生成`hello.exe`可执行文件

## 关键字

### project

指定工程的名字和语言，默认支持所有语言

- **`project(<projectname>)`** 指定工程的名字，支持所有语言

- **`project(<projectname> C)`** 支持C

- **`project(<projectname> C CXX)`** 支持C和C++

- **`project(<projectname> JAVA)`** 支持Java

也隐式定义了量CMake变量

- `<projectname>_BINARY_DIR`是编译路径`<project filepath>`

- `<projectname>_SOURCE_DIR`是工程路径

如果改了工程名，变量名也会改变



### set

用来显示指定变量

**`set(SRC_LIST main.cpp)`** `SRC_LIST`变量就包含了`main.cpp`

也可以包含多个，用空格隔开



### message

向终端输出信息

- SEND_ERROR 产生错误，跳过生成过程
- STATUS 输出前缀为一的信息
- FATAL_ERROR 立刻终止所有CMake进程



### **add_executable**

生成可执行文件

**`add_executable(hello ${SRC_LIST})`** 生成名为`hello`的可执行文件，源文件读取变量`SRC_LIST`中的内容

也可以直接写成`add_executable(hello main.cpp)`

工程名的`HELLO`和生成的可执行文件`hello`无关



以上过程可简化为

```cmake
PROJECT (HELLO)
ADD_EXECUTABLE(hello main.cpp)
```

## 语法原则

- 变量使用`${}`取值，在`IF`控制语句中直接使用

- 基本格式：`指令(参数1 参数2)`

  参数之间使用空格或分号隔开

  `add_executable(hello main.cpp func.cpp)`或`add_executable(hello main.cpp;func.cpp)`

- 指令无关大小写，参数和变量与大小写相关

- `set(SRC_LIST main.cpp)`可以写成`set(SRC_LIST "main.cpp")`，如果文件名存在空格需要使用双引号

## 内部构件和外部构建

上述为外部构建，临时文件较多

**外部构建就是新建一个`build`文件夹，然后在文件夹运行`cmake ..`**

`<projectname>_BINARY_DIR`是编译路径`<project filepath>/build`，`<projectname>_SOURCE_DIR`不变

hello.exe会在`<project filepath>/build`中

## 搭建工程

- 添加一个子目录`src`，用来存放源码
- 添加一个子目录`doc`，存放文档
- 添加文本文件`COPYRIGHT`，`README.md`
- 添加一个`run.sh`的脚本，用来调用二进制
- 将构建后的目标文件放入构建目录`bin`
- 将doc目录的内容以及`COPYRIGIHT`/`README.md`安装到`usr/share/doc/cmake`

### **将构建后的目标文件放入构建目录`bin`**

每个目录下都要有一个`CMakeLists.txt`

```shell
> tree

.
├─build
├─CMakeFiles.txt
└─src
  ├─CMakeFiles.txt
  └─main.cpp
```

外部`CMakeLists.txt`

```cmake
project (HELLO)
add_subdirectory(src bin)
```

`src`内的`CMakeLists.txt`

```cmake
add_executable(hello main.cpp)
```

### add_subdirectory

**`add_subdirectory(source_dir binary_dir EXCLUDE_FROM_ALL)`**

- 用于向当前工程添加存放源文件的子目录，并可以指定中间二进制和目标二进制的存放位置
- `EXCLUDE_FROM_ALL`是将写入的目录从程序中排除

### **更改二进制的保存路径**

用`set`重新定义`EXECUTABLE_OUTPUT_PATH`和`LIBRARY_OUTPUT_PATH`

写入`src`下的`CMakeLists.txt`

- **`set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)`**

- **`set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)`**

## 静态库和动态库

新建`hello.h`和`hello.cpp`

外部`CMakeLists.txt`

```cmake
project(HELLO)
add_subdirectory(src lib)
```

`src`内的`CMakeLists.txt`

```cmake
set(LIBHELLO_SRC hello.cpp)
add_library(hello SHARED ${LIBHELLO_SRC})
```

### **add_library**

**`add_library(hello SHARED ${LIBHELLO_SRC})`**

- `hello` 库名，生成动态 `libhello.dll(.a)`  静态库 `libhello.a/.lib`
- `SHARED` 动态库  `STATIC` 静态库

### **静态库和动态库**

- 静态库在编译时会直接链接到目标程序中，编译成功的可执行文件可独立运行
- 动态库在编译时不会链接到目标程序中，即可执行文件无法单独运行

### **同时构建静态库和动态库**

```cmake
//通过不同名称来生成不同的库，如果名称相同则只会构建前一个
add_library(hello SHARED ${LIBHELLO_SRC})
add_library(hellostatic STATIC ${LIBHELLO_SRC})
```

### **set_target_properties**

用来设置输出名称，动态库可以用来指定动态库的版本和API版本

```cmake
set(LIBHELLO_SRC hello.cpp)

add_library(hello_static STATIC ${LIBHELLO_SRC})

//对hello_static的重名为hello
set_target_properties(hello_static PROPERTIES  OUTPUT_NAME "hello")
//cmake 在构建一个新的target 时，会尝试清理掉其他使用这个名字的库，因为，在构建 libhello.so 时， 就会清理掉 libhello.a
set_target_properties(hello_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

add_library(hello SHARED ${LIBHELLO_SRC})

set_target_properties(hello PROPERTIES  OUTPUT_NAME "hello")
set_target_properties(hello PROPERTIES CLEAN_DIRECT_OUTPUT 1)
```

**构建**

```shell
> make
[ 25%] Building CXX object lib/CMakeFiles/hello_static.dir/hello.obj
[ 50%] Linking CXX static library libhello.a
[ 50%] Built target hello_static
[ 75%] Building CXX object lib/CMakeFiles/hello.dir/hello.obj
[100%] Linking CXX shared library libhello.dll
[100%] Built target hello
```

修改版本号

**`set_target_properties(hello PROPERTIES VERSION 1.2 SOVERSION 1)`**

`VERSION` 指代动态库版本，`SOVERSION` 指代 API 版本

### **安装**

将`hello.h`安装到`<filepath>/include/hello`目录

```cmake
//文件放到该目录下
install(FILES hello.h DESTINATION include/hello)

//二进制，静态库，动态库安装都用TARGETS
//ARCHIVE 特指静态库，LIBRARY 特指动态库，RUNTIME 特指可执行目标二进制。
install(TARGETS hello hello_static LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
```

安装的时候，指定一下路径 `cmake -DCMAKE_INSTALL_PREFIX=<filepath>`

### 头文件搜索路径 include_directories

这条指令可以用来向工程添加多个特定的头文件搜索路径，路径之间用空格分割

### **链接静态库 target_link_libraries**

`target_link_libraries(main libhello.a)`

# CMake 一些模板

## 对于OpenCV项目的编译

```cmake
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)	#设置编译的C++标准

#set the path installed OpenCV to ./opencv/build
#which includes folders /bin /include
set(OpenCV_DIR "G:/opecv_mingw/install")	#使用mingw重新编译之后的opencv库
set(OPENCV_FOUND TRUE)
find_package(OpenCV REQUIRED)

#print the imformation about current OpenCV
message(STATUS "OpenCV library status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "> libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS} ")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(SLAM "main.cpp")

target_link_libraries(SLAM ${OpenCV_LIBS})
```

Windows还需要把`...\install\x64\mingw\bin`添加到环境变量PATH中才能运行

## 对Eigen项目的编译

```cmake
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
#....
add_executable(xxx "main.cpp")
target_link_libraries(xxx Eigen3::Eigen)
```

