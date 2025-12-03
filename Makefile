all := libqllm

libqllm-obj-y-Linux := src/vulkan.o
libqllm-obj-y-Darwin := src/metal.o

llamacpp := submodules/llama.cpp/build

uname := $(shell uname)
uname != uname

arch := $(shell uname -m)
arch != uname -m

SDK_VERSION := 1.4.328.1
SDK_URL := https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkansdk-linux-${arch}-${SDK_VERSION}.tar.xz
vulkan := third_party/${SDK_VERSION}/${arch}

PREFIX ?= /usr
CFLAGS := -I${llamacpp}/../include -I${llamacpp}/../ggml/include -I${vulkan}/include

GGML_STATIC-Linux := ggml-vulkan/libggml-vulkan.a
GGML_STATIC-Darwin := ggml-metal/libggml-metal.a
GGML_BE := ${GGML_BE-${uname}}
GGML := libggml.a libggml-cpu.a libggml-base.a \
	${GGML_STATIC-${uname}}

STATIC := src/libllama.a ${GGML:%=ggml/src/%}
STATIC := ${STATIC:%=${llamacpp}/%}

LDLIBS := -Wl,--whole-archive $(STATIC) -Wl,--no-whole-archive \
          -ldl -lpthread -lm -lstdc++ -lgomp -lvulkan

CMAKE_FLAGS-Linux := -DGGML_VULKAN=ON
CMAKE_FLAGS-Darwin := -DGGML_METAL=ON

third_party-Linux := ${vulkan}/include/shaderc/shaderc.h

include ./../mk/include.mk

src/libqllm.o: $(llamacpp)/src/libllama.a

$(llamacpp)/src/libllama.a: ${llamacpp}/Makefile
	make -C ${llamacpp} -j4

$(llamacpp)/Makefile: ${third_party-${uname}}
	echo "[INFO] Using Vulkan SDK in $(vulkan)"; \
	mkdir -p ${llamacpp} 2>/dev/null || true
	export VULKAN_SDK="$(abspath $(vulkan))" && \
		export PATH="$$VULKAN_SDK/bin:$$PATH" && \
		export LD_LIBRARY_PATH="$$VULKAN_SDK/lib:$$LD_LIBRARY_PATH" && \
		cd ${llamacpp} && \
		cmake .. ${CMAKE_FLAGS-${uname}} \
			-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX}/share/qllm \
			-DLLAMA_CURL=OFF \
			-DBUILD_SHARED_LIBS=OFF \
			-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
			-DVulkan_INCLUDE_DIR=./../../../${vulkan}/include

$(vulkan)/include/shaderc/shaderc.h:
	mkdir third_party || true
	wget -qO third_party/vulkan-sdk.tar.xz "$(SDK_URL)"
	tar -xf third_party/vulkan-sdk.tar.xz -C third_party
