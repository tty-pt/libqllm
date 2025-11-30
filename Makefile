all := libqllm

llamacpp := submodules/llama.cpp/build

uname != uname
uname := $(shell uname)

arch != uname -m
arch ?= $(shell uname -m)

SDK_VERSION := 1.4.328.1
SDK_URL := https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkansdk-linux-${arch}-${SDK_VERSION}.tar.xz
vulkan-sdk := third_party/${SDK_VERSION}/${arch}

# prefix-Darwin-arm64  := /opt/homebrew
# prefix-Darwin-x86_64 := /usr/local
# prefix-Darwin := ${prefix-Darwin-${arch}}
# omp := ${prefix-Darwin}/Cellar/libomp/
# omp-version := $(shell ls ${omp} | head -n 1)
# omp := ${omp}/${omp-version}

# ggml := libllama libggml libggml-vulkan
# ggml := ${ggml:%=ggml/src/%}
# static := ${ggml} common/libcommon
# static := ${static:%=${llamacpp}/%.a}

LDLIBS := -lstdc++ -ldl -lpthread -lm -lllama -lggml -lggml-cpu
LDLIBS-Linux := -lggml-vulkan
LDLIBS-Darwin := -lggml-metal
CFLAGS := -Iusr/local/include
LDFLAGS := -Lusr/local/lib -Lusr/local/lib64 -L${omp}/lib

CMAKE_FLAGS-Linux := -DGGML_VULKAN=ON
CMAKE_FLAGS-Darwin := -DGGML_METAL=ON

third_party-Linux := ${vulkan-sdk}/include/shaderc/shaderc.h

include ./../mk/include.mk

src/libqllm.o: ${llamacpp}/ggml/src/libggml.a

$(llamacpp)/ggml/src/libggml.a: ${llamacpp}
	cd ${llamacpp} && \
		make -j4 && make install DESTDIR=../../..

$(llamacpp): ${third_party-${uname}}
	echo "[INFO] Using Vulkan SDK in $(vulkan-sdk)"; \
	mkdir -p $@ 2>/dev/null || true
	export VULKAN_SDK="$(abspath $(vulkan-sdk))" && \
		export PATH="$$VULKAN_SDK/bin:$$PATH" && \
		export LD_LIBRARY_PATH="$$VULKAN_SDK/lib:$$LD_LIBRARY_PATH" && \
		cd ${llamacpp} && \
		cmake .. ${CMAKE_FLAGS-${uname}} -DLLAMA_CURL=OFF \
			-DVulkan_INCLUDE_DIR=./../../../${vulkan-sdk}/include

$(vulkan-sdk)/include/shaderc/shaderc.h:
	mkdir third_party || true
	wget -qO third_party/vulkan-sdk.tar.xz "$(SDK_URL)"
	tar -xf third_party/vulkan-sdk.tar.xz -C third_party
