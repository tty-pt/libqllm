all := libaxil-qllm engine-test

SITE ?= ${HOME}/site

SITE_INC := -I${SITE}/external/axil/include \
	-I${SITE}/external/libxylem/include \
	-I${SITE}/external/libqmap/include

SITE_LIB := -L${SITE}/external/axil/lib \
	-L${SITE}/external/libxylem/lib \
	-L${SITE}/external/libqmap/lib \
	-Wl,-rpath,${SITE}/external/axil/lib \
	-Wl,-rpath,${SITE}/external/libxylem/lib \
	-Wl,-rpath,${SITE}/external/libqmap/lib

EXTRA_CFLAGS += ${SITE_INC}

INSTALL_BIN := qllm-chat qllm-path qllm-list

engine-test-obj-y :=
LDLIBS-engine-test := -laxil-qllm -ljson-c -lpthread -lm
LDFLAGS-engine-test = -Wl,-rpath,${pwd}/lib

libaxil-qllm-obj-y := src/libqllm.o src/qllm-engine.o src/openai_embed.o src/openai_chat.o
libaxil-qllm-obj-y-Linux := src/vulkan.o
libaxil-qllm-obj-y-Darwin := src/metal.o

LDLIBS-libaxil-qllm := -laxil -lxylem -lqmap -lqsys -ljson-c \
	-lllama -lggml -lggml-cpu -lggml-base -ldl -lpthread -lm -lstdc++
LDLIBS-libaxil-qllm-Linux := -lgomp -lvulkan -lggml-vulkan
LDLIBS-libaxil-qllm-Darwin := -lggml-metal -lggml-blas -lomp \
	-framework Foundation \
	-framework CoreFoundation \
	-framework IOKit \
	-framework Metal \
	-framework Accelerate

llamacpp := submodules/llama.cpp/build

uname := $(shell uname)
uname != uname

arch := $(shell uname -m)
arch != uname -m

SDK_VERSION := 1.4.328.1
SDK_URL := https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkansdk-linux-${arch}-${SDK_VERSION}.tar.xz
vulkan := third_party/${SDK_VERSION}/${arch}

PREFIX ?= /usr

prefix-Darwin-arm64  := /opt/homebrew
prefix-Darwin-x86_64 := /usr/local
prefix-Darwin := ${prefix-Darwin-${arch}}
omp := ${prefix-Darwin}/Cellar/libomp/
omp-version := $(shell ls ${omp} | head -n 1)
omp := ${omp}/${omp-version}

CFLAGS := -g -I${llamacpp}/../include -I${llamacpp}/../ggml/include -I${vulkan}/include
CFLAGS-Darwin := -I${omp}/include

ggmlp := ${llamacpp}/ggml/src

LDFLAGS-libaxil-qllm := -L${llamacpp}/src -L${ggmlp} ${SITE_LIB}
LDFLAGS-Linux := -L${ggmlp}/ggml-vulkan -L${vulkan}/lib
LDFLAGS-Darwin := -L${ggmlp}/ggml-metal -L${ggmlp}/ggml-blas -L${omp}/lib

CMAKE_FLAGS-Linux := -DGGML_VULKAN=ON
CMAKE_FLAGS-Darwin := -DGGML_METAL=ON

third_party-Linux := ${vulkan}/include/shaderc/shaderc.h

include ./../mk/include.mk

src/libqllm.o: $(llamacpp)/src/libllama.a

$(llamacpp)/src/libllama.a: ${llamacpp}/Makefile
	make -C ${llamacpp} -j1; \
	scripts/vkgen-repair.sh ${llamacpp} ${vulkan}/bin/glslc; \
	make -C ${llamacpp} -j1

$(llamacpp)/Makefile: ${third_party-${uname}}
	echo "[INFO] Using Vulkan SDK in $(vulkan)"; \
	mkdir -p ${llamacpp} 2>/dev/null || true
	export VULKAN_SDK="$(abspath $(vulkan))" && \
		export PATH="$$VULKAN_SDK/bin:$$PATH" && \
		export LD_LIBRARY_PATH="$$VULKAN_SDK/lib:$$LD_LIBRARY_PATH" && \
		export CC=clang && \
		export CXX=clang++ && \
		cd ${llamacpp} && \
		cmake .. ${CMAKE_FLAGS-${uname}} \
			-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX}/share/qllm \
			-DLLAMA_CURL=OFF \
			-DLLAMA_BUILD_EXAMPLES=OFF \
			-DLLAMA_BUILD_TESTS=OFF \
			-DLLAMA_BUILD_SERVER=OFF \
			-DLLAMA_BUILD_TOOLS=OFF \
			-DBUILD_SHARED_LIBS=OFF \
			-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
			-DVulkan_INCLUDE_DIR=./../../../${vulkan}/include

$(vulkan)/include/shaderc/shaderc.h:
	mkdir third_party || true
	wget -qO third_party/vulkan-sdk.tar.xz "$(SDK_URL)"
	tar -xf third_party/vulkan-sdk.tar.xz -C third_party