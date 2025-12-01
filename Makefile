all := libqllm

llamacpp := submodules/llama.cpp/build

uname != uname
uname := $(shell uname)

arch != uname -m
arch ?= $(shell uname -m)

SDK_VERSION := 1.4.328.1
SDK_URL := https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkansdk-linux-${arch}-${SDK_VERSION}.tar.xz
vulkan-sdk := third_party/${SDK_VERSION}/${arch}

LDLIBS := -lstdc++ -ldl -lpthread -lm -lllama -lggml -lggml-cpu
LDLIBS-Linux := -lggml-vulkan
LDLIBS-Darwin := -lggml-metal
PREFIX ?= /usr
prefix-qllm := ${DESTDIR}${PREFIX}/share/qllm
CFLAGS := -I${prefix-qllm}/include
LDFLAGS := -L${prefix-qllm}/lib -L${prefix-qllm}/lib64
LDFLAGS += -Wl,-rpath,'$$ORIGIN'

CMAKE_FLAGS-Linux := -DGGML_VULKAN=ON
CMAKE_FLAGS-Darwin := -DGGML_METAL=ON

third_party-Linux := ${vulkan-sdk}/include/shaderc/shaderc.h

include ./../mk/include.mk

src/libqllm.o: ${DESTDIR}${PREFIX}/share/qllm/lib/libllama.so

$(DESTDIR)$(PREFIX)/share/qllm/lib/libllama.so: ${llamacpp}/bin/libllama.so
	make -C ${llamacpp} install DESTDIR=${DESTDIR} \
		PREFIX=${PREFIX}/share/qllm

$(llamacpp)/bin/libllama.so: ${llamacpp}/Makefile
	make -C ${llamacpp} -j4

$(llamacpp)/Makefile: ${third_party-${uname}}
	echo "[INFO] Using Vulkan SDK in $(vulkan-sdk)"; \
	mkdir -p ${llamacpp} 2>/dev/null || true
	export VULKAN_SDK="$(abspath $(vulkan-sdk))" && \
		export PATH="$$VULKAN_SDK/bin:$$PATH" && \
		export LD_LIBRARY_PATH="$$VULKAN_SDK/lib:$$LD_LIBRARY_PATH" && \
		cd ${llamacpp} && \
		cmake .. ${CMAKE_FLAGS-${uname}} \
			-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX}/share/qllm \
			-DLLAMA_CURL=OFF \
			-DVulkan_INCLUDE_DIR=./../../../${vulkan-sdk}/include

$(vulkan-sdk)/include/shaderc/shaderc.h:
	mkdir third_party || true
	wget -qO third_party/vulkan-sdk.tar.xz "$(SDK_URL)"
	tar -xf third_party/vulkan-sdk.tar.xz -C third_party
