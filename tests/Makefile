CC := gcc
CFLAGS := -g -Wall -Wextra -Wno-unused-parameter -I./framework -I./mocks -I../include -DMOCK_BUILD
LDFLAGS := -lpthread -lm

TEST_SRC := $(wildcard unit/core/*.c unit/daemon/*.c unit/backend/*.c edge/*.c stress/*.c)
# Skip integration tests that require real implementations

FRAMEWORK_SRC := framework/test_utils.c framework/mock_registry.c
MOCK_SRC := mocks/llama_mock.c mocks/gguf_mock.c mocks/qmap_mock.c mocks/ndc_mock.c mocks/vulkan_mock.c
LIB_SRC := libqllm.c
MAIN_SRC := test_main.c

ALL_SRC := $(MAIN_SRC) $(FRAMEWORK_SRC) $(MOCK_SRC) $(LIB_SRC) $(TEST_SRC)
ALL_OBJ := $(ALL_SRC:.c=.o)

TARGET := test_runner

.PHONY: all clean run valgrind

all: $(TARGET)

$(TARGET): $(ALL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

libqllm.o: ../src/libqllm.c
	$(CC) $(CFLAGS) -I./mocks -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TARGET)

clean:
	rm -f $(TARGET) $(ALL_OBJ)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all      - Build the test runner"
	@echo "  run      - Build and run the tests"
	@echo "  valgrind - Run tests with valgrind memory checking"
	@echo "  clean    - Remove built files"
