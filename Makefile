CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =

# We will place .o files in the `build` directory (create it if it doesn't exist)
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))
REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@

# PHONY means these targets will always be executed
.PHONY: all infer_gpt2 clean

obj-m += kllm.o

kllm:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

# Add targets
TARGETS = infer_gpt2

all: $(TARGETS)

infer_gpt2: infer_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

clean:
	$(REMOVE_FILES) $(TARGETS)
	$(REMOVE_BUILD_OBJECT_FILES)
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
