CC ?= clang
CFLAGS = -O3 -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =

CC_FLAGS_FPU := -mhard-float
CFLAGS_kllm_gpt2.o += $(CC_FLAGS_FPU)
CFLAGS_REMOVE_kllm_gpt2.o += $(CC_FLAGS_NO_FPU)

# We will place .o files in the `build` directory (create it if it doesn't exist)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@

# PHONY means these targets will always be executed
.PHONY: all clean

obj-m += kllm.o
kllm-objs += kllm_main.o kllm_gpt2.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	$(REMOVE_FILES) $(TARGETS)
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
