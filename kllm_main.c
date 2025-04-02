// kllm.c - Kernel LLM Device Driver
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/firmware.h>
#include <linux/version.h>
#include <linux/mutex.h>
#include <asm/fpu/api.h>
#include "kllm_gpt2.h"
#include <linux/workqueue.h>
#include <linux/completion.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/cpumask.h>

#define DEVICE_NAME "llm0"
#define CLASS_NAME "kllm"
#define FW_MODEL "kllm-gpt2.bin"
#define FW_TOKENIZER "kllm-tokenizer.bin"
#define MAX_INPUT_SIZE 4096
#define MAX_OUTPUT_SIZE 16

MODULE_LICENSE("GPL");
MODULE_AUTHOR("David Li (randombk)");
MODULE_DESCRIPTION("LLM Inference in the Kernel");
MODULE_VERSION("0.1");
MODULE_SOFTDEP("pre: kernel-fpu");

static int major_number;
static struct class *kllm_class = NULL;
static struct device *kllm_device = NULL;
static struct cdev kllm_cdev;

// Firmware data
static const struct firmware *fw_model = NULL;
static const struct firmware *fw_tokenizer = NULL;

// GPT-2 model instance
static struct gpt2_model gpt2;

// Device buffers
static char *input_buffer = NULL;
static char *output_buffer = NULL;
static char *context_buffer = NULL;  // Buffer to maintain generation context
static size_t output_pos = 0;
static size_t total_output_len = 0;
static bool generation_complete = false;

// Device mutex
static struct mutex kllm_mutex;

// Kernel thread for generation
static struct task_struct *kllm_thread = NULL;
static bool thread_running = false;

// Generation thread function
static int kllm_generation_thread(void *data)
{
    int ret;
    char token_buffer[MAX_OUTPUT_SIZE];
    size_t output_pos = 0;
    
    // Restrict thread to CPU 1
    cpumask_t cpu_mask;
    cpumask_clear(&cpu_mask);
    cpumask_set_cpu(1, &cpu_mask);
    set_cpus_allowed_ptr(current, &cpu_mask);
    
    // Initialize context with input
    mutex_lock(&kllm_mutex);
    strcpy(context_buffer, input_buffer);
    mutex_unlock(&kllm_mutex);
    
    // Generate tokens one by one
    while (output_pos < MAX_OUTPUT_SIZE - 1) {
        // Generate next token
        ret = gpt2_generate_next_token(&gpt2, context_buffer, token_buffer);
        
        if (ret < 0) {
            mutex_lock(&kllm_mutex);
            snprintf(output_buffer, MAX_OUTPUT_SIZE, "Error generating response: %d", ret);
            total_output_len = strlen(output_buffer);
            mutex_unlock(&kllm_mutex);
            break;
        }
        
        if (ret == 0) {
            // End of generation
            break;
        }
        
        // Check if we have space for the new token
        if (output_pos + ret >= MAX_OUTPUT_SIZE - 1) {
            break;
        }
        
        // Update output buffer and context with mutex protection
        mutex_lock(&kllm_mutex);
        strcpy(output_buffer + output_pos, token_buffer);
        strcat(context_buffer, token_buffer);  // Append new token to context
        output_pos += ret;
        total_output_len = output_pos;
        mutex_unlock(&kllm_mutex);

        // Print the output thus far
        printk(KERN_INFO "KLLM: Generated response: %s\n", output_buffer);
        
        // Allow other tasks to run between tokens
        cond_resched();
    }
    
    // Ensure output is null terminated
    mutex_lock(&kllm_mutex);
    output_buffer[output_pos] = '\0';
    total_output_len = output_pos;
    generation_complete = true;
    mutex_unlock(&kllm_mutex);
    
    printk(KERN_INFO "KLLM: Generated response of %zu bytes\n", total_output_len);
    
    thread_running = false;
    return 0;
}

// Device open function
static int kllm_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "KLLM: Device opened\n");
    return 0;
}

// Device release function
static int kllm_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "KLLM: Device closed\n");
    return 0;
}

// Device write function
static ssize_t kllm_write(struct file *file, const char __user *buffer,
                         size_t len, loff_t *offset)
{
    size_t bytes_to_copy = min((size_t)MAX_INPUT_SIZE, len);
    
    // Try to acquire the mutex
    if (!mutex_trylock(&kllm_mutex)) {
        printk(KERN_ERR "KLLM: Device is busy processing another request\n");
        return -EBUSY;
    }
    
    // Reset state
    output_pos = 0;
    total_output_len = 0;
    generation_complete = false;
    memset(output_buffer, 0, MAX_OUTPUT_SIZE);
    memset(input_buffer, 0, MAX_INPUT_SIZE);
    memset(context_buffer, 0, MAX_INPUT_SIZE);
    
    // Copy from user space to kernel space
    if (copy_from_user(input_buffer, buffer, bytes_to_copy)) {
        mutex_unlock(&kllm_mutex);
        printk(KERN_ERR "KLLM: Failed to copy from user space\n");
        return -EFAULT;
    }
    
    // Start generation thread if not already running
    if (!thread_running) {
        thread_running = true;
        kllm_thread = kthread_run(kllm_generation_thread, NULL, "kllm_gen");
        if (IS_ERR(kllm_thread)) {
            thread_running = false;
            mutex_unlock(&kllm_mutex);
            printk(KERN_ERR "KLLM: Failed to start generation thread\n");
            return -ENOMEM;
        }
    }
    
    printk(KERN_INFO "KLLM: Received %zu bytes from user\n", bytes_to_copy);
    mutex_unlock(&kllm_mutex);
    return bytes_to_copy;
}

// Device read function
static ssize_t kllm_read(struct file *file, char __user *buffer,
                        size_t len, loff_t *offset)
{
    size_t bytes_to_copy;
    
    // Try to acquire the mutex
    if (!mutex_trylock(&kllm_mutex)) {
        printk(KERN_ERR "KLLM: Device is busy processing another request\n");
        return -EBUSY;
    }
    
    // Check if we've already sent the entire response
    if (*offset >= total_output_len) {
        mutex_unlock(&kllm_mutex);
        return 0;
    }
    
    // Calculate how many bytes to copy
    bytes_to_copy = min(len, total_output_len - *offset);
    
    // Copy from kernel space to user space
    if (copy_to_user(buffer, output_buffer + *offset, bytes_to_copy)) {
        mutex_unlock(&kllm_mutex);
        printk(KERN_ERR "KLLM: Failed to copy to user space\n");
        return -EFAULT;
    }
    
    // Update the offset for subsequent reads
    *offset += bytes_to_copy;
    
    // If we've sent the entire response, release the mutex
    mutex_unlock(&kllm_mutex);
    
    return bytes_to_copy;
}

// File operations structure
static struct file_operations kllm_fops = {
    .owner = THIS_MODULE,
    .open = kllm_open,
    .release = kllm_release,
    .read = kllm_read,
    .write = kllm_write,
};

// Module initialization function
static int __init kllm_init(void)
{
    int ret;
    
    // Initialize the mutex
    mutex_init(&kllm_mutex);
    
    // Allocate device buffers
    input_buffer = kmalloc(MAX_INPUT_SIZE, GFP_KERNEL);
    output_buffer = kmalloc(MAX_OUTPUT_SIZE, GFP_KERNEL);
    context_buffer = kmalloc(MAX_INPUT_SIZE, GFP_KERNEL);
    if (!input_buffer || !output_buffer || !context_buffer) {
        printk(KERN_ERR "KLLM: Failed to allocate memory\n");
        ret = -ENOMEM;
        goto fail_alloc;
    }
    
    // Register major number
    major_number = register_chrdev(0, DEVICE_NAME, &kllm_fops);
    if (major_number < 0) {
        printk(KERN_ERR "KLLM: Failed to register major number\n");
        ret = major_number;
        goto fail_chrdev;
    }
    
    // Register device class
    #if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
        kllm_class = class_create(CLASS_NAME);
    #else
        kllm_class = class_create(THIS_MODULE, CLASS_NAME);
    #endif
    if (IS_ERR(kllm_class)) {
        printk(KERN_ERR "KLLM: Failed to create device class\n");
        ret = PTR_ERR(kllm_class);
        goto fail_class;
    }
    
    // Create the device
    kllm_device = device_create(kllm_class, NULL, MKDEV(major_number, 0), NULL, DEVICE_NAME);
    if (IS_ERR(kllm_device)) {
        printk(KERN_ERR "KLLM: Failed to create device\n");
        ret = PTR_ERR(kllm_device);
        goto fail_device;
    }
    
    // Initialize character device
    cdev_init(&kllm_cdev, &kllm_fops);
    kllm_cdev.owner = THIS_MODULE;
    ret = cdev_add(&kllm_cdev, MKDEV(major_number, 0), 1);
    if (ret < 0) {
        printk(KERN_ERR "KLLM: Failed to add character device\n");
        goto fail_cdev;
    }
    
    // Load model firmware
    ret = request_firmware(&fw_model, FW_MODEL, kllm_device);
    if (ret) {
        printk(KERN_ERR "KLLM: Failed to load model firmware: %s\n", FW_MODEL);
        goto fail_fw_model;
    }
    
    // Load tokenizer firmware
    ret = request_firmware(&fw_tokenizer, FW_TOKENIZER, kllm_device);
    if (ret) {
        printk(KERN_ERR "KLLM: Failed to load tokenizer firmware: %s\n", FW_TOKENIZER);
        goto fail_fw_tokenizer;
    }
    
    // Initialize GPT-2 model
    kernel_fpu_begin();
    ret = gpt2_build_from_firmware(&gpt2, fw_model, fw_tokenizer);
    kernel_fpu_end();
    if (ret) {
        printk(KERN_ERR "KLLM: Failed to initialize GPT-2 model\n");
        goto fail_gpt2;
    }
    
    printk(KERN_INFO "KLLM: Module loaded successfully! Major number: %d\n", major_number);
    printk(KERN_INFO "KLLM: Device created at /dev/%s\n", DEVICE_NAME);
    printk(KERN_INFO "KLLM: Loaded model (%zu bytes) and tokenizer (%zu bytes)\n", 
           fw_model->size, fw_tokenizer->size);
    
    return 0;
    
fail_gpt2:
    release_firmware(fw_tokenizer);
fail_fw_tokenizer:
    release_firmware(fw_model);
fail_fw_model:
    cdev_del(&kllm_cdev);
fail_cdev:
    device_destroy(kllm_class, MKDEV(major_number, 0));
fail_device:
    class_destroy(kllm_class);
fail_class:
    unregister_chrdev(major_number, DEVICE_NAME);
fail_chrdev:
    kfree(input_buffer);
    kfree(output_buffer);
    kfree(context_buffer);
fail_alloc:
    return ret;
}

// Module cleanup function
static void __exit kllm_exit(void)
{
    // Stop generation thread if running
    if (thread_running && kllm_thread) {
        kthread_stop(kllm_thread);
        kllm_thread = NULL;
        thread_running = false;
    }
    
    // Free GPT-2 model
    gpt2_free(&gpt2);
    
    // Release firmware
    if (fw_tokenizer)
        release_firmware(fw_tokenizer);
    if (fw_model)
        release_firmware(fw_model);
    
    // Cleanup device
    cdev_del(&kllm_cdev);
    device_destroy(kllm_class, MKDEV(major_number, 0));
    class_destroy(kllm_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    
    // Free memory
    kfree(input_buffer);
    kfree(output_buffer);
    kfree(context_buffer);
    
    // Destroy the mutex
    mutex_destroy(&kllm_mutex);
    
    printk(KERN_INFO "KLLM: Module unloaded\n");
}

module_init(kllm_init);
module_exit(kllm_exit);