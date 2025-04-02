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

#define DEVICE_NAME "llm0"
#define CLASS_NAME "kllm"
#define FW_MODEL "kllm-gpt2.bin"
#define FW_TOKENIZER "kllm-tokenizer.bin"
#define MAX_INPUT_SIZE 4096
#define MAX_OUTPUT_SIZE 4096

MODULE_LICENSE("GPL");
MODULE_AUTHOR("David Li (randombk)");
MODULE_DESCRIPTION("LLM Inference in the Kernel");
MODULE_VERSION("0.1");

static int major_number;
static struct class *kllm_class = NULL;
static struct device *kllm_device = NULL;
static struct cdev kllm_cdev;

// Firmware data
static const struct firmware *fw_model = NULL;
static const struct firmware *fw_tokenizer = NULL;

// Device buffers
static char *input_buffer = NULL;
static char *output_buffer = NULL;

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

// Process input and generate response
// This is a simplified placeholder - a real implementation would use the loaded models
static void process_input(const char *input, char *output, size_t max_len)
{
    // A very simple echo response for demonstration
    snprintf(output, max_len, "KLLM response to: %s", input);
}

// Device write function
static ssize_t kllm_write(struct file *file, const char __user *buffer,
                         size_t len, loff_t *offset)
{
    size_t bytes_to_copy = min((size_t)MAX_INPUT_SIZE, len);
    
    // Clear input buffer
    memset(input_buffer, 0, MAX_INPUT_SIZE);
    
    // Copy from user space to kernel space
    if (copy_from_user(input_buffer, buffer, bytes_to_copy)) {
        printk(KERN_ERR "KLLM: Failed to copy from user space\n");
        return -EFAULT;
    }
    
    // Process the input and generate a response
    memset(output_buffer, 0, MAX_OUTPUT_SIZE);
    process_input(input_buffer, output_buffer, MAX_OUTPUT_SIZE);
    
    printk(KERN_INFO "KLLM: Received %zu bytes from user\n", bytes_to_copy);
    return bytes_to_copy;
}

// Device read function
static ssize_t kllm_read(struct file *file, char __user *buffer,
                        size_t len, loff_t *offset)
{
    size_t response_len = strlen(output_buffer);
    size_t bytes_to_copy;
    
    // Check if we've already sent the entire response
    if (*offset >= response_len)
        return 0;
    
    // Calculate how many bytes to copy
    bytes_to_copy = min(len, response_len - *offset);
    
    // Copy from kernel space to user space
    if (copy_to_user(buffer, output_buffer + *offset, bytes_to_copy)) {
        printk(KERN_ERR "KLLM: Failed to copy to user space\n");
        return -EFAULT;
    }
    
    // Update the offset for subsequent reads
    *offset += bytes_to_copy;
    
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
    
    // Allocate device buffers
    input_buffer = kmalloc(MAX_INPUT_SIZE, GFP_KERNEL);
    output_buffer = kmalloc(MAX_OUTPUT_SIZE, GFP_KERNEL);
    if (!input_buffer || !output_buffer) {
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
    
    printk(KERN_INFO "KLLM: Module loaded successfully! Major number: %d\n", major_number);
    printk(KERN_INFO "KLLM: Device created at /dev/%s\n", DEVICE_NAME);
    printk(KERN_INFO "KLLM: Loaded model (%zu bytes) and tokenizer (%zu bytes)\n", 
           fw_model->size, fw_tokenizer->size);
    
    return 0;
    
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
fail_alloc:
    return ret;
}

// Module cleanup function
static void __exit kllm_exit(void)
{
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
    
    printk(KERN_INFO "KLLM: Module unloaded\n");
}

module_init(kllm_init);
module_exit(kllm_exit);