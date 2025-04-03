# VM Setup

Inside the VM:

```bash
# Create a mount point
sudo mkdir -p /mnt/shared

# Mount the shared folder
sudo mount -t 9p -o trans=virtio,version=9p2000.L shared_folder /mnt/shared
```
