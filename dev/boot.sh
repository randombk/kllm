#!/bin/bash
qemu-system-x86_64 -m 4096 -smp 4 -drive file=disk.img,format=qcow2 \
    -virtfs local,path=..,mount_tag=shared_folder,security_model=mapped-file,id=host_share \
    -enable-kvm