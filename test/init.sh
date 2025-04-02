#!/bin/bash
if [ ! -f disk.img ]; then
    qemu-img create -f qcow2 disk.img 10G
    wget https://cdimage.debian.org/debian-cd/current/amd64/iso-cd/debian-12.10.0-amd64-netinst.iso
    qemu-system-x86_64 -m 2048 -drive file=disk.img,format=qcow2 -cdrom debian-12.10.0-amd64-netinst.iso -boot d -enable-kvm
fi
