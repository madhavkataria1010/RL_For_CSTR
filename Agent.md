## CUDA Server Access

Use SSH to connect to the CUDA server:

```bash
ssh root@10.36.16.15
```

An SSH key is already configured, so no password should be required.

Use `/madhav` as the working directory. Sync your code there (for example with `rsync`) and run your jobs from that path.

The server has an NVIDIA A5000 GPU with 24 GB VRAM, and you can use it freely.