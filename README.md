# System-wide_Atomic

A sample code which reproduces erroneaous behaviors of multi GeForce RTX-4090 systems.
This code produces non-deterministic results on multi 4090 systems whereas it runs correctly on single 4090 or on multi 3090 systems.

My experience is limitted to docker container environments ( I cannot afford my own multi-4090 system ) and the phenomena can be different when it runs directly on a physical server.


    Compilation:
    
    $ nvcc -arch=sm_60 atomic-01.cu
    
    Execution example:
    
    # single 4090 execution
    $ CUDA_VISIBLE_DEVICES=0 ./a.out 
    atomic-01.cu 
    5000000 
    
    # dual 4090 execution
    $ CUDA_VISIBLE_DEVICES=0,1 ./a.out 
    atomic-01.cu 
    5000269
