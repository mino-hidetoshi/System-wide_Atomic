# System-wide_Atomic

A sample code which reproduces a erroneaous behavior of multi GeForce RTX-4090 systems.
This code produces non-deterministic results on multi 4090 systems, it runs correctly on single 4090 or on multi 3090 systems though.

My experience is limitted to docker ontainer environments ( I cannot afford my own multi-4090 system ) and the phenomena can be different when it run directly on a physical server.


    Compilation:
    
    $ nvcc -arch=sm_60 atomic-01.cu
    
    Execution example:
    
    $ CUDA_VISIBLE_DEVICES=0 ./a.out 
    atomic-01.cu 
    5000000 
    
    $ CUDA_VISIBLE_DEVICES=0,1 ./a.out 
    atomic-01.cu 
    5000269
