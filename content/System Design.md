Title: A Software Engineer's System Design Handbook
Date: 2024-09-08
Category: System Design 
Status: Draft

[TOC]


### Understanding Computer Architecture: The Building Blocks of Modern Systems
Before diving into system design, it’s crucial to first understand the fundamental building blocks of a computer. These components—the disk, RAM, CPU, and caches—are the core of any computing system. They not only form the basis of a computer's operation but also have a significant impact on how we design software and larger distributed systems.

#### Components of a Computer System
1. Disk

The disk is the primary storage device in a computer and plays a crucial role in maintaining persistent data. Persistence means that the information stored on a disk remains intact even after the machine is turned off. Modern disks store data in terabytes (TBs), where 1 terabyte equals 1 trillion bytes. Disks are available in two common forms:

HDD (Hard Disk Drive): HDDs are mechanical storage devices with moving parts. They read and write data using a mechanical arm that moves over a spinning disk. Over time, this mechanical system can degrade, leading to slower performance.

SSD (Solid-State Drive): SSDs, in contrast, do not have any moving parts and are significantly faster than HDDs. They store and retrieve data electronically, similar to RAM, which makes them more efficient. However, they tend to be more expensive than HDDs.

Both HDDs and SSDs are forms of non-volatile storage, meaning that the data is retained even after power loss. SSDs are increasingly replacing HDDs due to their superior speed and reliability.

2. RAM (Random Access Memory)
RAM is used for temporarily storing data that the CPU might need to access quickly. It is much faster than disk storage but smaller in size, typically ranging from 1GB to 128GB. The key advantage of RAM is its speed, enabling data to be written and read in microseconds, which is much faster than writing to disk.

RAM is considered volatile memory because it only retains information as long as the computer is powered on. Once the system is shut down, any data stored in RAM is erased. This is why saving work to a disk is important—so the data persists even after a shutdown. RAM’s role is to hold running applications and active processes, providing the CPU with immediate access to data and instructions.

3. CPU (Central Processing Unit)
The CPU, often referred to as the "brain" of the computer, handles all computation and processing tasks. It is responsible for fetching, decoding, and executing instructions, which are usually stored in RAM. When you run a program, it translates into a set of binary instructions that the CPU reads and acts upon. The CPU works at lightning speeds, performing operations like addition, subtraction, and multiplication in mere milliseconds.

The CPU also manages data transfer between RAM and the disk. For example, if you open a file, the CPU retrieves the necessary data from the disk and loads it into RAM for faster access. In this way, the CPU serves as the primary intermediary between storage devices and memory.

4. Cache
The cache is a small, ultra-fast memory located directly on the CPU. It stores frequently accessed data to prevent the CPU from having to retrieve it from the slower RAM or disk. Most modern CPUs are equipped with multiple levels of cache—L1, L2, and L3—each progressively larger and slower but still much faster than RAM. The cache is crucial for reducing the time it takes for the CPU to access data.

The concept of caching extends beyond computer architecture. For example, web browsers cache web pages to load them faster on subsequent visits by storing commonly used resources like HTML, CSS, and JavaScript files. Similarly, the CPU’s cache helps speed up data access by storing frequently used information close to the processing unit.

#### Moore’s Law
Moore's Law is a prediction made by Gordon Moore, co-founder of Intel, which states that the number of transistors in a CPU doubles approximately every two years. As the number of transistors increases, so does the processing power of CPUs, while the cost of computing continues to decrease. However, in recent years, the pace of this exponential growth has started to slow down, and we are approaching the physical limitations of transistor-based computing.

#### Interaction Between Components
To understand the relationship between the components discussed, it’s helpful to visualize the following:

The CPU fetches instructions from RAM.
If the data the CPU needs is stored in the cache, it retrieves it from there first (this is the fastest route).
If the data is not in the cache, the CPU will check RAM (slower than cache but still fast).
If the data isn’t in RAM, the CPU will read it from the disk, which is much slower compared to RAM and cache.

Each of these components—disk, RAM, CPU, and cache—plays a vital role in system performance. The disk provides persistent storage, RAM offers fast, temporary data access, the CPU executes computations, and the cache ensures the most frequently accessed data is retrieved as quickly as possible.

Understanding how these components interact is fundamental for designing efficient systems, from personal computers to large-scale distributed applications. 