# Debuggers: Finding and Fixing Bugs

## What is a Debugger?

A **debugger** is a tool that lets you:
- **Pause** program execution at specific points
- **Inspect** variable values and memory
- **Step through** code line by line
- **Find** where and why your program crashes

Without a debugger, you're limited to `printf` statements. With a debugger, you can see exactly what's happening inside your program.

---

## Common Debuggers

| Debugger | Platform | Languages | IDE Integration |
|----------|----------|-----------|-----------------|
| **GDB** | Linux, Mac, Windows (MinGW) | C, C++, Fortran | VS Code, CLion, Emacs |
| **LLDB** | Mac, Linux | C, C++, Objective-C | Xcode, VS Code |
| **Visual Studio Debugger** | Windows | C, C++, C# | Visual Studio |
| **CUDA-GDB** | Linux | CUDA | VS Code, Nsight |
| **Nsight** | Windows, Linux | CUDA | Visual Studio, Standalone |

---

## Key Debugging Concepts

### 1. Breakpoints
Pause execution at a specific line:
```
Line 42: x = calculate();  ← BREAKPOINT (program pauses here)
Line 43: y = x + 10;       ← You can inspect 'x' before this runs
```

### 2. Stepping
- **Step Over (F10)** - Execute current line, move to next
- **Step Into (F11)** - Enter function call, debug inside it
- **Step Out (Shift+F11)** - Finish current function, return to caller
- **Continue (F5)** - Run until next breakpoint

### 3. Watch Variables
Monitor specific variables as you step through code.

### 4. Call Stack
See the chain of function calls that led to current position:
```
main()
  └── processData()
        └── calculateSum()  ← You are here
```

### 5. Memory Inspection
View raw memory, useful for pointer debugging.

---

## Compiling for Debugging

**Debug symbols** map machine code back to source code. Without them, debuggers can't show variable names or line numbers.

### MSVC (Windows)
```powershell
cl /Zi /Od program.c /Fe:debug.exe
```
- `/Zi` - Generate debug info
- `/Od` - Disable optimization (code matches source)

### GCC/Clang (Linux/Mac)
```bash
gcc -g -O0 program.c -o debug
```
- `-g` - Generate debug info
- `-O0` - No optimization

### NVCC (CUDA)
```powershell
nvcc -g -G kernel.cu -o debug.exe
```
- `-g` - Debug host code
- `-G` - Debug device code (GPU)

---

# GDB (GNU Debugger)

The most widely used debugger for C/C++ on Linux and Unix systems.

## Installing GDB

### Linux (Ubuntu/Debian)
```bash
sudo apt install gdb
```

### Mac
```bash
brew install gdb
```

### Windows
- Comes with MinGW/MSYS2
- Or use Windows Subsystem for Linux (WSL)

---

## GDB Basic Usage

### Starting GDB

```bash
# Compile with debug symbols
gcc -g -O0 program.c -o program

# Start GDB with your program
gdb ./program

# Or attach to running process
gdb -p <process_id>
```

### Essential GDB Commands

| Command | Short | Description |
|---------|-------|-------------|
| `run` | `r` | Start program execution |
| `break <location>` | `b` | Set breakpoint |
| `continue` | `c` | Continue to next breakpoint |
| `next` | `n` | Step over (execute line) |
| `step` | `s` | Step into (enter function) |
| `finish` | `fin` | Step out (finish function) |
| `print <expr>` | `p` | Print variable/expression |
| `display <expr>` | `disp` | Print every time you stop |
| `info locals` | | Show local variables |
| `info args` | | Show function arguments |
| `backtrace` | `bt` | Show call stack |
| `frame <n>` | `f` | Select stack frame |
| `list` | `l` | Show source code |
| `quit` | `q` | Exit GDB |

---

## GDB Walkthrough Example

### Sample Program (buggy.c)
```c
#include <stdio.h>

int divide(int a, int b) {
    return a / b;  // Bug: no check for b == 0
}

int main() {
    int x = 10;
    int y = 0;
    int result = divide(x, y);  // Crash!
    printf("Result: %d\n", result);
    return 0;
}
```

### Debug Session
```bash
$ gcc -g -O0 buggy.c -o buggy
$ gdb ./buggy
```

```gdb
(gdb) break main          # Set breakpoint at main
Breakpoint 1 at 0x1234: file buggy.c, line 8.

(gdb) run                  # Start program
Starting program: ./buggy
Breakpoint 1, main () at buggy.c:8
8	    int x = 10;

(gdb) next                 # Step over
9	    int y = 0;

(gdb) next
10	    int result = divide(x, y);

(gdb) print x              # Check variable
$1 = 10

(gdb) print y
$2 = 0

(gdb) step                 # Step INTO divide()
divide (a=10, b=0) at buggy.c:4
4	    return a / b;

(gdb) print a
$3 = 10

(gdb) print b              # Found the bug!
$4 = 0

(gdb) backtrace            # See call stack
#0  divide (a=10, b=0) at buggy.c:4
#1  main () at buggy.c:10

(gdb) quit
```

---

## GDB Breakpoint Commands

### Setting Breakpoints
```gdb
break main              # Break at function
break 42                # Break at line 42
break file.c:42         # Break at line in specific file
break divide if b == 0  # Conditional breakpoint
```

### Managing Breakpoints
```gdb
info breakpoints        # List all breakpoints
disable 1               # Disable breakpoint #1
enable 1                # Enable breakpoint #1
delete 1                # Remove breakpoint #1
clear main              # Remove breakpoint at main
```

---

## GDB Print Commands

```gdb
print x                 # Print variable
print x + y             # Print expression
print *ptr              # Dereference pointer
print arr[5]            # Array element
print sizeof(x)         # Size of variable
print (float)x          # Cast and print

print/x x               # Print in hexadecimal
print/d x               # Print as decimal
print/t x               # Print in binary
print/c x               # Print as character

# Print array
print *arr@10           # Print first 10 elements
```

---

## GDB Memory Commands

```gdb
# Examine memory: x/[count][format][size] address
x/10x &arr              # 10 hex words starting at arr
x/20b ptr               # 20 bytes at ptr
x/s str                 # Print string at str
x/i $pc                 # Disassemble instruction at PC

# Format: x=hex, d=decimal, s=string, i=instruction
# Size: b=byte, h=halfword, w=word, g=giant (8 bytes)
```

---

## GDB Watchpoints

Watchpoints pause when a variable **changes**:

```gdb
watch x                 # Break when x changes
watch *ptr              # Break when memory at ptr changes
rwatch x                # Break when x is read
awatch x                # Break on read or write

info watchpoints        # List watchpoints
```

---

## GDB with Core Dumps

When a program crashes, it can create a **core dump** (memory snapshot):

```bash
# Enable core dumps
ulimit -c unlimited

# Run program (crashes and creates core file)
./program
Segmentation fault (core dumped)

# Debug the core dump
gdb ./program core
```

```gdb
(gdb) backtrace         # See where it crashed
#0  divide (a=10, b=0) at buggy.c:4
#1  main () at buggy.c:10

(gdb) frame 0           # Go to crash location
(gdb) list              # See source code
(gdb) print b           # Inspect variables
$1 = 0
```

---

## GDB Text User Interface (TUI)

GDB has a built-in visual mode:

```bash
gdb -tui ./program
```

Or toggle in GDB:
```gdb
(gdb) tui enable        # Enable TUI
(gdb) tui disable       # Disable TUI
(gdb) layout src        # Show source code
(gdb) layout asm        # Show assembly
(gdb) layout split      # Show both
(gdb) layout regs       # Show registers
```

---

## GDB Init File (.gdbinit)

Create `~/.gdbinit` for custom settings:

```gdb
# Pretty print structures
set print pretty on

# Print array indexes
set print array-indexes on

# History
set history save on
set history size 10000

# Don't ask for confirmation
set confirm off

# Better output
set pagination off
```

---

# Visual Studio Debugger (Windows)

The debugger built into Visual Studio and available via VS Code.

## Using with MSVC

```powershell
# Compile with debug info
cl /Zi /Od program.c /Fe:debug.exe

# Debug with Visual Studio
devenv debug.exe
```

## Key Features

- **Breakpoints** - Click in the gutter to set
- **Data Tips** - Hover over variables to see values
- **Watch Window** - Track specific variables
- **Call Stack Window** - See function chain
- **Locals Window** - See all local variables
- **Memory Window** - View raw memory
- **Disassembly** - See assembly code

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Start Debugging | F5 |
| Stop Debugging | Shift+F5 |
| Step Over | F10 |
| Step Into | F11 |
| Step Out | Shift+F11 |
| Toggle Breakpoint | F9 |
| Run to Cursor | Ctrl+F10 |

---

# CUDA Debugging

## CUDA-GDB (Linux)

Extension of GDB for CUDA programs:

```bash
# Compile with debug flags
nvcc -g -G kernel.cu -o kernel

# Debug with cuda-gdb
cuda-gdb ./kernel
```

### CUDA-specific Commands
```gdb
info cuda threads       # List CUDA threads
info cuda blocks        # List CUDA blocks
info cuda kernels       # List running kernels

cuda thread (0,0,0)     # Switch to thread (0,0,0)
cuda block (1,0,0)      # Switch to block (1,0,0)

print threadIdx.x       # Print thread index
print blockIdx.x        # Print block index
print blockDim.x        # Print block dimension
```

## Nsight (Windows/Linux)

NVIDIA's visual debugger for CUDA:

1. **Nsight Visual Studio Edition** - Integrates with Visual Studio
2. **Nsight Eclipse Edition** - For Linux
3. **Nsight Compute** - Kernel profiling
4. **Nsight Systems** - System-wide profiling

### Setup in Visual Studio
1. Install CUDA Toolkit (includes Nsight)
2. Open project in Visual Studio
3. Use **Nsight → Start CUDA Debugging**

---

# VS Code Debugging

## Setup for C/C++

### 1. Install Extensions
- **C/C++** (Microsoft)
- **CodeLLDB** (for LLDB)

### 2. Create launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug (GDB)",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/program.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        },
        {
            "name": "Debug (MSVC)",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/program.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "console": "integratedTerminal"
        }
    ]
}
```

### 3. Set Breakpoints
Click in the gutter (left of line numbers) to set breakpoints.

### 4. Start Debugging
Press F5 or use Run → Start Debugging

---

# Common Debugging Scenarios

## 1. Segmentation Fault (Crash)

```gdb
(gdb) run
Program received signal SIGSEGV, Segmentation fault.
0x0000555555555160 in process (ptr=0x0) at main.c:15

(gdb) backtrace
#0  process (ptr=0x0) at main.c:15    ← NULL pointer!
#1  main () at main.c:25

(gdb) print ptr
$1 = (int *) 0x0                       ← Confirmed NULL
```

**Fix:** Add NULL check before using pointer.

## 2. Infinite Loop

```gdb
(gdb) run
^C                                     # Ctrl+C to interrupt
Program received signal SIGINT

(gdb) backtrace                        # See where stuck
(gdb) print i                          # Check loop variable
(gdb) print limit                      # Check condition
```

## 3. Wrong Output

```gdb
(gdb) break calculate
(gdb) run
(gdb) step                             # Step through function
(gdb) print intermediate_value         # Check calculations
(gdb) next
(gdb) print result                     # Check result
```

## 4. Memory Corruption

```gdb
(gdb) watch *ptr                       # Watch for changes
(gdb) run
Hardware watchpoint hit: *ptr
Old value = 42
New value = 0                          # Something zeroed it!

(gdb) backtrace                        # See what caused it
```

---

# Debugging Tips

## 1. Reproduce Consistently
Make sure you can trigger the bug reliably before debugging.

## 2. Isolate the Problem
Use binary search with breakpoints to narrow down location.

## 3. Check Assumptions
Print values you "know" are correct - they might not be.

## 4. Read Error Messages
Compiler warnings and runtime errors often point directly to bugs.

## 5. Rubber Duck Debugging
Explain the code line-by-line. Often reveals the bug.

## 6. Debug Build vs Release Build
- Always debug with optimization disabled (`-O0` or `/Od`)
- Some bugs only appear in release builds (timing, optimization)

## 7. Use Version Control
`git bisect` can help find when a bug was introduced.

---

# Quick Reference

## GDB Commands
```gdb
gdb ./program           # Start GDB
run                     # Run program
break main              # Set breakpoint
continue                # Continue execution
next                    # Step over
step                    # Step into
print var               # Print variable
backtrace               # Show call stack
quit                    # Exit
```

## Compile for Debug
```bash
# GCC
gcc -g -O0 file.c -o debug

# MSVC
cl /Zi /Od file.c /Fe:debug.exe

# NVCC
nvcc -g -G file.cu -o debug
```

## VS Code Shortcuts
| Action | Shortcut |
|--------|----------|
| Start Debug | F5 |
| Step Over | F10 |
| Step Into | F11 |
| Step Out | Shift+F11 |
| Toggle Breakpoint | F9 |

---

## Summary

1. **Debuggers** let you pause, inspect, and step through code
2. **Compile with debug flags** (`-g`, `/Zi`) for symbol info
3. **GDB** is the standard debugger for C/C++ on Linux
4. **Key commands**: `break`, `run`, `next`, `step`, `print`, `backtrace`
5. **Breakpoints** pause at specific lines
6. **Watchpoints** pause when variables change
7. **CUDA-GDB** and **Nsight** for GPU debugging
8. **VS Code** integrates with GDB/LLDB/MSVC debuggers
