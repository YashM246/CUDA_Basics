# Makefiles: Automating the Build Process

## What is a Makefile?

A **Makefile** is a script that automates compiling and building your project. Instead of typing long compiler commands, you just type `make`.

```
Without Makefile:
cl /O2 /W4 main.c utils.c math.c /Fe:program.exe

With Makefile:
make
```

---

## Why Use Makefiles?

1. **Automation** - One command builds everything
2. **Incremental builds** - Only recompiles changed files (saves time)
3. **Consistency** - Everyone builds the same way
4. **Documentation** - Build process is clearly defined
5. **Dependency management** - Handles file relationships

---

## Basic Syntax

A Makefile consists of **rules**:

```makefile
target: dependencies
	recipe (commands)
```

- **target** - What you want to create (usually a file)
- **dependencies** - Files the target depends on
- **recipe** - Commands to create the target (MUST use TAB, not spaces!)

### Example

```makefile
program.exe: main.c utils.c
	cl main.c utils.c /Fe:program.exe
```

Run with: `make program.exe` or just `make` (runs first target)

---

## Important: Tabs vs Spaces

**Recipes MUST start with a TAB character, not spaces!**

```makefile
# CORRECT (TAB before cl)
program.exe: main.c
	cl main.c /Fe:program.exe

# WRONG (spaces before cl) - Will fail!
program.exe: main.c
    cl main.c /Fe:program.exe
```

This is a common source of errors. Configure your editor to show whitespace.

---

## Variables

Variables make Makefiles maintainable:

```makefile
# Define variables
CC = cl
CFLAGS = /O2 /W4
TARGET = program.exe
SOURCES = main.c utils.c math.c

# Use variables with $(NAME)
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) /Fe:$(TARGET)
```

### Common Variable Names

| Variable | Purpose | Example |
|----------|---------|---------|
| `CC` | C compiler | `cl`, `gcc` |
| `CXX` | C++ compiler | `cl`, `g++` |
| `NVCC` | CUDA compiler | `nvcc` |
| `CFLAGS` | C compiler flags | `/O2 /W4` |
| `CXXFLAGS` | C++ compiler flags | `/O2 /W4` |
| `LDFLAGS` | Linker flags | `/link /LIBPATH:lib` |
| `TARGET` | Output file | `program.exe` |
| `SOURCES` | Source files | `main.c utils.c` |
| `OBJECTS` | Object files | `main.obj utils.obj` |

---

## Pattern Rules and Automatic Variables

### Automatic Variables

| Variable | Meaning |
|----------|---------|
| `$@` | Target name |
| `$<` | First dependency |
| `$^` | All dependencies |
| `$*` | Stem (matched by %) |

### Pattern Rule Example

```makefile
# Compile any .c file to .obj
%.obj: %.c
	$(CC) /c $< /Fo:$@

# Usage:
# main.obj depends on main.c
# $< = main.c, $@ = main.obj
```

---

## Complete Makefile Example (C Project)

```makefile
# ===========================================
# Makefile for C Project (Windows/MSVC)
# ===========================================

# Compiler settings
CC = cl
CFLAGS = /O2 /W4

# Project files
TARGET = program.exe
SOURCES = main.c utils.c math.c
OBJECTS = main.obj utils.obj math.obj

# ===========================================
# Rules
# ===========================================

# Default target
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) /Fe:$(TARGET)

# Compile .c to .obj
%.obj: %.c
	$(CC) /c $(CFLAGS) $< /Fo:$@

# Clean build artifacts
clean:
	del /Q *.obj *.exe 2>nul

# Rebuild from scratch
rebuild: clean all

# Phony targets (not actual files)
.PHONY: all clean rebuild
```

### Usage

```powershell
make          # Build the project
make clean    # Remove compiled files
make rebuild  # Clean and rebuild
```

---

## CUDA Makefile Example

```makefile
# ===========================================
# Makefile for CUDA Project (Windows)
# ===========================================

# Compilers
NVCC = nvcc
CC = cl

# Flags
NVCC_FLAGS = -arch=sm_89 -O3
CFLAGS = /O2

# Project files
TARGET = cuda_program.exe
CU_SOURCES = kernel.cu
C_SOURCES = main.c
OBJECTS = kernel.obj main.obj

# ===========================================
# Rules
# ===========================================

all: $(TARGET)

# Link everything
$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET)

# Compile CUDA files
%.obj: %.cu
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

# Compile C files
%.obj: %.c
	$(CC) /c $(CFLAGS) $< /Fo:$@

clean:
	del /Q *.obj *.exe *.exp *.lib 2>nul

.PHONY: all clean
```

---

## Phony Targets

**Phony targets** are not actual files - they're just names for commands:

```makefile
.PHONY: all clean rebuild test

all: program.exe

clean:
	del /Q *.obj *.exe

test:
	.\program.exe --test

rebuild: clean all
```

Without `.PHONY`, if a file named `clean` existed, `make clean` wouldn't run.

---

## Dependencies and Incremental Builds

Make only rebuilds what's necessary:

```makefile
program.exe: main.obj utils.obj
	cl main.obj utils.obj /Fe:program.exe

main.obj: main.c utils.h
	cl /c main.c

utils.obj: utils.c utils.h
	cl /c utils.c
```

**If you change:**
- `utils.c` → Rebuilds `utils.obj` and `program.exe`
- `utils.h` → Rebuilds both `.obj` files and `program.exe`
- `main.c` → Rebuilds only `main.obj` and `program.exe`

---

## Conditional Logic

```makefile
# Detect OS
ifeq ($(OS),Windows_NT)
    RM = del /Q
    EXE = .exe
else
    RM = rm -f
    EXE =
endif

TARGET = program$(EXE)

clean:
	$(RM) *.obj *.o $(TARGET)
```

---

## Multiple Targets

```makefile
# Build multiple programs
all: program1.exe program2.exe

program1.exe: prog1.c
	cl prog1.c /Fe:program1.exe

program2.exe: prog2.c
	cl prog2.c /Fe:program2.exe

clean:
	del /Q *.exe
```

---

## Include Other Makefiles

```makefile
# Include shared configuration
include config.mk

# Or conditionally
-include local.mk  # Dash means "ignore if missing"
```

---

## Debug vs Release Builds

```makefile
# Default to release
BUILD ?= release

ifeq ($(BUILD),debug)
    CFLAGS = /Od /Zi /DDEBUG
    NVCC_FLAGS = -g -G -DDEBUG
else
    CFLAGS = /O2 /DNDEBUG
    NVCC_FLAGS = -O3 -DNDEBUG
endif

program.exe: main.c
	cl $(CFLAGS) main.c /Fe:program.exe
```

**Usage:**
```powershell
make                  # Release build
make BUILD=debug      # Debug build
```

---

## Common Make Commands

| Command | Description |
|---------|-------------|
| `make` | Build default target (first one) |
| `make target` | Build specific target |
| `make -n` | Dry run (show commands without executing) |
| `make -B` | Force rebuild all |
| `make -j4` | Parallel build (4 jobs) |
| `make -f MyMakefile` | Use different makefile |
| `make VAR=value` | Override variable |

---

## Windows vs Linux Differences

| Aspect | Windows (nmake/make) | Linux (make) |
|--------|---------------------|--------------|
| Delete | `del /Q file` | `rm -f file` |
| Executable | `program.exe` | `program` |
| Path separator | `\` | `/` |
| Null device | `2>nul` | `2>/dev/null` |
| Object extension | `.obj` | `.o` |

---

## Make on Windows

### Option 1: nmake (comes with MSVC)
```powershell
nmake /f Makefile
```

### Option 2: GNU Make (via chocolatey or MinGW)
```powershell
# Install via chocolatey
choco install make

# Then use
make
```

### Option 3: Use in Developer PowerShell
Make sure you're in Developer PowerShell where `cl` is available.

---

## CMake: A Modern Alternative

For larger projects, consider **CMake** - a cross-platform build system generator:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyProject)

add_executable(program main.c utils.c)
```

```powershell
mkdir build && cd build
cmake ..
cmake --build .
```

CMake generates Makefiles (Linux) or Visual Studio projects (Windows).

---

## Quick Reference

### Minimal Makefile
```makefile
program.exe: main.c
	cl main.c /Fe:program.exe
```

### Standard Structure
```makefile
CC = cl
CFLAGS = /O2 /W4
TARGET = program.exe
SOURCES = main.c utils.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) /Fe:$(TARGET)

clean:
	del /Q *.obj *.exe

.PHONY: all clean
```

### CUDA Structure
```makefile
NVCC = nvcc
ARCH = -arch=sm_89
TARGET = cuda_app.exe

all: $(TARGET)

$(TARGET): main.cu kernel.cu
	$(NVCC) $(ARCH) $^ -o $@

clean:
	del /Q *.exe *.obj *.exp *.lib

.PHONY: all clean
```

---

## Summary

1. **Makefiles automate builds** - No more typing long commands
2. **Use TABs for recipes** - Not spaces (common error!)
3. **Variables** make maintenance easier (`CC`, `CFLAGS`, etc.)
4. **Automatic variables**: `$@` (target), `$<` (first dep), `$^` (all deps)
5. **Incremental builds** - Only recompiles what changed
6. **Phony targets** for commands like `clean`, `all`, `test`
7. **Works with CUDA** - Just use `nvcc` instead of `cl`
