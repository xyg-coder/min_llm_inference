.PHONY: all build test all_test build_flag_on build_flag_off clean

all: build test

build:
	mkdir -p build
	cd build && cmake -DDEBUG_MODE=OFF .. && make

debug_build:
	mkdir -p build
	cd build && cmake -DDEBUG_MODE=ON .. && make

test:
	cd build && ctest

all_test: build_async_flag_on build_async_flag_off

build_async_flag_on:
	rm -rf build
	mkdir -p build
	cd build && cmake -DUSE_ASYNC_ALLOC=ON .. && make && ctest

build_async_flag_off:
	rm -rf build
	mkdir -p build
	cd build && cmake -DUSE_ASYNC_ALLOC=OFF .. && make && ctest

clean:
	rm -rf build