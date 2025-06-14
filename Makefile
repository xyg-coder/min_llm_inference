.PHONY: all build test all_test build_flag_on build_flag_off clean

all: build test

build:
	mkdir -p build
	cd build && cmake -DDEBUG_MODE=OFF -DBUILD_TESTS=OFF .. && make

debug_build:
	mkdir -p build
	cd build && cmake -DDEBUG_MODE=ON .. && make

test:
	cd build && ctest

profile:
	mkdir -p build
	cd build && cmake -DUSE_ASYNC_ALLOC=OFF -DDEBUG_MODE=OFF -DBUILD_TESTS=OFF .. && make && nsys profile ./to_profile

all_test: test_async_flag_on test_async_flag_off

test_async_flag_on:
	rm -rf build
	mkdir -p build
	cd build && cmake -DUSE_ASYNC_ALLOC=ON -DDEBUG_MODE=OFF -DBUILD_TESTS=ON .. && make && ctest

test_async_flag_off:
	rm -rf build
	mkdir -p build
	cd build && cmake -DUSE_ASYNC_ALLOC=OFF -DDEBUG_MODE=OFF -DBUILD_TESTS=ON .. && make && ctest

clean:
	rm -rf build