#include "throughput_counter.h"
#include <cassert>
#include <chrono>
#include <iostream>

ThroughputCounter::ThroughputCounter(): total_tokens(0), milli_seconds(0), in_recording(false) {}

void ThroughputCounter::print_throughput() {
    assert(milli_seconds > 0);
    std::cout << "Total tokens: " << total_tokens << ", seconds: " << (milli_seconds / 1000.0) << ", throughput: " << total_tokens / (milli_seconds / 1000.0) << std::endl;
}

void ThroughputCounter::start_record() {
    if (!in_recording) {
        last_timestamp = std::chrono::high_resolution_clock::now();
        in_recording = true;
    }
}

void ThroughputCounter::add_record_if_recording(int new_tokens) {
    if (in_recording) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_timestamp);
        milli_seconds += elapsed.count();
        total_tokens += new_tokens;
        in_recording = false;

        start_record();
    }
}

ThroughputCounter& get_global_throughput_counter() {
    static ThroughputCounter global_counter;
    return global_counter;
}
