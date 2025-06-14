#pragma once

#include <chrono>

class ThroughputCounter {
public:
    ThroughputCounter();
    void print_throughput();
    void start_record();
    void add_record_if_recording(int new_tokens);
private:
    int total_tokens;
    long long milli_seconds;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_timestamp;
    bool in_recording;
};

ThroughputCounter& get_global_throughput_counter();
