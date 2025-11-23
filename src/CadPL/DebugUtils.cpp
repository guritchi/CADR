#include <CadPL/DebugUtils.h>

#include <map>
#include <mutex>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>

namespace CadPL {

    struct TraceEvent {
        long long duration = 0;
        std::string text;
    };
    struct EventEntry {
        long long total = 0;
        size_t count = 0;
    };
    static std::mutex mutex;
    static std::vector<TraceEvent> debugEvents;
    static std::map<std::thread::id, std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>>> threadInfo;
    static std::map<std::string, EventEntry> entries;
    static std::map<std::string, size_t> counters;
    static std::map<std::string, std::pair<double, size_t>> averages;

    void Debug::enterThread(const std::string &key)
    {
        auto now = std::chrono::steady_clock::now();
        auto id = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(mutex);
        threadInfo[id][key] = now;
    }
    void Debug::exitThread(const std::string &key)
    {
        auto id = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(mutex);
        threadInfo[id].erase(key);
    }
    std::string Debug::getThreadInfo()
    {
        std::stringstream output;
        // std::lock_guard<std::mutex> lock(mutex);
        auto now = std::chrono::steady_clock::now();
        for (const auto &t : threadInfo) {
            for (const auto &k : t.second) {
                auto duration = (now - k.second).count();
                output << "thread " << t.first << " (" << k.first << "): ";
                printDuration(duration, output);
                output << "\n";
            }
            if (t.second.size() > 1) {
                output << "\n";
            }
        }
        return output.str();
    }
    std::string Debug::getCompileStats()
    {
        std::stringstream output;
        output << "background compile time: " << getDurationNoLock("threadCompileTime") << ", " << getCounterNoLock("threadCreateCount") << " create calls\n";
        output << "main compile time: " << getDurationNoLock("mainCompileTime") << ", " << getCounterNoLock("mainCreateCount") << " create calls\n";
        output << "waitIfCompiling(): " << getDurationNoLock("waitIfCompiling()") << "\n";
        output << "vkCreateGraphicsPipelines(): " << getDurationNoLock("vkCreateGraphicsPipelines()") << "\n";
        auto hit = getCounterNoLock("cacheHitCount");
        auto miss = getCounterNoLock("cacheMissCount");
        auto total = hit + miss;
        if (total > 0) {
            output << "cache hit (" << ((hit * 100) / total) << "%): " << hit << ", miss: " << miss << "\n";
        }
        auto compileRequired = getCounterNoLock("VK_PIPELINE_COMPILE_REQUIRED");
        if (compileRequired > 0) {
            output << "VK_PIPELINE_COMPILE_REQUIRED pipelines: " << compileRequired << "\n";
        }
        output << "Generated GLSL total size: " << getCounterNoLock("glslCodeSize") << "B\n";
        output << "Generated SpirV total size: " << getCounterNoLock("spirvCodeSize") << "B\n";
        output << "Average GLSL compilation: " << getAverageNoLock("glslToSpirV") << "\n";
        return output.str();
    }
    void Debug::getNonblocking(std::string *threadInfo, std::string *compileInfo)
    {
        std::unique_lock lock(mutex, std::defer_lock);
        if (lock.try_lock()) {
            if (threadInfo) {
                std::stringstream output;
                auto now = std::chrono::steady_clock::now();
                for (const auto &t : CadPL::threadInfo) {
                    for (const auto &k : t.second) {
                        auto duration = (now - k.second).count();
                        output << "thread " << t.first << " (" << k.first << "): ";
                        printDuration(duration, output);
                        output << "\n";
                    }
                    if (t.second.size() > 1) {
                        output << "\n";
                    }
                }
                *threadInfo = std::move(output.str());
            }
            if (compileInfo) {
                *compileInfo = std::move(getCompileStats());
            }
        }
    }

    void Debug::log(std::string &&text, const std::string &key, uint64_t duration, bool print)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto &e = entries[key];
        e.total += duration;
        ++e.count;
    }

    void Debug::log(std::string &&text, uint64_t duration, bool print)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto &dst = debugEvents.emplace_back();
        dst.duration = duration;
        dst.text = std::move(text);
        if (print) {
            std::cout << dst.text << ": ";
            Debug::printDuration(dst.duration, std::cout);
            std::cout << '\n';
        }
    }

    void Debug::logAverage(const std::string &key, double duration)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& it = averages[key];
        it.first += duration;
        ++it.second;
    }

    void Debug::increment(const std::string &key, size_t amount)
    {
        std::lock_guard<std::mutex> lock(mutex);
        counters[key] += amount;
    }

    size_t Debug::getCounterNoLock(const std::string &key) {
        return counters[key];
    }
    size_t Debug::getCounter(const std::string &key)
    {
        std::lock_guard<std::mutex> lock(mutex);
        return getCounterNoLock(key);
    }

    std::string Debug::getDurationNoLock(const std::string &key) {
        auto it = entries.find(key);
        if (it == entries.end()) {
            return "";
        }
        std::stringstream str;
        Debug::printDuration(it->second.total, str);
        return str.str();
    }

    std::string Debug::getDuration(const std::string &key)
    {
        std::lock_guard<std::mutex> lock(mutex);
        return getDurationNoLock(key);
    }

    std::string Debug::getAverageNoLock(const std::string &key)
    {
        auto &it = averages[key];
        if (it.second == 0) {
            return "0s";
        }
        std::stringstream output;
        printDuration(it.first / static_cast<double>(it.second), output);
        return output.str();
    }

    void Debug::printDuration(long long nanoseconds, std::ostream &os)
    {
        if (nanoseconds == 0) {
            os << " 0s";
        }
        else if (nanoseconds < static_cast<uint64_t>(1e3)) {
            os << nanoseconds << "ns";
        }
        else if (nanoseconds < static_cast<uint64_t>(1e6)) {
            os << nanoseconds / static_cast<uint64_t>(1e3) << "us";
        }
        else if (nanoseconds <  static_cast<uint64_t>(1e9)) {
            os << nanoseconds / static_cast<uint64_t>(1e6) << "ms";
        }
        else {
            os << nanoseconds / static_cast<uint64_t>(1e9) << "s";
        }
    }

    void Debug::printDuration(double seconds, std::ostream &os)
    {
        if (seconds == 0.0) {
            os << " 0s";
        }
        else if (seconds < 1e-6) {
            os << seconds * 1e9 << "ns";
        }
        else if (seconds < 1e-3) {
            os << seconds * 1e6 << "us";
        }
        else if (seconds < 1.0) {
            os << seconds * 1e3 << "ms";
        }
        else {
            os << seconds << "s";
        }
    }

    void Debug::logToFile(const std::stringstream &text, const std::string_view name)
    {
        // for (const auto &e : debugEvents) {
        //     std::stringstream str;
        //     str << e.text;
        //     if (e.duration > 0) {
        //         str << ": ";
        //         Debug::printDuration(e.duration, str);
        //     }
        //     std::cout << str.str() << '\n';
        // }

        std::ostringstream filename;
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        filename << "log_";
        if (!name.empty()) {
            filename << name << "_";
        }
        filename << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S") << ".txt";

        std::cout << "logging to " << filename.str() << std::endl;
        std::cout << text.str() << std::endl;

        std::ofstream file(filename.str(), std::ios::out);
        if (file.is_open()) {
            file << text.str();
            file.close();
        }
        else {
            std::cerr << "Error writing file: " << std::strerror(errno) << '\n';
        }
    }
}


