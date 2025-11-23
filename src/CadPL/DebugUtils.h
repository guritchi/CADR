#pragma once

#include <ostream>
#include <string>

namespace CadPL {

    class CADPL_EXPORT Debug {
    public:
        static void enterThread(const std::string &key);
        static void exitThread(const std::string &key);
        static std::string getThreadInfo();
        static std::string getCompileStats();
        static void getNonblocking(std::string *threadInfo, std::string *compileInfo);
        static void log(std::string &&text, const std::string &key, uint64_t duration, bool print = false);
        static void log(std::string &&text, uint64_t duration, bool print = false);
        static void logAverage(const std::string &key, double duration);
        static void increment(const std::string &key, size_t amount = 1);
        static std::string getDurationNoLock(const std::string &key);
        static size_t getCounterNoLock(const std::string &key);
        static std::string getDuration(const std::string &key);
        static std::string getAverageNoLock(const std::string &key);
        static size_t getCounter(const std::string &key);
        static void printDuration(long long nanoseconds, std::ostream &os);
        static void printDuration(double seconds, std::ostream &os);
        static void logToFile(const std::stringstream &text, std::string_view name);
    };

}