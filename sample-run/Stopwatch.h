#ifndef KMEANSCLUSTERING_STOPWATCH_H
#define KMEANSCLUSTERING_STOPWATCH_H

#include <chrono>
#include <string>

using namespace std;


class Stopwatch
{
    string message;
    chrono::time_point<chrono::steady_clock> begin;
public:
    void start(string message);
    void stop();
};


#endif //KMEANSCLUSTERING_STOPWATCH_H
