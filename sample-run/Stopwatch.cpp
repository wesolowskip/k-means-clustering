#include "Stopwatch.h"
#include <iostream>

void Stopwatch::start(string message)
{
    this->message = message;
    begin = chrono::high_resolution_clock::now();
}

void Stopwatch::stop()
{
    auto end = chrono::high_resolution_clock::now();
    auto dur = end - begin;
    auto ms = chrono::duration_cast<chrono::milliseconds>(dur).count();
    auto us = chrono::duration_cast<chrono::microseconds>(dur).count() - ms*1000;
    cout << message << " - " << ms << "ms" << " " << us << "us\n";
}