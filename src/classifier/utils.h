//! MIT License
//!
//! Copyright (c) 2025 Santos O. G., Helen C. S. C. Lima,
//! Copyright (c) 2009 Jr, C. N. S. Freitas
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>

using namespace std;

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

#define DEBUG_WARN(text) \
    std::cout << YELLOW << "[WARN] " << text << RESET << std::endl

#define DEBUG_ERR(text) \
    std::cout << RED << "[ERR] " << text << RESET << std::endl

#define DEBUG_DBG(text) \
    std::cout << BLUE << "[DBG] " << text << RESET << std::endl

#define DEBUG_PRINT(type, text) \
    do { \
        if (type == "warn") { \
            DEBUG_WARN(text); \
        } else if (type == "err") { \
            DEBUG_ERR(text); \
        } else if (type == "dbg") { \
            DEBUG_DBG(text); \
        } \
    } while(0)

string lowercase(string str);

int str2int(const string &value);

void getDatasetsProfile(const string &trainingFile,
                        const string &testFile,
                        unsigned int &numberOfTrainingExamples,
                        unsigned int &numberOfTestExamples,
                        unsigned int &numberOfAttributes);

#endif // UTILS_H
